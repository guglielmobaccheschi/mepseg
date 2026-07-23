"""Elaborazione in streaming per nuvole troppo grandi per la RAM.

Per file E57 da centinaia di milioni di punti la nuvola non puo' essere
caricata per intero: questo modulo legge il file a blocchi con l'API
low-level di libe57, accumula un sottocampionamento voxel globale
(centroide per voxel) e, dopo la segmentazione sulla nuvola ridotta,
riproietta le etichette a piena risoluzione scrivendo il LAS a blocchi.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger("mepseg")

CAMPI_XYZ = ("cartesianX", "cartesianY", "cartesianZ")
CAMPI_RGB = ("colorRed", "colorGreen", "colorBlue")


def conta_punti_e57(percorso: str | Path) -> int:
    """Punti totali dichiarati nell'header, senza leggere i dati."""
    import pye57

    e57 = pye57.E57(str(percorso))
    return sum(e57.get_header(i).point_count for i in range(e57.scan_count))


def leggi_e57_a_blocchi(
    percorso: str | Path, dimensione_blocco: int = 5_000_000
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Genera blocchi ``(xyz (N,3) float64, rgb (N,3) uint8 | None)``.

    Le coordinate sono gia' rototraslate. Il colore (``colorRed/Green/Blue``)
    e' letto quando presente e allineato ai punti; ``None`` se la scansione
    non ha colore. Usa CompressedVectorReader di libe57: i buffer hanno
    capienza ``dimensione_blocco`` e ogni ``read()`` restituisce il blocco
    successivo. I punti non validi (cartesianInvalidState != 0) sono scartati.
    """
    import pye57
    from pye57 import libe57

    e57 = pye57.E57(str(percorso))
    for indice in range(e57.scan_count):
        header = e57.get_header(indice)
        campi = set(header.point_fields)
        if not set(CAMPI_XYZ) <= campi:
            raise ValueError(
                f"Scansione {indice}: coordinate cartesiane assenti "
                f"(campi: {sorted(campi)}). Nuvole in coordinate sferiche "
                "non sono supportate in streaming."
            )
        da_leggere = list(CAMPI_XYZ)
        if "cartesianInvalidState" in campi:
            da_leggere.append("cartesianInvalidState")
        ha_colore = set(CAMPI_RGB) <= campi
        if ha_colore:
            da_leggere += list(CAMPI_RGB)

        buffers = libe57.VectorSourceDestBuffer()
        arrays: dict[str, np.ndarray] = {}
        for campo in da_leggere:
            arr, buf = e57.make_buffer(campo, dimensione_blocco)
            arrays[campo] = arr
            buffers.append(buf)

        # rototraslazione della scansione (se presente)
        try:
            rotazione = header.rotation_matrix
            traslazione = header.translation
        except Exception:
            rotazione, traslazione = np.eye(3), np.zeros(3)

        lettore = header.points.reader(buffers)
        while True:
            n = lettore.read()
            if n <= 0:
                break
            xyz = np.column_stack(
                [arrays[c][:n] for c in CAMPI_XYZ]
            ).astype(np.float64)
            if "cartesianInvalidState" in arrays:
                valido = arrays["cartesianInvalidState"][:n] == 0
            else:
                valido = np.ones(len(xyz), dtype=bool)
            xyz = xyz[valido]
            if len(xyz) == 0:
                continue
            rgb = None
            if ha_colore:
                rgb = np.column_stack(
                    [arrays[c][:n] for c in CAMPI_RGB]
                ).astype(np.float64)[valido]
                if rgb.size and rgb.max() > 255:  # E57 puo' dare RGB a 16 bit
                    rgb = rgb / 257.0
                rgb = rgb.round().clip(0, 255).astype(np.uint8)
            yield xyz @ rotazione.T + traslazione, rgb
        lettore.close()


_OFFSET_VOXEL = 1 << 20  # indici voxel ammessi in (-2^20, 2^20): +-20 km a 2 cm


class AccumulatoreVoxel:
    """Sottocampionamento voxel incrementale su griglia globale.

    Mantiene per ogni voxel la somma dei punti e il conteggio; il risultato
    e' il centroide per voxel. La griglia e' ancorata all'origine globale,
    quindi blocchi diversi si fondono in modo coerente. Le chiavi voxel 3D
    sono impacchettate in un singolo int64 (21 bit per asse): l'unificazione
    su chiavi 1D e' molto piu' veloce che su righe 3D.
    """

    def __init__(self, voxel: float):
        self.voxel = float(voxel)
        self._chiavi = np.empty(0, dtype=np.int64)
        self._somme = np.empty((0, 3))
        self._conteggi = np.empty(0, dtype=np.int64)
        # somma colore per voxel (float64 per non traboccare su molti punti);
        # None finche' non arriva il primo blocco con colore
        self._somme_colore: np.ndarray | None = None

    @property
    def n_voxel(self) -> int:
        return len(self._chiavi)

    def _impacchetta(self, punti: np.ndarray) -> np.ndarray:
        indici = np.floor(punti / self.voxel).astype(np.int64) + _OFFSET_VOXEL
        if indici.min() < 0 or indici.max() >= (1 << 21):
            raise ValueError(
                "Coordinate fuori scala per la griglia voxel: la nuvola "
                "sembra essere in millimetri o georeferenziata molto lontano "
                "dall'origine. Riportarla vicino all'origine in metri."
            )
        return (indici[:, 0] << 42) | (indici[:, 1] << 21) | indici[:, 2]

    def aggiungi(self, punti: np.ndarray, colori: np.ndarray | None = None) -> None:
        chiavi = np.concatenate([self._chiavi, self._impacchetta(punti)])
        somme = np.vstack([self._somme, punti])
        conteggi = np.concatenate(
            [self._conteggi, np.ones(len(punti), dtype=np.int64)]
        )
        con_colore = colori is not None
        if con_colore:
            # le somme colore per voxel gia' accumulate + i colori grezzi del
            # nuovo blocco, ri-aggregati con lo stesso group-by dei punti
            vecchie = self._somme_colore if self._somme_colore is not None \
                else np.empty((0, 3))
            somme_colore = np.vstack([vecchie, np.asarray(colori, dtype=np.float64)])
        uniche, inversi = np.unique(chiavi, return_inverse=True)
        nuove_somme = np.empty((len(uniche), 3))
        for c in range(3):
            nuove_somme[:, c] = np.bincount(
                inversi, weights=somme[:, c], minlength=len(uniche)
            )
        nuovi_conteggi = np.bincount(
            inversi, weights=conteggi, minlength=len(uniche)
        ).astype(np.int64)
        if con_colore:
            nuove_colore = np.empty((len(uniche), 3))
            for c in range(3):
                nuove_colore[:, c] = np.bincount(
                    inversi, weights=somme_colore[:, c], minlength=len(uniche)
                )
            self._somme_colore = nuove_colore
        self._chiavi, self._somme, self._conteggi = (
            uniche, nuove_somme, nuovi_conteggi,
        )

    def risultato(self) -> np.ndarray:
        if len(self._chiavi) == 0:
            return np.empty((0, 3))
        return self._somme / self._conteggi[:, None]

    def risultato_colore(self) -> np.ndarray | None:
        """Colore medio per voxel (M,3 uint8), o None se non accumulato."""
        if self._somme_colore is None or len(self._chiavi) == 0:
            return None
        colore = self._somme_colore / self._conteggi[:, None]
        return colore.round().clip(0, 255).astype(np.uint8)


def _filtra_bbox(blocco: np.ndarray, bbox: tuple | None) -> np.ndarray:
    if bbox is None:
        return blocco
    minimo, massimo = np.asarray(bbox[:3]), np.asarray(bbox[3:])
    dentro = np.all((blocco >= minimo) & (blocco <= massimo), axis=1)
    return blocco[dentro]


def sottocampiona_e57_streaming(
    percorso: str | Path,
    voxel: float,
    dimensione_blocco: int = 5_000_000,
    bbox: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Prima passata: legge tutto il file e ritorna ``(nuvola ridotta,
    colori ridotti | None)``. Se l'E57 ha colore, viene mediato per voxel
    (allineato ai punti) e serve al taglio per colore del clustering."""
    accumulatore = AccumulatoreVoxel(voxel)
    letti = 0
    colore_mode: bool | None = None
    colore_valido = True
    minimo = np.asarray(bbox[:3]) if bbox is not None else None
    massimo = np.asarray(bbox[3:]) if bbox is not None else None
    for blocco, rgb in leggi_e57_a_blocchi(percorso, dimensione_blocco):
        letti += len(blocco)
        if colore_mode is None:
            colore_mode = rgb is not None
        if bbox is not None:
            dentro = np.all((blocco >= minimo) & (blocco <= massimo), axis=1)
            blocco = blocco[dentro]
            if rgb is not None:
                rgb = rgb[dentro]
        if len(blocco):
            if colore_mode:
                # in modalita' colore l'accumulatore va nutrito col colore a
                # ogni blocco per restare allineato; se un blocco non ha
                # colore (E57 incoerente) si segna e si scarta il colore alla
                # fine, senza disallineare i voxel
                if rgb is None:
                    colore_valido = False
                    rgb = np.full((len(blocco), 3), 128, dtype=np.uint8)
                accumulatore.aggiungi(blocco, rgb)
            else:
                accumulatore.aggiungi(blocco)
        log.info(
            "  streaming: %s punti letti, %s voxel accumulati",
            f"{letti:,}", f"{accumulatore.n_voxel:,}",
        )
    colori = (
        accumulatore.risultato_colore()
        if colore_mode and colore_valido else None
    )
    if colori is not None:
        log.info("  streaming: colore RGB mediato per voxel disponibile")
    elif colore_mode and not colore_valido:
        log.warning("  streaming: colore incoerente tra le scansioni, scartato")
    return accumulatore.risultato(), colori


def esporta_las_streaming(
    percorso_e57: str | Path,
    percorso_las: str | Path,
    punti_ridotti: np.ndarray,
    etichette_ridotte: np.ndarray,
    dimensione_blocco: int = 5_000_000,
    bbox: tuple | None = None,
    dividi_gruppi: bool = True,
) -> Path:
    """Seconda passata: rilegge il file, propaga le etichette dal punto
    ridotto piu' vicino e scrive il LAS classificato a blocchi.

    Con ``dividi_gruppi`` (default) scrive ANCHE, in parallelo e nella stessa
    passata, i tre LAS separati per macro-gruppo scan-to-BIM
    (``..._segmentata_strutturale.las`` / ``_mep.las`` / ``_scarto.las``): ogni
    punto viene instradato nel file del proprio gruppo mantenendo etichetta e
    colore per classe. I file di gruppo rimasti vuoti vengono rimossi a fine
    export."""
    import laspy
    from contextlib import ExitStack

    from scipy.spatial import cKDTree

    from .classi import GRUPPI_MACRO, colora_etichette, gruppo_per_codice

    percorso_las = Path(percorso_las)
    percorso_las.parent.mkdir(parents=True, exist_ok=True)
    albero = cKDTree(punti_ridotti)

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.offsets = punti_ridotti.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    # writer per gruppo: header indipendenti ma con gli stessi offset/scale del
    # file completo, cosi' i record creati sull'header principale sono validi
    # per tutti. La LUT codice-classe -> indice-gruppo instrada in un colpo.
    nomi_gruppi = list(GRUPPI_MACRO.keys())
    lut_gruppo = gruppo_per_codice()
    percorsi_gruppo = {
        g: percorso_las.with_name(f"{percorso_las.stem}_{g}{percorso_las.suffix}")
        for g in nomi_gruppi
    }

    scritti = 0
    scritti_gruppo = {g: 0 for g in nomi_gruppi}
    with ExitStack() as stack:
        scrittore = stack.enter_context(
            laspy.open(str(percorso_las), mode="w", header=header)
        )
        scrittori_gruppo: dict[str, object] = {}
        if dividi_gruppi:
            for g in nomi_gruppi:
                h = laspy.LasHeader(version="1.4", point_format=7)
                h.offsets = header.offsets
                h.scales = header.scales
                scrittori_gruppo[g] = stack.enter_context(
                    laspy.open(str(percorsi_gruppo[g]), mode="w", header=h)
                )

        for blocco, _ in leggi_e57_a_blocchi(percorso_e57, dimensione_blocco):
            blocco = _filtra_bbox(blocco, bbox)
            if len(blocco) == 0:
                continue
            _, indici = albero.query(blocco, k=1, workers=-1)
            classi = etichette_ridotte[indici]
            colori = colora_etichette(classi).astype(np.uint16) * 257

            record = laspy.ScaleAwarePointRecord.zeros(
                len(blocco), header=header
            )
            record.x, record.y, record.z = blocco[:, 0], blocco[:, 1], blocco[:, 2]
            record.red, record.green, record.blue = (
                colori[:, 0], colori[:, 1], colori[:, 2],
            )
            record.classification = classi.astype(np.uint8)
            scrittore.write_points(record)
            scritti += len(blocco)

            if dividi_gruppi:
                indice_gruppo = lut_gruppo[classi.astype(np.int64)]
                for idx, g in enumerate(nomi_gruppi):
                    sel = indice_gruppo == idx
                    n_sel = int(sel.sum())
                    if n_sel == 0:
                        continue
                    sub_blocco, sub_colori = blocco[sel], colori[sel]
                    sub = laspy.ScaleAwarePointRecord.zeros(n_sel, header=header)
                    sub.x, sub.y, sub.z = (
                        sub_blocco[:, 0], sub_blocco[:, 1], sub_blocco[:, 2],
                    )
                    sub.red, sub.green, sub.blue = (
                        sub_colori[:, 0], sub_colori[:, 1], sub_colori[:, 2],
                    )
                    sub.classification = classi[sel].astype(np.uint8)
                    scrittori_gruppo[g].write_points(sub)
                    scritti_gruppo[g] += n_sel
            log.info("  export LAS: %s punti scritti", f"{scritti:,}")

    if dividi_gruppi:
        for g in nomi_gruppi:
            if scritti_gruppo[g] == 0:
                # gruppo assente in questa nuvola: rimuovo il LAS vuoto
                percorsi_gruppo[g].unlink(missing_ok=True)
            else:
                log.info(
                    "  LAS gruppo '%s': %s punti -> %s",
                    g, f"{scritti_gruppo[g]:,}", percorsi_gruppo[g].name,
                )
    return percorso_las
