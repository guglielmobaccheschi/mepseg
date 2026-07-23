"""Interfaccia a riga di comando: ``mepseg`` oppure ``python -m mepseg``."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .classi import ClasseMEP, NOMI_CLASSI
from .io_nuvole import (
    carica_nuvola,
    salva_las_classificata,
    salva_ply_classificata,
)
from .pipeline import segmenta
from .report import genera_report

PERCORSO_CONFIG_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def carica_config(percorso: str | Path | None) -> dict:
    with open(percorso or PERCORSO_CONFIG_DEFAULT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def principale(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mepseg",
        description=(
            "Segmentazione di nuvole di punti per impianti MEP (scan-to-BIM): "
            "etichetta e colora condotte, tubazioni, luci, sprinkler e altri "
            "elementi impiantistici."
        ),
    )
    parser.add_argument(
        "ingresso", nargs="?",
        help="nuvola di punti (.e57 .las .laz .ply .pcd .xyz)",
    )
    parser.add_argument(
        "--controlla", action="store_true",
        help="verifica l'ambiente (Python e dipendenze) ed esce, senza elaborare",
    )
    parser.add_argument(
        "-o", "--output", default="output", help="cartella di output (default: output/)"
    )
    parser.add_argument("-c", "--config", help="file YAML di configurazione personalizzato")
    parser.add_argument("--voxel", type=float, help="dimensione voxel in metri (override)")
    parser.add_argument(
        "--pesi", help="checkpoint della rete (.pth) per la fusione deep learning"
    )
    parser.add_argument(
        "--las", action="store_true", help="esporta anche un .las classificato"
    )
    parser.add_argument(
        "--escludi",
        help=(
            "classi impossibili nel contesto, separate da virgola "
            "(es. --escludi sprinkler,terminale_aria per un data center)"
        ),
    )
    parser.add_argument(
        "--modelli",
        help=(
            "file JSON di modelli confermati ('trova simili'): le istanze "
            "dubbie che combaciano con un modello ne ereditano la classe"
        ),
    )
    parser.add_argument(
        "--salva-modello",
        action="append",
        metavar="ID:NOME[:CLASSE]",
        help=(
            "estrae la firma geometrica dell'istanza ID di questo run e la "
            "salva come modello NOME nel file --modelli (CLASSE opzionale "
            "se l'istanza non ha gia' la classe giusta, es. 90:plafoniera:luce). "
            "Ripetibile per salvare piu' modelli in un solo run"
        ),
    )
    parser.add_argument(
        "--conferme",
        help=(
            "file conferme.json della Fase 4 (conferme, correzioni e rifiuti "
            "dell'utente): le istanze confermate restano della loro classe "
            "(pin), le firme confermate propagano la classe alle istanze "
            "simili, i rifiuti bloccano le conversioni"
        ),
    )
    parser.add_argument(
        "--bbox",
        help=(
            "ritaglia la nuvola al box xmin,ymin,zmin,xmax,ymax,zmax in metri "
            "(scarta i residui di scansioni adiacenti fuori dal locale)"
        ),
    )
    parser.add_argument(
        "--margine",
        type=float,
        default=0.0,
        help=(
            "estende il --bbox di N metri per l'ANALISI (contesto: tubi che "
            "attraversano le pareti, ancoraggi) ma esporta solo l'interno "
            "del bbox originale. Consigliato: 1.0"
        ),
    )
    parser.add_argument("-v", "--verboso", action="store_true")
    argomenti = parser.parse_args(argv)

    # autodiagnosi dell'ambiente: non richiede una nuvola in ingresso
    if argomenti.controlla:
        from .diagnostica import stampa_report

        return 0 if stampa_report() else 1
    if not argomenti.ingresso:
        parser.error("serve la nuvola di ingresso (oppure usa --controlla)")

    logging.basicConfig(
        level=logging.DEBUG if argomenti.verboso else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    log = logging.getLogger("mepseg")

    cfg = carica_config(argomenti.config)
    if argomenti.voxel is not None:
        cfg["preprocess"]["voxel"] = argomenti.voxel
    if argomenti.pesi:
        cfg.setdefault("dl", {})["pesi"] = argomenti.pesi
    if argomenti.escludi:
        cfg.setdefault("regole", {})["classi_escluse"] = [
            v for v in argomenti.escludi.split(",") if v.strip()
        ]

    # modelli confermati ("trova simili")
    richieste_salva = []
    for voce in argomenti.salva_modello or []:
        if not argomenti.modelli:
            parser.error("--salva-modello richiede --modelli (file di destinazione)")
        parti = voce.split(":")
        if len(parti) not in (2, 3) or not parti[0].strip().isdigit():
            parser.error("--salva-modello vuole il formato ID:NOME[:CLASSE]")
        richieste_salva.append({
            "id": int(parti[0]),
            "nome": parti[1].strip(),
            "classe": parti[2].strip().lower() if len(parti) == 3 else None,
        })
    if argomenti.modelli:
        from .somiglianza import carica_modelli

        modelli = carica_modelli(argomenti.modelli)
        if modelli:
            cfg.setdefault("somiglianza", {})["modelli"] = modelli
            log.info(
                "Caricati %d modelli confermati da %s",
                len(modelli), argomenti.modelli,
            )
        elif not richieste_salva:
            log.warning("Nessun modello in %s", argomenti.modelli)

    # conferme della Fase 4 (pin + propagazione + rifiuti)
    if argomenti.conferme:
        from .somiglianza import carica_conferme

        dati_conferme = carica_conferme(argomenti.conferme)
        if dati_conferme["conferme"] or dati_conferme["rifiuti"]:
            cfg.setdefault("somiglianza", {})["conferme"] = dati_conferme["conferme"]
            cfg["somiglianza"]["rifiuti"] = dati_conferme["rifiuti"]
            log.info(
                "Caricate %d conferme e %d rifiuti da %s",
                len(dati_conferme["conferme"]), len(dati_conferme["rifiuti"]),
                argomenti.conferme,
            )
        else:
            log.warning("Nessuna conferma in %s", argomenti.conferme)

    inizio = time.time()
    bbox = None
    if argomenti.bbox:
        valori = [float(v) for v in argomenti.bbox.split(",")]
        if len(valori) != 6:
            parser.error("--bbox richiede 6 valori: xmin,ymin,zmin,xmax,ymax,zmax")
        bbox = tuple(valori)

    # l'analisi puo' usare un box allargato (--margine) per non tagliare il
    # contesto (tubi passanti, ancoraggi); l'export resta sul bbox originale
    bbox_analisi = bbox
    if bbox is not None and argomenti.margine > 0:
        m = argomenti.margine
        bbox_analisi = (
            bbox[0] - m, bbox[1] - m, bbox[2] - m,
            bbox[3] + m, bbox[4] + m, bbox[5] + m,
        )

    percorso_ingresso = Path(argomenti.ingresso)
    cartella = Path(argomenti.output)
    nome_base = percorso_ingresso.stem

    # nuvole E57 enormi: streaming a blocchi invece del caricamento in RAM
    streaming = False
    if percorso_ingresso.suffix.lower() == ".e57":
        from .streaming import conta_punti_e57

        n_dichiarati = conta_punti_e57(percorso_ingresso)
        soglia = int(cfg.get("streaming", {}).get("soglia_punti", 20_000_000))
        streaming = n_dichiarati > soglia
        if streaming:
            log.info(
                "Nuvola da %s punti: attivo l'elaborazione in streaming "
                "(soglia: %s)", f"{n_dichiarati:,}", f"{soglia:,}",
            )

    if streaming:
        from .io_nuvole import Nuvola
        from .streaming import esporta_las_streaming, sottocampiona_e57_streaming

        voxel = float(cfg["preprocess"].get("voxel", 0.03))
        # cache su disco della nuvola ridotta: le iterazioni successive
        # (nuove soglie, --escludi, --bbox) non rileggono il file E57 intero.
        # Il colore, se presente, vive in una cache AFFIANCATA _rgb.npy: le
        # vecchie cache XYZ-only restano leggibili (ma senza taglio per colore)
        percorso_cache = cartella / f"{nome_base}_ridotta_{voxel * 1000:g}mm.npy"
        percorso_cache_rgb = (
            cartella / f"{nome_base}_ridotta_{voxel * 1000:g}mm_rgb.npy"
        )
        colori_ridotti = None
        if percorso_cache.exists():
            log.info("Riuso la nuvola ridotta in cache: %s", percorso_cache)
            punti_ridotti = np.load(percorso_cache)
            if percorso_cache_rgb.exists():
                colori_ridotti = np.load(percorso_cache_rgb)
            else:
                log.warning(
                    "Cache XYZ-only (senza colore): il taglio per colore NON "
                    "agira'. Cancella %s per rigenerarla con l'RGB.",
                    percorso_cache.name,
                )
        else:
            log.info(
                "Passata 1: sottocampionamento voxel %.3f m in streaming...", voxel
            )
            punti_ridotti, colori_ridotti = sottocampiona_e57_streaming(
                percorso_ingresso, voxel
            )
            cartella.mkdir(parents=True, exist_ok=True)
            np.save(percorso_cache, punti_ridotti)
            if colori_ridotti is not None:
                np.save(percorso_cache_rgb, colori_ridotti)
            log.info(
                "Nuvola ridotta salvata in cache: %s%s", percorso_cache,
                " (con colore RGB)" if colori_ridotti is not None else "",
            )
        if bbox_analisi is not None:
            minimo, massimo = np.array(bbox_analisi[:3]), np.array(bbox_analisi[3:])
            dentro = np.all(
                (punti_ridotti >= minimo) & (punti_ridotti <= massimo), axis=1
            )
            punti_ridotti = punti_ridotti[dentro]
            if colori_ridotti is not None:
                colori_ridotti = colori_ridotti[dentro]
        log.info("Nuvola ridotta: %s punti", f"{len(punti_ridotti):,}")
        cfg["preprocess"]["voxel_streaming"] = voxel  # per l'eps di DBSCAN
        cfg["preprocess"]["voxel"] = 0.0  # gia' sottocampionata
        nuvola = Nuvola(
            punti_ridotti, colori_ridotti, origine=str(percorso_ingresso)
        )
    else:
        log.info("Caricamento di %s ...", argomenti.ingresso)
        nuvola = carica_nuvola(argomenti.ingresso)
        if bbox_analisi is not None:
            minimo, massimo = np.array(bbox_analisi[:3]), np.array(bbox_analisi[3:])
            dentro = np.all(
                (nuvola.punti >= minimo) & (nuvola.punti <= massimo), axis=1
            )
            log.info(
                "Ritaglio bbox: tengo %d punti su %d", int(dentro.sum()), len(nuvola)
            )
            from .io_nuvole import Nuvola

            nuvola = Nuvola(
                nuvola.punti[dentro],
                nuvola.colori[dentro] if nuvola.colori is not None else None,
                nuvola.intensita[dentro] if nuvola.intensita is not None else None,
                origine=nuvola.origine,
            )

    risultato = segmenta(nuvola, cfg)

    for richiesta in richieste_salva:
        from .somiglianza import crea_modello, salva_modello

        istanza = next(
            (i for i in risultato.istanze if i.id == richiesta["id"]), None
        )
        if istanza is None:
            log.error(
                "--salva-modello: istanza %d non trovata in questo run",
                richiesta["id"],
            )
            continue
        classe = (
            ClasseMEP[richiesta["classe"].upper()] if richiesta["classe"] else None
        )
        modello = crea_modello(
            istanza, richiesta["nome"], classe, origine=percorso_ingresso.name
        )
        salva_modello(argomenti.modelli, modello)
        log.info(
            "Modello '%s' (%s) salvato in %s",
            modello["nome"], modello["classe"], argomenti.modelli,
        )

    # con --margine l'analisi ha usato il box allargato: l'export torna al
    # bbox originale richiesto dall'utente
    punti_out, etichette_out = nuvola.punti, risultato.etichette
    if bbox is not None and argomenti.margine > 0:
        minimo, massimo = np.array(bbox[:3]), np.array(bbox[3:])
        dentro = np.all((punti_out >= minimo) & (punti_out <= massimo), axis=1)
        punti_out, etichette_out = punti_out[dentro], etichette_out[dentro]

    suffisso = "_segmentata_ridotta" if streaming else "_segmentata"
    percorso_ply = salva_ply_classificata(
        cartella / f"{nome_base}{suffisso}.ply", punti_out, etichette_out
    )
    log.info("Nuvola segmentata: %s", percorso_ply)

    # export di debug: nuvola ridotta colorata per famiglia di superficie
    # (fase A dell'architettura per-superficie), per giudicare a video la
    # qualita' delle patch prima delle regole
    if (
        cfg.get("clustering", {}).get("salva_famiglie", False)
        and risultato.famiglie_superfici is not None
    ):
        import open3d as o3d

        colori_fam = np.array(
            [
                [150, 150, 150],  # 0 nessuna / struttura
                [255, 140, 0],    # 1 piana orizzontale (arancio)
                [0, 150, 255],    # 2 piana verticale (azzurro)
                [0, 200, 80],     # 3 cilindrica (verde)
                [255, 105, 180],  # 4 irregolare (rosa)
            ],
            dtype=np.float64,
        ) / 255.0
        pcd_fam = o3d.geometry.PointCloud()
        pcd_fam.points = o3d.utility.Vector3dVector(risultato.punti_ridotti)
        pcd_fam.colors = o3d.utility.Vector3dVector(
            colori_fam[risultato.famiglie_superfici]
        )
        percorso_fam = cartella / f"{nome_base}_famiglie.ply"
        o3d.io.write_point_cloud(str(percorso_fam), pcd_fam)
        log.info("Famiglie di superficie (debug): %s", percorso_fam)
    if argomenti.las:
        if streaming:
            log.info("Passata 2: export LAS a piena risoluzione in streaming...")
            percorso_las = esporta_las_streaming(
                percorso_ingresso,
                cartella / f"{nome_base}_segmentata.las",
                nuvola.punti,
                risultato.etichette,
                bbox=bbox,
            )
        else:
            percorso_las = salva_las_classificata(
                cartella / f"{nome_base}_segmentata.las",
                punti_out,
                etichette_out,
            )
        log.info("LAS classificato: %s", percorso_las)

    percorso_json, percorso_csv = genera_report(risultato, cartella, nome_base)
    log.info("Report: %s, %s", percorso_json, percorso_csv)

    # riepilogo a terminale
    print("\n=== Riepilogo segmentazione ===")
    totale = len(risultato.etichette)
    for classe in ClasseMEP:
        n = int((risultato.etichette == int(classe)).sum())
        if n:
            print(f"  {NOMI_CLASSI[classe]:<28} {n:>12,} punti ({100.0 * n / totale:5.1f}%)")
    print(f"  {'Istanze MEP individuate':<28} {len(risultato.istanze):>12,}")
    print(f"  Tempo totale: {time.time() - inizio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(principale())
