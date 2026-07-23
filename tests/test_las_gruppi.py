"""Export del LAS a piena risoluzione diviso in tre macro-gruppi.

Chiusura della pipeline scan-to-BIM: oltre al LAS completo, il tool scrive
tre nuvole .las separate — strutturale/architettonico, impianti MEP, scarto
(arredo + waste) — instradando ogni punto nel file del proprio gruppo e
conservando etichetta e colore per classe.

Il test verifica:
  - il partizionamento delle 17 classi e' ESAUSTIVO ed ESCLUSIVO;
  - la LUT codice-classe -> indice-gruppo e' coerente con GRUPPI_MACRO;
  - il round-trip: i punti scritti in ogni file appartengono SOLO al proprio
    gruppo, e l'unione dei tre ricompone la nuvola completa (nessun punto
    perso, nessun duplicato).

Esecuzione:  python tests/test_las_gruppi.py   (oppure pytest tests/)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from mepseg.classi import (  # noqa: E402
    GRUPPI_MACRO,
    NUM_CLASSI,
    ClasseMEP,
    gruppo_per_codice,
)
from mepseg.io_nuvole import salva_las_per_gruppi  # noqa: E402


def test_partizionamento_esaustivo_ed_esclusivo():
    viste: list[int] = []
    for classi in GRUPPI_MACRO.values():
        viste.extend(int(c) for c in classi)
    # ogni codice compare una sola volta...
    assert len(viste) == len(set(viste)), "una classe appartiene a piu' gruppi"
    # ...e tutte le classi sono coperte
    assert set(viste) == {int(c) for c in ClasseMEP}, "classi scoperte dai gruppi"
    assert len(viste) == NUM_CLASSI


def test_gruppo_per_codice_coerente():
    lut = gruppo_per_codice()
    assert lut.shape == (NUM_CLASSI,)
    assert (lut >= 0).all(), "codice classe senza gruppo assegnato"
    for indice, classi in enumerate(GRUPPI_MACRO.values()):
        for classe in classi:
            assert lut[int(classe)] == indice


def _nuvola_di_prova() -> tuple[np.ndarray, np.ndarray]:
    """Una nuvola con punti da tutte e tre le macro-categorie."""
    rng = np.random.default_rng(0)
    campioni = [
        (ClasseMEP.PARETE, 50),          # strutturale
        (ClasseMEP.SOFFITTO, 30),        # strutturale
        (ClasseMEP.CONDOTTA_CIRCOLARE, 40),  # mep
        (ClasseMEP.LUCE, 20),            # mep
        (ClasseMEP.WASTE, 15),           # scarto
        (ClasseMEP.NON_MEP_ARREDO, 10),  # scarto
    ]
    punti, etichette = [], []
    for classe, n in campioni:
        punti.append(rng.uniform(-5, 5, size=(n, 3)))
        etichette.append(np.full(n, int(classe), dtype=np.uint8))
    return np.vstack(punti), np.concatenate(etichette)


def test_round_trip_split_las():
    import laspy

    punti, etichette = _nuvola_di_prova()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "prova_segmentata.las"
        scritti = salva_las_per_gruppi(base, punti, etichette)

        # tutti e tre i gruppi presenti in questa nuvola
        assert set(scritti) == set(GRUPPI_MACRO), "manca un file di gruppo"

        totale_riletto = 0
        for gruppo, percorso in scritti.items():
            assert percorso.exists()
            assert percorso.name == f"prova_segmentata_{gruppo}.las"
            codici_gruppo = {int(c) for c in GRUPPI_MACRO[gruppo]}
            las = laspy.read(str(percorso))
            classi = np.asarray(las.classification)
            # solo classi del proprio gruppo
            assert set(np.unique(classi)).issubset(codici_gruppo), (
                f"il file '{gruppo}' contiene classi estranee"
            )
            # conteggio coerente con la nuvola sorgente
            atteso = int(np.isin(etichette, list(codici_gruppo)).sum())
            assert len(classi) == atteso
            totale_riletto += len(classi)

        # l'unione dei tre ricompone la nuvola completa
        assert totale_riletto == len(etichette)


def test_gruppo_vuoto_saltato():
    # nuvola di soli punti MEP: niente file strutturale ne' scarto
    punti = np.random.default_rng(1).uniform(-1, 1, size=(20, 3))
    etichette = np.full(20, int(ClasseMEP.TUBAZIONE), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "solo_mep_segmentata.las"
        scritti = salva_las_per_gruppi(base, punti, etichette)
        assert set(scritti) == {"mep"}
        assert not (Path(tmp) / "solo_mep_segmentata_strutturale.las").exists()
        assert not (Path(tmp) / "solo_mep_segmentata_scarto.las").exists()


if __name__ == "__main__":
    test_partizionamento_esaustivo_ed_esclusivo()
    test_gruppo_per_codice_coerente()
    test_round_trip_split_las()
    test_gruppo_vuoto_saltato()
    print("test_las_gruppi: tutti i test superati")
