"""Adattatore dei dataset pubblici -> triple di training di mepseg.

Verifica sull'importatore PSNET5 (formato stile S3DIS):
  - la mappatura YAML si carica e punta a ClasseMEP valide;
  - la classe si estrae dal nome file, i suffissi di scena non confondono;
  - round-trip su una micro-cartella sintetica: le triple prodotte hanno
    lunghezze coerenti, fiducia piena (2) e SOLO le classi mappate;
  - le classi non mappate vengono saltate, non indovinate;
  - lo split train/val e' per area (nessun punto mescolato).

Esecuzione:  python tests/test_importa_pubblico.py   (oppure pytest tests/)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from mepseg.classi import ClasseMEP  # noqa: E402
from mepseg.dl.importa_pubblico import (  # noqa: E402
    _classe_da_nome_file,
    carica_mappatura,
    importa_psnet5,
)


def test_mappatura_psnet5_valida():
    mappa = carica_mappatura("psnet5")
    # tutte le destinazioni sono ClasseMEP valide
    for chiave, codice in mappa.items():
        assert 0 <= codice < len(ClasseMEP)
        assert isinstance(ClasseMEP(codice), ClasseMEP)
    # le cinque classi note di PSNET5 ci sono
    assert {"pipe", "pump", "tank", "ibeam", "rectangularbeam"} <= set(mappa)
    assert mappa["pipe"] == int(ClasseMEP.TUBAZIONE)
    assert mappa["ibeam"] == int(ClasseMEP.TRAVE)


def test_classe_da_nome_file():
    chiavi = ["pipe", "pump", "tank", "ibeam", "rectangularbeam"]
    assert _classe_da_nome_file("pipe_chiller", chiavi) == "pipe"
    assert _classe_da_nome_file("rectangularbeam_OSCG", chiavi) == "rectangularbeam"
    assert _classe_da_nome_file("ibeam_WRT2", chiavi) == "ibeam"
    # scena con piu' token non rompe l'estrazione
    assert _classe_da_nome_file("pump_SPH_2", chiavi) == "pump"
    # classe sconosciuta -> None (verra' saltata, mai indovinata)
    assert _classe_da_nome_file("valve_chiller", chiavi) is None


def _scrivi_annotazione(percorso: Path, punti: np.ndarray) -> None:
    percorso.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((len(punti), 3), 128)
    righe = [
        f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}"
        for (x, y, z), (r, g, b) in zip(punti, rgb)
    ]
    percorso.write_text("\n".join(righe) + "\n", encoding="ascii")


def _micro_psnet5(radice: Path) -> None:
    """Due aree in stile S3DIS con classi mappate + una NON mappata."""
    rng = np.random.default_rng(0)
    for area, scena in (("Area_1", "chiller"), ("Area_2", "OSCG")):
        base = radice / area / "Room_1" / "Annotations"
        for classe in ("pipe", "pump", "ibeam", "rectangularbeam"):
            _scrivi_annotazione(
                base / f"{classe}_{scena}.txt", rng.uniform(0, 3, size=(200, 3))
            )
        # classe non presente nella mappatura: deve essere saltata
        _scrivi_annotazione(
            base / f"valve_{scena}.txt", rng.uniform(0, 3, size=(50, 3))
        )


def test_round_trip_importa_psnet5():
    mappa = carica_mappatura("psnet5")
    classi_mappate = set(mappa.values())
    codice_valve_assente = int(ClasseMEP.MEP_GENERICO)  # non deve comparire
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sorgente = tmp / "PSNet5"
        _micro_psnet5(sorgente)
        out = tmp / "dataset_mep"

        # voxel piccolo per non collassare i 200 punti sintetici
        stat = importa_psnet5(sorgente, out, mappa, voxel=0.001, val_area="Area_2")
        assert stat["train"] == ["Area_1"]
        assert stat["val"] == ["Area_2"]

        for split, area in (("train", "area_1"), ("val", "area_2")):
            base = out / split / f"psnet5_{area}"
            P = np.load(base.with_suffix(".npy"))
            E = np.load(base.with_name(base.name + "_etichette.npy"))
            Fd = np.load(base.with_name(base.name + "_fiducia.npy"))
            assert len(P) == len(E) == len(Fd)
            # verita' annotata -> fiducia piena ovunque
            assert set(np.unique(Fd)) == {2}
            # SOLO classi mappate, e la 'valve' non mappata NON c'e'
            assert set(np.unique(E)).issubset(classi_mappate)
            assert codice_valve_assente not in set(np.unique(E))


if __name__ == "__main__":
    test_mappatura_psnet5_valida()
    test_classe_da_nome_file()
    test_round_trip_importa_psnet5()
    print("test_importa_pubblico: tutti i test superati")
