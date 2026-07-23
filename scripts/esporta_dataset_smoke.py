"""Dataset di SMOKE-TEST per dl/allenamento.py da un run della pipeline.

Versione di lavoro (l'export vero sara' `mepseg-esporta-dataset`, passo 4
del piano): prende la PLY classificata di un run, la spezza in train/val
lungo y e scrive le triple nome.npy + _etichette.npy + _fiducia.npy.

Fiducia (documento Fase 4, §4.2):
  2 = punti delle istanze confermate/corrette in Fase 4 (da conferme.json,
      ri-identificate nel report per firma e mappate sui punti via KD-tree
      dei punti-istanza del run);
  1 = classi assegnate dalla pipeline senza conferma;
  0 = incerto: struttura generica, MEP generico, arredo, waste.

Esempio:
  python scripts/esporta_dataset_smoke.py output_se_gui -o dataset_mep_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mepseg.classi import ClasseMEP  # noqa: E402
from mepseg.io_nuvole import leggi_ply_classificata  # noqa: E402
from mepseg.somiglianza import (  # noqa: E402
    carica_conferme,
    firma_da_riga,
    firme_coincidono,
)

CLASSI_INCERTE = {
    int(ClasseMEP.STRUTTURA),
    int(ClasseMEP.MEP_GENERICO),
    int(ClasseMEP.NON_MEP_ARREDO),
    int(ClasseMEP.WASTE),
}


def principale() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="cartella di un run (PLY + report + conferme)")
    parser.add_argument("-o", "--output", default="dataset_mep_smoke")
    parser.add_argument(
        "--frazione-val", type=float, default=0.3,
        help="frazione della nuvola (lungo y) riservata alla validazione",
    )
    parser.add_argument(
        "--max-punti", type=int, default=400_000,
        help="sottocampionamento casuale per split (0 = tutti)",
    )
    argomenti = parser.parse_args()

    cartella_run = Path(argomenti.run)
    ply = next(cartella_run.glob("*_segmentata*.ply"))
    nome_base = ply.stem.split("_segmentata")[0]
    d = leggi_ply_classificata(ply)
    punti = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    etichette = d["classe"].astype(np.uint8)
    print(f"Nuvola: {ply.name}, {len(punti):,} punti")

    # fiducia di base: 1 dalle regole, 0 per le classi incerte
    fiducia = np.ones(len(punti), dtype=np.uint8)
    incerti = np.isin(etichette, list(CLASSI_INCERTE))
    fiducia[incerti] = 0

    # fiducia 2: punti delle istanze confermate/corrette in Fase 4
    percorso_conferme = cartella_run / "conferme.json"
    percorso_npz = cartella_run / f"{nome_base}_istanze_punti.npz"
    percorso_report = cartella_run / f"{nome_base}_report.json"
    n_confermati = 0
    n_istanze_confermate = 0
    if percorso_conferme.exists() and percorso_npz.exists():
        from scipy.spatial import cKDTree

        conferme = carica_conferme(percorso_conferme)["conferme"]
        report = json.loads(percorso_report.read_text(encoding="utf-8"))
        npz = np.load(percorso_npz)
        punti_ist, id_ist = npz["punti"], npz["id_istanza"]
        id_confermati = set()
        for c in conferme:
            for riga in report["istanze"]:
                if firme_coincidono(c["firma"], firma_da_riga(riga)):
                    id_confermati.add(riga["id"])
                    break
        n_istanze_confermate = len(id_confermati)
        if id_confermati:
            scelti = np.isin(id_ist, list(id_confermati))
            if scelti.any():
                albero = cKDTree(punti_ist[scelti])
                dist, _ = albero.query(punti, distance_upper_bound=0.05)
                vicini = np.isfinite(dist)
                fiducia[vicini] = 2
                n_confermati = int(vicini.sum())
    print(
        f"Fiducia: {n_confermati:,} confermati (da {n_istanze_confermate} istanze), "
        f"{int((fiducia == 1).sum()):,} da regole, {int((fiducia == 0).sum()):,} incerti"
    )

    # split train/val lungo y (zone diverse, non punti mescolati)
    soglia_y = float(np.quantile(punti[:, 1], 1.0 - argomenti.frazione_val))
    maschere = {
        "train": punti[:, 1] < soglia_y,
        "val": punti[:, 1] >= soglia_y,
    }
    rng = np.random.default_rng(0)
    meta = {"origine": str(ply), "soglia_y": round(soglia_y, 3), "split": {}}
    for split, maschera in maschere.items():
        idx = np.where(maschera)[0]
        if argomenti.max_punti and len(idx) > argomenti.max_punti:
            idx = rng.choice(idx, argomenti.max_punti, replace=False)
        cartella = Path(argomenti.output) / split
        cartella.mkdir(parents=True, exist_ok=True)
        nome = nome_base.replace(" ", "_").lower()
        np.save(cartella / f"{nome}.npy", punti[idx])
        np.save(cartella / f"{nome}_etichette.npy", etichette[idx])
        np.save(cartella / f"{nome}_fiducia.npy", fiducia[idx])
        meta["split"][split] = {
            "punti": int(len(idx)),
            "confermati": int((fiducia[idx] == 2).sum()),
        }
        print(f"{split}: {len(idx):,} punti -> {cartella / (nome + '.npy')}")
    (Path(argomenti.output) / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(principale())
