"""Export progressivo del dataset di fine-tuning da un run della pipeline.

Il dataset e' CUMULATIVO e multi-nuvola: ogni run esportato aggiunge (o
sostituisce, per la stessa nuvola) le proprie triple in ``train/`` e
``val/``. Piu' nuvole si confermano, piu' il fine-tuning ha da imparare —
e' il cuore del tool progressivo.

Struttura prodotta (convenzione di :mod:`mepseg.dl.dataset`):
  dataset_mep/
    train/  <nuvola>.npy + <nuvola>_etichette.npy + <nuvola>_fiducia.npy
    val/    <nuvola>_val.npy + ...
    pesi/   rete_mep_vN.pth   (checkpoint del training)
    meta.json                      (per nuvola: origine, punti, conferme)

Fiducia (documento Fase 4, §4.2):
  2 = punti delle istanze confermate/corrette in Fase 4 (ri-identificate
      nel report per firma, mappate sui punti via KD-tree dei punti-istanza)
  1 = classi assegnate dalla pipeline senza conferma
  0 = incerto: struttura generica, MEP generico, arredo, waste

Uso da CLI:  python -m mepseg.dl.esporta cartella_run -o dataset_mep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ..classi import ClasseMEP

log = logging.getLogger("mepseg")

CLASSI_INCERTE = {
    int(ClasseMEP.STRUTTURA),
    int(ClasseMEP.MEP_GENERICO),
    int(ClasseMEP.NON_MEP_ARREDO),
    int(ClasseMEP.WASTE),
}


def _slug(nome: str) -> str:
    return nome.replace(" ", "_").lower()


def esporta_run(
    cartella_run: str | Path,
    cartella_dataset: str | Path,
    frazione_val: float = 0.3,
    max_punti_split: int = 800_000,
) -> dict:
    """Esporta un run nel dataset cumulativo. Ritorna le statistiche."""
    from scipy.spatial import cKDTree

    from ..io_nuvole import leggi_ply_classificata
    from ..somiglianza import carica_conferme, firma_da_riga, firme_coincidono

    cartella_run = Path(cartella_run)
    cartella_dataset = Path(cartella_dataset)

    ply = next(cartella_run.glob("*_segmentata*.ply"), None)
    if ply is None:
        raise FileNotFoundError(f"Nessuna PLY segmentata in {cartella_run}")
    nome_base = ply.stem.split("_segmentata")[0]
    nome = _slug(nome_base)

    d = leggi_ply_classificata(ply)
    punti = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    etichette = d["classe"].astype(np.uint8)

    # fiducia di base: 1 dalle regole, 0 per le classi incerte
    fiducia = np.ones(len(punti), dtype=np.uint8)
    fiducia[np.isin(etichette, list(CLASSI_INCERTE))] = 0

    # fiducia 2: punti delle istanze confermate/corrette in Fase 4
    n_confermati, n_istanze_confermate = 0, 0
    percorso_conferme = cartella_run / "conferme.json"
    percorso_npz = cartella_run / f"{nome_base}_istanze_punti.npz"
    percorso_report = cartella_run / f"{nome_base}_report.json"
    if percorso_conferme.exists() and percorso_npz.exists() and percorso_report.exists():
        conferme = carica_conferme(percorso_conferme)["conferme"]
        report = json.loads(percorso_report.read_text(encoding="utf-8"))
        npz = np.load(percorso_npz)
        id_confermati = set()
        for c in conferme:
            for riga in report["istanze"]:
                if firme_coincidono(c["firma"], firma_da_riga(riga)):
                    id_confermati.add(riga["id"])
                    break
        n_istanze_confermate = len(id_confermati)
        scelti = np.isin(npz["id_istanza"], list(id_confermati))
        if scelti.any():
            albero = cKDTree(npz["punti"][scelti])
            dist, _ = albero.query(punti, distance_upper_bound=0.05)
            vicini = np.isfinite(dist)
            fiducia[vicini] = 2
            n_confermati = int(vicini.sum())

    # split train/val per ZONE lungo y (mai punti mescolati) + eventuale
    # sottocampionamento per tenere i file maneggevoli
    soglia_y = float(np.quantile(punti[:, 1], 1.0 - frazione_val))
    rng = np.random.default_rng(0)
    stat_split = {}
    for split, maschera, suffisso in (
        ("train", punti[:, 1] < soglia_y, ""),
        ("val", punti[:, 1] >= soglia_y, "_val"),
    ):
        idx = np.where(maschera)[0]
        if max_punti_split and len(idx) > max_punti_split:
            idx = rng.choice(idx, max_punti_split, replace=False)
        cartella = cartella_dataset / split
        cartella.mkdir(parents=True, exist_ok=True)
        np.save(cartella / f"{nome}{suffisso}.npy", punti[idx])
        np.save(cartella / f"{nome}{suffisso}_etichette.npy", etichette[idx])
        np.save(cartella / f"{nome}{suffisso}_fiducia.npy", fiducia[idx])
        stat_split[split] = {
            "punti": int(len(idx)),
            "confermati": int((fiducia[idx] == 2).sum()),
        }

    # meta.json: una voce per nuvola (riesportare sostituisce la voce)
    percorso_meta = cartella_dataset / "meta.json"
    meta = {"nuvole": {}}
    if percorso_meta.exists():
        meta = json.loads(percorso_meta.read_text(encoding="utf-8"))
        meta.setdefault("nuvole", {})
    meta["nuvole"][nome] = {
        "origine": str(ply),
        "esportata": datetime.now().isoformat(timespec="seconds"),
        "punti": int(len(punti)),
        "istanze_confermate": n_istanze_confermate,
        "punti_confermati": n_confermati,
        "split": stat_split,
    }
    percorso_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stat = {
        "nuvola": nome,
        "punti": int(len(punti)),
        "punti_confermati": n_confermati,
        "istanze_confermate": n_istanze_confermate,
        "fiducia_regole": int((fiducia == 1).sum()),
        "fiducia_incerti": int((fiducia == 0).sum()),
        "nuvole_nel_dataset": len(meta["nuvole"]),
        "split": stat_split,
    }
    log.info(
        "Dataset: esportata '%s' (%s punti, %s confermati da %d istanze); "
        "nuvole nel dataset: %d",
        nome, f"{stat['punti']:,}", f"{n_confermati:,}",
        n_istanze_confermate, stat["nuvole_nel_dataset"],
    )
    return stat


def stato_dataset(cartella_dataset: str | Path) -> dict:
    """Statistiche del dataset cumulativo (per la GUI)."""
    cartella_dataset = Path(cartella_dataset)
    percorso_meta = cartella_dataset / "meta.json"
    stato = {"esiste": False, "nuvole": {}, "pesi": []}
    if percorso_meta.exists():
        meta = json.loads(percorso_meta.read_text(encoding="utf-8"))
        stato["esiste"] = True
        stato["nuvole"] = meta.get("nuvole", {})
    cartella_pesi = cartella_dataset / "pesi"
    if cartella_pesi.exists():
        for p in sorted(cartella_pesi.glob("rete_mep_v*.pth")):
            voce = {"nome": p.name, "percorso": str(p)}
            try:
                import torch

                ck = torch.load(p, map_location="cpu")
                voce["miou_val"] = round(float(ck.get("miou_val", 0.0)), 3)
                voce["epoca"] = int(ck.get("epoca", 0))
            except Exception:  # noqa: BLE001 — checkpoint illeggibile: solo nome
                pass
            stato["pesi"].append(voce)
    return stato


def prossimo_checkpoint(cartella_dataset: str | Path) -> Path:
    """Percorso del prossimo checkpoint versionato (v1, v2, ...)."""
    cartella_pesi = Path(cartella_dataset) / "pesi"
    versione = 1
    while (cartella_pesi / f"rete_mep_v{versione}.pth").exists():
        versione += 1
    return cartella_pesi / f"rete_mep_v{versione}.pth"


def ultimo_checkpoint(cartella_dataset: str | Path) -> Path | None:
    """L'ultimo checkpoint versionato, se esiste."""
    esistenti = sorted(
        (Path(cartella_dataset) / "pesi").glob("rete_mep_v*.pth"),
        key=lambda p: int(p.stem.split("_v")[-1]),
    )
    return esistenti[-1] if esistenti else None


def principale(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Esporta un run nel dataset cumulativo di fine-tuning"
    )
    parser.add_argument("run", help="cartella del run (PLY + report + conferme)")
    parser.add_argument("-o", "--output", default="dataset_mep")
    parser.add_argument("--frazione-val", type=float, default=0.3)
    parser.add_argument("--max-punti", type=int, default=800_000)
    argomenti = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    stat = esporta_run(
        argomenti.run, argomenti.output, argomenti.frazione_val, argomenti.max_punti
    )
    print(json.dumps(stat, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(principale())
