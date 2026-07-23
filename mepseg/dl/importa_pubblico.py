"""Importa un dataset pubblico annotato nel formato di training di mepseg.

I dataset pubblici della letteratura (PSNET5, ...) usano tassonomie e formati
diversi dai nostri. Questo modulo li converte nelle triple che il trainer
(:mod:`mepseg.dl.dataset`) gia' consuma:

  dataset_mep/
    train/  <nome>.npy (N,3) + <nome>_etichette.npy + <nome>_fiducia.npy
    val/    idem

Le etichette del dataso pubblico sono verita' annotata da esperti -> tutta la
fiducia e' 2 (peso pieno nel training). La corrispondenza tra le classi del
dataset e le nostre 17 e' definita in un file di mappatura versionato e
MODIFICABILE (``mepseg/dl/mappature/<dataset>.yaml``): il tool non cabla mai
una classe, la mappatura la fornisce l'utente per il dataset specifico.

Uso da CLI:
  python -m mepseg.dl.importa_pubblico <cartella_sorgente> --dataset psnet5 \
      -o dataset_mep [--voxel 0.02] [--val-area Area_4] [--max-punti N]

Formato PSNET5 (stile S3DIS): la classe e' il prefisso del nome file
dell'annotazione (``Area_*/Room_*/Annotations/<classe>_<scena>.txt``); ogni
riga e' ``X Y Z R G B`` (il colore non serve al training e viene ignorato).
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

PERCORSO_MAPPATURE = Path(__file__).resolve().parent / "mappature"


def carica_mappatura(dataset: str) -> dict[str, int]:
    """Legge ``mappature/<dataset>.yaml`` -> {classe_dataset: codice ClasseMEP}."""
    import yaml

    percorso = PERCORSO_MAPPATURE / f"{dataset}.yaml"
    if not percorso.exists():
        disponibili = ", ".join(p.stem for p in PERCORSO_MAPPATURE.glob("*.yaml"))
        raise FileNotFoundError(
            f"Nessuna mappatura per '{dataset}' in {PERCORSO_MAPPATURE}. "
            f"Mappature disponibili: {disponibili or '(nessuna)'}"
        )
    dati = yaml.safe_load(percorso.read_text(encoding="utf-8")) or {}
    grezza = dati.get("mappa", {})
    if not grezza:
        raise ValueError(f"Mappatura '{dataset}' vuota o senza chiave 'mappa'")
    mappa: dict[str, int] = {}
    for chiave, valore in grezza.items():
        try:
            mappa[str(chiave).lower()] = int(ClasseMEP[str(valore).upper()])
        except KeyError as exc:
            raise ValueError(
                f"Mappatura '{dataset}': '{valore}' non e' una ClasseMEP valida "
                f"(chiave '{chiave}')"
            ) from exc
    return mappa


def _classe_da_nome_file(stem: str, chiavi) -> str | None:
    """Estrae la chiave-classe dal nome file (prefisso), es.
    ``rectangularbeam_chiller`` -> ``rectangularbeam``. Sceglie la chiave
    piu' lunga che combacia, per non confondere prefissi annidati."""
    s = stem.lower()
    candidati = [k for k in chiavi if s == k or s.startswith(k + "_")]
    return max(candidati, key=len) if candidati else None


def _voxel_downsample(
    punti: np.ndarray, etichette: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Un punto rappresentante per voxel (griglia globale coerente). Ai bordi
    tra due classi vince il primo punto del voxel: frazione trascurabile."""
    if voxel <= 0 or len(punti) == 0:
        return punti, etichette
    chiave = np.floor(punti / voxel).astype(np.int64)
    _, indici = np.unique(chiave, axis=0, return_index=True)
    return punti[indici], etichette[indici]


def _leggi_xyz(percorso: Path) -> np.ndarray:
    """Legge X Y Z (R G B ignorati) via Open3D (parser C++, ~18x piu' veloce
    del parsing testo in Python). Ritorna (N,3) float32."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(percorso), format="xyzrgb")
    return np.asarray(pcd.points, dtype=np.float32)


def importa_psnet5(
    cartella_sorgente: str | Path,
    cartella_dataset: str | Path,
    mappa: dict[str, int],
    voxel: float = 0.02,
    val_area: str | None = None,
    max_punti: int | None = None,
) -> dict:
    """Converte PSNET5 (stile S3DIS) nelle triple di training di mepseg.

    Una nuvola di training per AREA. Per default l'ultima area diventa
    validazione (convenzione S3DIS: hold-out per area, mai punti mescolati
    tra train e val). Ritorna le statistiche."""
    cartella_sorgente = Path(cartella_sorgente)
    cartella_dataset = Path(cartella_dataset)
    aree = sorted(d for d in cartella_sorgente.glob("Area_*") if d.is_dir())
    if not aree:
        raise FileNotFoundError(
            f"Nessuna cartella Area_* in {cartella_sorgente} "
            "(atteso il formato PSNET5 stile S3DIS)"
        )
    if val_area is None:
        val_area = aree[-1].name

    stat_aree: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    for area in aree:
        annotazioni = sorted(area.glob("Room_*/Annotations/*.txt"))
        blocchi_punti, blocchi_etich = [], []
        saltate: set[str] = set()
        for percorso in annotazioni:
            chiave = _classe_da_nome_file(percorso.stem, mappa)
            if chiave is None:
                saltate.add(percorso.stem.rsplit("_", 1)[0])
                continue
            codice = mappa[chiave]
            xyz = _leggi_xyz(percorso)
            # voxel PER FILE subito: tiene la RAM bassa (pipe = 22M punti)
            xyz, et = _voxel_downsample(
                xyz, np.full(len(xyz), codice, dtype=np.uint8), voxel
            )
            blocchi_punti.append(xyz)
            blocchi_etich.append(et)
            log.info(
                "  %-28s -> %-16s %s voxel",
                percorso.name, ClasseMEP(codice).name, f"{len(xyz):,}",
            )
        for nome in sorted(saltate):
            log.warning("  classe non mappata: '%s' (saltata)", nome)
        if not blocchi_punti:
            log.warning("  %s: nessuna classe mappata, area saltata", area.name)
            continue

        punti = np.vstack(blocchi_punti)
        etichette = np.concatenate(blocchi_etich)
        # voxel GLOBALE dell'area: unifica i bordi tra file sulla stessa griglia
        punti, etichette = _voxel_downsample(punti, etichette, voxel)
        if max_punti and len(punti) > max_punti:
            sel = rng.choice(len(punti), max_punti, replace=False)
            punti, etichette = punti[sel], etichette[sel]

        split = "val" if area.name == val_area else "train"
        cartella = cartella_dataset / split
        cartella.mkdir(parents=True, exist_ok=True)
        nome = f"psnet5_{area.name.lower()}"
        np.save(cartella / f"{nome}.npy", punti.astype(np.float32))
        np.save(cartella / f"{nome}_etichette.npy", etichette.astype(np.uint8))
        # verita' annotata da esperti -> fiducia piena
        np.save(
            cartella / f"{nome}_fiducia.npy",
            np.full(len(punti), 2, dtype=np.uint8),
        )
        conteggio = {
            ClasseMEP(c).name: int((etichette == c).sum())
            for c in np.unique(etichette)
        }
        stat_aree[area.name] = {
            "split": split, "punti": int(len(punti)), "classi": conteggio,
        }
        log.info(
            "Area %s -> %s: %s punti (%s)",
            area.name, split, f"{len(punti):,}", conteggio,
        )

    _scrivi_meta(cartella_dataset, "psnet5", cartella_sorgente, voxel, stat_aree)
    return {
        "dataset": "psnet5",
        "aree": stat_aree,
        "train": sorted(a for a, s in stat_aree.items() if s["split"] == "train"),
        "val": sorted(a for a, s in stat_aree.items() if s["split"] == "val"),
    }


def _scrivi_meta(
    cartella_dataset: Path, dataset: str, sorgente: Path, voxel: float, aree: dict
) -> None:
    percorso = cartella_dataset / "meta_pubblico.json"
    meta = {}
    if percorso.exists():
        meta = json.loads(percorso.read_text(encoding="utf-8"))
    meta.setdefault("dataset_pubblici", {})[dataset] = {
        "sorgente": str(sorgente),
        "voxel": voxel,
        "importato": datetime.now().isoformat(timespec="seconds"),
        "aree": aree,
    }
    cartella_dataset.mkdir(parents=True, exist_ok=True)
    percorso.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


_IMPORTATORI = {"psnet5": importa_psnet5}


def principale(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mepseg-importa-pubblico",
        description="Converte un dataset pubblico annotato nel formato di "
        "training di mepseg (triple .npy con fiducia piena)",
    )
    parser.add_argument("sorgente", help="cartella radice del dataset scaricato")
    parser.add_argument(
        "--dataset", required=True, choices=sorted(_IMPORTATORI),
        help="quale dataset pubblico (determina lettore + mappatura)",
    )
    parser.add_argument("-o", "--output", default="dataset_mep")
    parser.add_argument(
        "--voxel", type=float, default=0.02,
        help="lato voxel in metri per il sottocampionamento (default 0.02, "
        "coerente con la pipeline geometrica)",
    )
    parser.add_argument(
        "--val-area", default=None,
        help="nome dell'area da tenere per validazione (default: l'ultima)",
    )
    parser.add_argument(
        "--max-punti", type=int, default=None,
        help="tetto di punti per area, per run rapidi (default: nessuno)",
    )
    argomenti = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    mappa = carica_mappatura(argomenti.dataset)
    importatore = _IMPORTATORI[argomenti.dataset]
    stat = importatore(
        argomenti.sorgente, argomenti.output, mappa,
        voxel=argomenti.voxel, val_area=argomenti.val_area,
        max_punti=argomenti.max_punti,
    )
    print(json.dumps(stat, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(principale())
