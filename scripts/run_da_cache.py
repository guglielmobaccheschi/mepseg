"""Run della pipeline direttamente dalla cache ridotta .npy, senza E57.

Utile quando il file E57 originale non e' raggiungibile (disco esterno
scollegato) ma la cache della nuvola ridotta esiste gia': replica il ramo
streaming di cli.py (bbox+margine, voxel_streaming) e produce PLY ridotta
+ report nella cartella di output.

Esempio (il run di riferimento FiberCop 1):
  python scripts/run_da_cache.py "output_locale1_v9/FiberCop 1_ridotta_20mm.npy" \
      -o output_locale1_v9 --voxel 0.02 --bbox=-8,-13,-2.3,8,30,3.2 \
      --margine 1.0 --escludi sprinkler --modelli modelli_fibercop.json
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mepseg.classi import ClasseMEP, NOMI_CLASSI  # noqa: E402
from mepseg.cli import carica_config  # noqa: E402
from mepseg.io_nuvole import Nuvola, salva_ply_classificata  # noqa: E402
from mepseg.pipeline import segmenta  # noqa: E402
from mepseg.report import genera_report  # noqa: E402


def principale() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", help="cache .npy della nuvola ridotta")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-c", "--config")
    parser.add_argument("--voxel", type=float, default=0.02,
                        help="voxel con cui la cache era stata generata")
    parser.add_argument("--bbox")
    parser.add_argument("--margine", type=float, default=0.0)
    parser.add_argument("--escludi")
    parser.add_argument("--modelli")
    parser.add_argument("--conferme")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("mepseg")
    inizio = time.time()

    cfg = carica_config(argomenti.config)
    cfg["preprocess"]["voxel_streaming"] = argomenti.voxel
    cfg["preprocess"]["voxel"] = 0.0  # gia' sottocampionata
    if argomenti.escludi:
        cfg.setdefault("regole", {})["classi_escluse"] = [
            v for v in argomenti.escludi.split(",") if v.strip()
        ]
    if argomenti.modelli:
        from mepseg.somiglianza import carica_modelli

        modelli = carica_modelli(argomenti.modelli)
        if modelli:
            cfg.setdefault("somiglianza", {})["modelli"] = modelli
            log.info("Caricati %d modelli da %s", len(modelli), argomenti.modelli)
    if argomenti.conferme:
        from mepseg.somiglianza import carica_conferme

        dati = carica_conferme(argomenti.conferme)
        cfg.setdefault("somiglianza", {})["conferme"] = dati["conferme"]
        cfg["somiglianza"]["rifiuti"] = dati["rifiuti"]
        log.info(
            "Caricate %d conferme e %d rifiuti", len(dati["conferme"]),
            len(dati["rifiuti"]),
        )

    punti = np.load(argomenti.cache)
    bbox = None
    if argomenti.bbox:
        bbox = tuple(float(v) for v in argomenti.bbox.split(","))
        m = argomenti.margine
        minimo = np.array(bbox[:3]) - m
        massimo = np.array(bbox[3:]) + m
        punti = punti[np.all((punti >= minimo) & (punti <= massimo), axis=1)]
    log.info("Nuvola ridotta dalla cache: %s punti", f"{len(punti):,}")

    nuvola = Nuvola(punti, origine=argomenti.cache)
    risultato = segmenta(nuvola, cfg)

    cartella = Path(argomenti.output)
    nome_base = Path(argomenti.cache).stem.split("_ridotta")[0]
    punti_out, etichette_out = nuvola.punti, risultato.etichette
    if bbox is not None and argomenti.margine > 0:
        dentro = np.all(
            (punti_out >= np.array(bbox[:3])) & (punti_out <= np.array(bbox[3:])),
            axis=1,
        )
        punti_out, etichette_out = punti_out[dentro], etichette_out[dentro]
    percorso_ply = salva_ply_classificata(
        cartella / f"{nome_base}_segmentata_ridotta.ply", punti_out, etichette_out
    )
    log.info("Nuvola segmentata: %s", percorso_ply)
    genera_report(risultato, cartella, nome_base)

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
