"""Genera anteprime PNG (pianta e sezione) da una nuvola segmentata.

Uso:
    python scripts/anteprima.py output/xxx_segmentata.ply [-o cartella]
                                [--max-punti 300000] [--sezione xz|yz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def leggi_ply_segmentato(percorso: str | Path):
    """Legge il PLY binario prodotto da mepseg (x,y,z,rgb,classe)."""
    with open(percorso, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            riga = f.readline()
            if not riga:
                raise ValueError(f"Header PLY non valido: {percorso}")
            header += riga
        dt = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                ("classe", "u1"),
            ]
        )
        return np.fromfile(f, dtype=dt)


def principale() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mepseg.classi import COLORI_CLASSI, ClasseMEP, NOMI_CLASSI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ingresso", help="PLY segmentato prodotto da mepseg")
    parser.add_argument("-o", "--output", help="cartella PNG (default: quella del PLY)")
    parser.add_argument("--max-punti", type=int, default=300_000)
    parser.add_argument("--sezione", choices=["xz", "yz"], default="yz")
    argomenti = parser.parse_args()

    d = leggi_ply_segmentato(argomenti.ingresso)
    if len(d) > argomenti.max_punti:
        rng = np.random.default_rng(0)
        d = d[rng.choice(len(d), argomenti.max_punti, replace=False)]
    colori = np.column_stack([d["red"], d["green"], d["blue"]]) / 255.0
    mep = d["classe"] != 0

    cartella = Path(argomenti.output or Path(argomenti.ingresso).parent)
    cartella.mkdir(parents=True, exist_ok=True)
    nome = Path(argomenti.ingresso).stem

    legenda = [
        plt.Line2D(
            [0], [0], marker="o", ls="",
            color=np.array(COLORI_CLASSI[c]) / 255.0, label=NOMI_CLASSI[c],
        )
        for c in ClasseMEP
    ]

    fig, assi = plt.subplots(1, 2, figsize=(18, 9), facecolor="black")
    viste = [(np.ones(len(d), bool), "Classi (pianta)"), (mep, "Solo MEP (pianta)")]
    for ax, (maschera, titolo) in zip(assi, viste):
        ax.set_facecolor("black")
        ax.scatter(d["x"][maschera], d["y"][maschera], s=0.25,
                   c=colori[maschera], marker=".")
        ax.set_title(titolo, color="white")
        ax.set_aspect("equal")
        ax.tick_params(colors="gray")
    fig.legend(handles=legenda, loc="lower center", ncol=5,
               facecolor="black", labelcolor="white", edgecolor="gray")
    percorso_pianta = cartella / f"{nome}_pianta.png"
    plt.savefig(percorso_pianta, dpi=100, bbox_inches="tight", facecolor="black")
    print(f"anteprima: {percorso_pianta}")

    orizz = "x" if argomenti.sezione == "xz" else "y"
    fig2, ax = plt.subplots(figsize=(20, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(d[orizz], d["z"], s=0.25, c=colori, marker=".")
    ax.set_title(f"Sezione {argomenti.sezione.upper()}", color="white")
    ax.set_aspect("equal")
    ax.tick_params(colors="gray")
    fig2.legend(handles=legenda, loc="lower center", ncol=5,
                facecolor="black", labelcolor="white", edgecolor="gray")
    percorso_sezione = cartella / f"{nome}_sezione.png"
    plt.savefig(percorso_sezione, dpi=100, bbox_inches="tight", facecolor="black")
    print(f"anteprima: {percorso_sezione}")


if __name__ == "__main__":
    principale()
