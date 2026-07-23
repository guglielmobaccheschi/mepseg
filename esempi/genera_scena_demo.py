"""Genera una scena sintetica di test: un locale con impianti MEP tipici.

Crea ``esempi/scena_demo.ply`` con: pavimento, soffitto, due pareti,
una condotta circolare, una condotta rettangolare, tre tubazioni,
una tubazione sprinkler con calate e teste, quattro corpi illuminanti,
due diffusori, una passerella cavi e un'apparecchiatura a pavimento.

Uso:
    python esempi/genera_scena_demo.py
    mepseg esempi/scena_demo.ply -o output_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUMORE = 0.003          # sigma del rumore gaussiano (simula il rumore scanner)
PASSO_STRUTTURA = 0.035  # spaziatura punti su pavimenti/pareti
PASSO_OGGETTI = 0.02     # spaziatura punti sugli oggetti MEP

rng = np.random.default_rng(42)


def piano_xy(x0, x1, y0, y1, z, passo=PASSO_STRUTTURA):
    xx, yy = np.meshgrid(np.arange(x0, x1, passo), np.arange(y0, y1, passo))
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def piano_xz(x0, x1, z0, z1, y, passo=PASSO_STRUTTURA):
    xx, zz = np.meshgrid(np.arange(x0, x1, passo), np.arange(z0, z1, passo))
    return np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()])


def piano_yz(y0, y1, z0, z1, x, passo=PASSO_STRUTTURA):
    yy, zz = np.meshgrid(np.arange(y0, y1, passo), np.arange(z0, z1, passo))
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def cilindro_x(x0, x1, y, z, raggio, passo=PASSO_OGGETTI):
    """Superficie di cilindro con asse parallelo a X."""
    n_ang = max(int(2 * np.pi * raggio / passo), 8)
    angoli = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    xs = np.arange(x0, x1, passo)
    aa, xx = np.meshgrid(angoli, xs)
    return np.column_stack([
        xx.ravel(),
        y + raggio * np.cos(aa.ravel()),
        z + raggio * np.sin(aa.ravel()),
    ])


def cilindro_z(x, y, z0, z1, raggio, passo=PASSO_OGGETTI):
    """Superficie di cilindro verticale (asse Z)."""
    n_ang = max(int(2 * np.pi * raggio / passo), 8)
    angoli = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    zs = np.arange(z0, z1, passo)
    aa, zz = np.meshgrid(angoli, zs)
    return np.column_stack([
        x + raggio * np.cos(aa.ravel()),
        y + raggio * np.sin(aa.ravel()),
        zz.ravel(),
    ])


def sfera(x, y, z, raggio, passo=PASSO_OGGETTI):
    n = max(int(4 * np.pi * raggio**2 / passo**2), 30)
    direzioni = rng.normal(size=(n, 3))
    direzioni /= np.linalg.norm(direzioni, axis=1, keepdims=True)
    return np.array([x, y, z]) + raggio * direzioni


def scatola(cx, cy, cz, lx, ly, lz, passo=PASSO_OGGETTI, facce="tutte"):
    """Superficie di un parallelepipedo centrato in (cx, cy, cz)."""
    blocchi = []
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    if facce in ("tutte", "laterali+sopra", "laterali"):
        for sy in (-1, 1):  # facce y
            xx, zz = np.meshgrid(
                np.arange(-hx, hx, passo), np.arange(-hz, hz, passo)
            )
            blocchi.append(np.column_stack(
                [xx.ravel(), np.full(xx.size, sy * hy), zz.ravel()]
            ))
        for sx in (-1, 1):  # facce x
            yy, zz = np.meshgrid(
                np.arange(-hy, hy, passo), np.arange(-hz, hz, passo)
            )
            blocchi.append(np.column_stack(
                [np.full(yy.size, sx * hx), yy.ravel(), zz.ravel()]
            ))
    if facce in ("tutte", "laterali+sopra"):
        xx, yy = np.meshgrid(np.arange(-hx, hx, passo), np.arange(-hy, hy, passo))
        blocchi.append(np.column_stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, hz)]
        ))
    if facce == "tutte":
        xx, yy = np.meshgrid(np.arange(-hx, hx, passo), np.arange(-hy, hy, passo))
        blocchi.append(np.column_stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, -hz)]
        ))
    return np.vstack(blocchi) + np.array([cx, cy, cz])


def genera() -> np.ndarray:
    blocchi: list[np.ndarray] = []

    # --- struttura: locale 12 x 8 m, altezza 3.5 m -------------------------
    blocchi.append(piano_xy(0, 12, 0, 8, 0.0))    # pavimento
    blocchi.append(piano_xy(0, 12, 0, 8, 3.5))    # soffitto
    blocchi.append(piano_xz(0, 12, 0, 3.5, 0.0))  # parete y=0
    blocchi.append(piano_xz(0, 12, 0, 3.5, 8.0))  # parete y=8
    blocchi.append(piano_yz(0, 8, 0, 3.5, 0.0))   # parete x=0
    blocchi.append(piano_yz(0, 8, 0, 3.5, 12.0))  # parete x=12

    # gli impianti arrivano a filo parete: nella realta' proseguono
    # attraverso di essa (la regola di ancoraggio richiede continuita')
    # --- condotta circolare D600, sotto il soffitto ------------------------
    blocchi.append(cilindro_x(0.1, 11.9, y=2.0, z=3.0, raggio=0.30))

    # --- condotta rettangolare 600x400 --------------------------------------
    blocchi.append(scatola(6.0, 4.0, 2.9, 11.8, 0.60, 0.40, facce="tutte"))

    # --- tre tubazioni idroniche --------------------------------------------
    blocchi.append(cilindro_x(0.1, 11.9, y=5.5, z=3.2, raggio=0.05))
    blocchi.append(cilindro_x(0.1, 11.9, y=5.8, z=3.2, raggio=0.04))
    blocchi.append(cilindro_x(0.1, 11.9, y=6.1, z=3.2, raggio=0.06))

    # --- tubazione sprinkler con calate e teste ------------------------------
    blocchi.append(cilindro_x(0.1, 11.9, y=6.8, z=3.15, raggio=0.04))
    for x in np.arange(1.5, 11.5, 2.0):
        blocchi.append(cilindro_z(x, 6.8, 2.91, 3.11, raggio=0.02))  # calata
        blocchi.append(sfera(x, 6.8, 2.88, raggio=0.05))             # testa

    # --- corpi illuminanti lineari 1.2 x 0.3, sospesi sotto il soffitto -----
    for x, y in [(3.0, 1.0), (9.0, 1.0), (3.0, 7.3), (9.0, 7.3)]:
        blocchi.append(scatola(x, y, 3.35, 1.20, 0.30, 0.06, facce="tutte"))

    # --- diffusori quadrati 600x600 ------------------------------------------
    for x in (3.0, 8.0):
        blocchi.append(scatola(x, 2.95, 3.32, 0.60, 0.60, 0.06, facce="tutte"))

    # --- passerella cavi 300x80 (fondo + sponde) ------------------------------
    xx, yy = np.meshgrid(np.arange(0.1, 11.9, PASSO_OGGETTI),
                         np.arange(7.15, 7.45, PASSO_OGGETTI))
    blocchi.append(np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, 3.05)]
    ))
    for y in (7.15, 7.45):
        xx, zz = np.meshgrid(np.arange(0.1, 11.9, PASSO_OGGETTI),
                             np.arange(3.05, 3.13, PASSO_OGGETTI))
        blocchi.append(np.column_stack(
            [xx.ravel(), np.full(xx.size, y), zz.ravel()]
        ))

    # --- pilastri strutturali: uno tondo e uno rettangolare, da pavimento
    #     a soffitto (devono diventare PILASTRO, non apparecchiatura)
    blocchi.append(cilindro_z(5.0, 1.0, 0.0, 3.5, raggio=0.20))
    blocchi.append(
        scatola(7.0, 5.0, 1.75, 0.40, 0.60, 3.5, facce="laterali")
    )

    # --- apparecchiatura a pavimento (es. pompa/quadro) ----------------------
    blocchi.append(
        scatola(1.5, 6.3, 0.70, 1.20, 0.80, 1.40, facce="laterali+sopra")
    )

    # --- armadietto aperto (arredo): mezzo guscio cilindrico verticale
    #     appoggiato al pavimento, lontano da impianti e apparecchiature.
    #     Geometricamente e' una "condotta circolare", ma non e' ancorato in
    #     alto: la regola di contesto deve declassarlo ad arredo/non-MEP.
    armadietto = cilindro_z(9.5, 1.5, 0.0, 1.8, raggio=0.30)
    blocchi.append(armadietto[armadietto[:, 1] <= 1.5])

    punti = np.vstack(blocchi)
    punti += rng.normal(scale=RUMORE, size=punti.shape)
    return punti


def salva_ply(percorso: Path, punti: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(punti)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(percorso, "wb") as f:
        f.write(header.encode("ascii"))
        punti.astype("<f4").tofile(f)


if __name__ == "__main__":
    percorso = Path(__file__).resolve().parent / "scena_demo.ply"
    punti = genera()
    salva_ply(percorso, punti)
    print(f"Scena demo generata: {percorso} ({len(punti):,} punti)")
