"""Caricamento e salvataggio di nuvole di punti in vari formati.

Formati supportati in lettura:
  - .ply / .pcd  (Open3D)
  - .xyz / .txt / .asc  (testo: x y z [r g b] oppure x y z [intensita'])
  - .las / .laz  (laspy)
  - .e57         (pye57, dipendenza opzionale)

In scrittura la pipeline produce un .ply colorato con campo ``classe`` e,
se richiesto, un .las con campo ``classification`` standard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classi import colora_etichette


@dataclass
class Nuvola:
    """Contenitore minimale per una nuvola di punti."""

    punti: np.ndarray                     # (N, 3) float64
    colori: np.ndarray | None = None      # (N, 3) uint8
    intensita: np.ndarray | None = None   # (N,) float32
    origine: str = ""                     # percorso del file di provenienza

    def __post_init__(self) -> None:
        self.punti = np.ascontiguousarray(self.punti, dtype=np.float64)
        if self.punti.ndim != 2 or self.punti.shape[1] != 3:
            raise ValueError(f"Attesi punti (N,3), ottenuto {self.punti.shape}")

    def __len__(self) -> int:
        return len(self.punti)


ESTENSIONI_SUPPORTATE = {".ply", ".pcd", ".xyz", ".txt", ".asc", ".las", ".laz", ".e57"}


def carica_nuvola(percorso: str | Path) -> Nuvola:
    """Carica una nuvola di punti riconoscendo il formato dall'estensione."""
    percorso = Path(percorso)
    if not percorso.exists():
        raise FileNotFoundError(f"File non trovato: {percorso}")
    est = percorso.suffix.lower()
    if est in (".ply", ".pcd"):
        return _carica_open3d(percorso)
    if est in (".xyz", ".txt", ".asc"):
        return _carica_testo(percorso)
    if est in (".las", ".laz"):
        return _carica_las(percorso)
    if est == ".e57":
        return _carica_e57(percorso)
    raise ValueError(
        f"Formato '{est}' non supportato. Formati validi: {sorted(ESTENSIONI_SUPPORTATE)}"
    )


def _carica_open3d(percorso: Path) -> Nuvola:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(percorso))
    if len(pcd.points) == 0:
        raise ValueError(f"Nessun punto letto da {percorso}")
    colori = None
    if pcd.has_colors():
        colori = (np.asarray(pcd.colors) * 255.0).round().astype(np.uint8)
    return Nuvola(np.asarray(pcd.points), colori, origine=str(percorso))


def _carica_testo(percorso: Path) -> Nuvola:
    dati = np.loadtxt(percorso, dtype=np.float64, ndmin=2)
    if dati.shape[1] < 3:
        raise ValueError(f"{percorso}: servono almeno 3 colonne (x y z)")
    punti = dati[:, :3]
    colori = None
    intensita = None
    if dati.shape[1] >= 6:
        rgb = dati[:, 3:6]
        # euristica: valori 0-1 oppure 0-255
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        colori = rgb.round().clip(0, 255).astype(np.uint8)
    elif dati.shape[1] == 4:
        intensita = dati[:, 3].astype(np.float32)
    return Nuvola(punti, colori, intensita, origine=str(percorso))


def _carica_las(percorso: Path) -> Nuvola:
    import laspy

    las = laspy.read(str(percorso))
    punti = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    colori = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        if rgb.max() > 255:  # LAS memorizza tipicamente RGB a 16 bit
            rgb = rgb / 257.0
        colori = rgb.round().clip(0, 255).astype(np.uint8)
    intensita = None
    if "intensity" in las.point_format.dimension_names:
        intensita = np.asarray(las.intensity, dtype=np.float32)
    return Nuvola(punti, colori, intensita, origine=str(percorso))


def _carica_e57(percorso: Path) -> Nuvola:
    try:
        import pye57
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Per leggere file E57 installare la dipendenza opzionale: pip install pye57"
        ) from exc

    e57 = pye57.E57(str(percorso))
    blocchi_punti, blocchi_colori, blocchi_int = [], [], []
    for i in range(e57.scan_count):
        scan = e57.read_scan(i, ignore_missing_fields=True, colors=True, intensity=True)
        xyz = np.column_stack([
            scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]
        ]).astype(np.float64)
        blocchi_punti.append(xyz)
        if "colorRed" in scan:
            blocchi_colori.append(np.column_stack([
                scan["colorRed"], scan["colorGreen"], scan["colorBlue"]
            ]).astype(np.uint8))
        if "intensity" in scan:
            blocchi_int.append(np.asarray(scan["intensity"], dtype=np.float32))
    punti = np.vstack(blocchi_punti)
    colori = np.vstack(blocchi_colori) if len(blocchi_colori) == e57.scan_count else None
    intensita = (
        np.concatenate(blocchi_int) if len(blocchi_int) == e57.scan_count else None
    )
    return Nuvola(punti, colori, intensita, origine=str(percorso))


# ---------------------------------------------------------------------------
# Scrittura
# ---------------------------------------------------------------------------

def salva_ply_classificata(
    percorso: str | Path, punti: np.ndarray, etichette: np.ndarray
) -> Path:
    """Salva un PLY binario con colore per classe e proprieta' scalare ``classe``.

    Il campo scalare viene scritto come proprieta' PLY aggiuntiva ``classe``
    (uchar), leggibile in CloudCompare come scalar field.
    """
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    colori = colora_etichette(etichette)
    n = len(punti)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classe\n"
        "end_header\n"
    )
    record = np.zeros(
        n,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("classe", "u1"),
        ],
    )
    record["x"], record["y"], record["z"] = punti[:, 0], punti[:, 1], punti[:, 2]
    record["red"], record["green"], record["blue"] = colori[:, 0], colori[:, 1], colori[:, 2]
    record["classe"] = etichette.astype(np.uint8)
    with open(percorso, "wb") as f:
        f.write(header.encode("ascii"))
        record.tofile(f)
    return percorso


def leggi_ply_classificata(percorso: str | Path) -> np.ndarray:
    """Rilegge il PLY binario prodotto da :func:`salva_ply_classificata`.

    Ritorna un array strutturato con campi x, y, z, red, green, blue, classe.
    """
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


def salva_las_classificata(
    percorso: str | Path, punti: np.ndarray, etichette: np.ndarray
) -> Path:
    """Salva un LAS 1.4 con RGB per classe e campo ``classification``."""
    import laspy

    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    colori = colora_etichette(etichette).astype(np.uint16) * 257  # RGB a 16 bit

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.offsets = punti.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = punti[:, 0], punti[:, 1], punti[:, 2]
    las.red, las.green, las.blue = colori[:, 0], colori[:, 1], colori[:, 2]
    las.classification = etichette.astype(np.uint8)
    las.write(str(percorso))
    return percorso
