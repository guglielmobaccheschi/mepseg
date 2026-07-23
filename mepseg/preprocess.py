"""Pre-elaborazione: sottocampionamento voxel, rimozione outlier, normali."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def sottocampiona_e_prepara(
    punti: np.ndarray, cfg: dict, colori: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sottocampiona la nuvola a voxel e stima le normali.

    Ritorna (punti_ridotti (M,3), normali (M,3), colori_ridotti (M,3) uint8 o
    None). Se ``colori`` (N,3 uint8) e' fornito, viene portato ATTRAVERSO il
    voxel e la rimozione outlier restando allineato ai punti (Open3D media il
    colore per voxel come per le coordinate): serve al taglio per colore del
    clustering. La classificazione avviene sulla nuvola ridotta e viene poi
    ripropagata alla risoluzione originale con :func:`propaga_etichette`.
    """
    import open3d as o3d

    voxel = float(cfg.get("voxel", 0.03))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(punti)
    ha_colore = colori is not None and len(colori) == len(punti)
    if ha_colore:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colori, dtype=np.float64) / 255.0
        )

    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    if cfg.get("rimuovi_outlier", True) and len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(cfg.get("outlier_vicini", 16)),
            std_ratio=float(cfg.get("outlier_std", 2.5)),
        )

    raggio_normali = max(voxel * 4.0, 0.06)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=raggio_normali, max_nn=30)
    )
    colori_ridotti = None
    if ha_colore and pcd.has_colors():
        colori_ridotti = (
            np.asarray(pcd.colors) * 255.0
        ).round().clip(0, 255).astype(np.uint8)
    return np.asarray(pcd.points), np.asarray(pcd.normals), colori_ridotti


def propaga_etichette(
    punti_ridotti: np.ndarray,
    etichette_ridotte: np.ndarray,
    punti_originali: np.ndarray,
    blocco: int = 2_000_000,
) -> np.ndarray:
    """Propaga le etichette dalla nuvola ridotta a quella a piena risoluzione.

    Ogni punto originale eredita l'etichetta del punto ridotto piu' vicino
    (nearest neighbor su KD-tree, elaborato a blocchi per contenere la RAM).
    """
    albero = cKDTree(punti_ridotti)
    etichette = np.empty(len(punti_originali), dtype=etichette_ridotte.dtype)
    for inizio in range(0, len(punti_originali), blocco):
        fine = min(inizio + blocco, len(punti_originali))
        _, indici = albero.query(punti_originali[inizio:fine], k=1, workers=-1)
        etichette[inizio:fine] = etichette_ridotte[indici]
    return etichette
