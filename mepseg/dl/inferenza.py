"""Inferenza della rete: preparazione degli ingressi e predizione a blocchi.

La ricerca KNN e il sottocampionamento casuale sono precomputati su CPU
(scikit-learn), la rete gira su GPU se disponibile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RAPPORTO_SUB = 4  # fattore di sottocampionamento per livello encoder


def prepara_ingressi(punti: np.ndarray, n_livelli: int, k: int, device) -> dict:
    """Costruisce la piramide di coordinate/indici richiesta dalla rete.

    ``punti``: (N, 3) gia' normalizzati e con N multiplo di RAPPORTO_SUB**n_livelli.
    """
    import torch
    from sklearn.neighbors import NearestNeighbors

    xyz_livelli, idx_vicini, idx_sub, idx_interp = [], [], [], []
    xyz = punti
    for _ in range(n_livelli):
        nn = NearestNeighbors(n_neighbors=k).fit(xyz)
        _, vicini = nn.kneighbors(xyz)                    # (N_l, K)
        n_sub = len(xyz) // RAPPORTO_SUB
        # sottocampionamento casuale: i primi n_sub di una permutazione
        perm = np.random.permutation(len(xyz))
        idx_campione = perm[:n_sub]
        xyz_sub = xyz[idx_campione]
        # per l'upsampling: per ogni punto fine, il punto grosso piu' vicino
        nn_sub = NearestNeighbors(n_neighbors=1).fit(xyz_sub)
        _, interp = nn_sub.kneighbors(xyz)                # (N_l, 1)

        xyz_livelli.append(torch.from_numpy(xyz.T[None]).float().to(device))
        idx_vicini.append(torch.from_numpy(vicini[None]).long().to(device))
        idx_sub.append(
            torch.from_numpy(vicini[idx_campione][None]).long().to(device)
        )
        idx_interp.append(torch.from_numpy(interp[None]).long().to(device))
        xyz = xyz_sub

    return {
        "xyz": xyz_livelli,
        "idx_vicini": idx_vicini,
        "idx_sub": idx_sub,
        "idx_interp": idx_interp,
    }


def carica_modello(percorso_pesi: str | Path, device):
    import torch

    from .rete import ReteSegmentazione

    from ..classi import NUM_CLASSI

    checkpoint = torch.load(str(percorso_pesi), map_location=device)
    cfg_rete = checkpoint.get("config", {})
    modello = ReteSegmentazione(
        d_in=int(cfg_rete.get("d_in", 3)),
        num_classi=int(cfg_rete.get("num_classi", NUM_CLASSI)),
        d_encoder=tuple(cfg_rete.get("d_encoder", (16, 64, 128, 256))),
    )
    modello.load_state_dict(checkpoint["stato_modello"])
    modello.to(device).eval()
    return modello


def segmenta_dl(punti: np.ndarray, percorso_pesi: str | Path, cfg: dict) -> np.ndarray:
    """Predice le etichette per punto con la rete, elaborando a blocchi.

    Ritorna un array (N,) uint8 di codici classe.
    """
    import torch

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
        else "cpu"
    )
    modello = carica_modello(percorso_pesi, device)
    n_livelli = len(modello.blocchi)
    k = int(cfg.get("k_vicini", 16))
    max_punti = int(cfg.get("max_punti_blocco", 65536))
    multiplo = RAPPORTO_SUB ** n_livelli

    etichette = np.zeros(len(punti), dtype=np.uint8)
    ordine = np.random.permutation(len(punti))

    with torch.no_grad():
        for inizio in range(0, len(punti), max_punti):
            idx_blocco = ordine[inizio: inizio + max_punti]
            blocco = punti[idx_blocco]

            # padding: N deve essere multiplo di RAPPORTO_SUB**n_livelli
            n_pad = (-len(blocco)) % multiplo
            if n_pad:
                extra = np.random.choice(len(blocco), n_pad)
                blocco = np.vstack([blocco, blocco[extra]])

            centro = blocco.mean(axis=0)
            blocco_norm = blocco - centro

            ingressi = prepara_ingressi(blocco_norm, n_livelli, k, device)
            feature = torch.from_numpy(blocco_norm.T[None]).float().to(device)
            if modello.d_in > 3:  # canali extra (es. RGB) non disponibili: zero
                zeri = torch.zeros(
                    1, modello.d_in - 3, feature.shape[2], device=device
                )
                feature = torch.cat([feature, zeri], dim=1)
            ingressi["feature"] = feature

            logit = modello(ingressi)  # (1, C, N)
            pred = logit.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            etichette[idx_blocco] = pred[: len(idx_blocco)]

    return etichette
