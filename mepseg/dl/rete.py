"""Rete neurale per la segmentazione semantica di nuvole di punti.

Architettura encoder-decoder con sottocampionamento casuale tra i livelli
e aggregazione locale attentiva dei vicini (riferimento scientifico nella
bibliografia della tesi). Implementazione propria, pensata per chiarezza e
portabilita'
(CPU o GPU CUDA senza operatori custom): la ricerca dei vicini e il
sottocampionamento casuale vengono precomputati fuori dalla rete
(vedi :mod:`mepseg.dl.inferenza`) e passati come indici.

Architettura: encoder con blocchi residuali dilatati (LocSE + attentive
pooling) e sottocampionamento casuale 1/4 a ogni livello, decoder con
upsampling nearest-neighbor e skip connection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp1d(d_in: int, d_out: int, bn: bool = True, act: bool = True) -> nn.Sequential:
    strati: list[nn.Module] = [nn.Conv1d(d_in, d_out, 1, bias=not bn)]
    if bn:
        strati.append(nn.BatchNorm1d(d_out, momentum=0.99, eps=1e-6))
    if act:
        strati.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*strati)


def raccogli_vicini(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather delle feature dei vicini.

    x: (B, C, N) feature; idx: (B, N', K) indici dei vicini in [0, N).
    Ritorna (B, C, N', K).
    """
    b, c, n = x.shape
    _, n2, k = idx.shape
    idx_flat = idx.reshape(b, 1, n2 * k).expand(-1, c, -1)
    raccolte = torch.gather(x, 2, idx_flat)
    return raccolte.reshape(b, c, n2, k)


class CodificaSpazialeLocale(nn.Module):
    """LocSE: codifica la geometria relativa dei K vicini di ogni punto."""

    def __init__(self, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(10, d_out, 1, bias=False),
            nn.BatchNorm2d(d_out, momentum=0.99, eps=1e-6),
            nn.LeakyReLU(0.2),
        )

    def forward(self, xyz: torch.Tensor, idx_vicini: torch.Tensor) -> torch.Tensor:
        # xyz: (B, 3, N); idx_vicini: (B, N, K)
        vicini = raccogli_vicini(xyz, idx_vicini)          # (B, 3, N, K)
        centri = xyz.unsqueeze(-1).expand_as(vicini)       # (B, 3, N, K)
        rel = centri - vicini
        dist = rel.norm(dim=1, keepdim=True)               # (B, 1, N, K)
        codifica = torch.cat([centri, vicini, rel, dist], dim=1)  # (B, 10, N, K)
        return self.mlp(codifica)


class PoolingAttentivo(nn.Module):
    """Aggrega le feature dei vicini con pesi appresi (softmax sui K vicini)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.punteggi = nn.Conv2d(d_in, d_in, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(d_in, d_out, 1, bias=False),
            nn.BatchNorm1d(d_out, momentum=0.99, eps=1e-6),
            nn.LeakyReLU(0.2),
        )

    def forward(self, feature_vicini: torch.Tensor) -> torch.Tensor:
        # feature_vicini: (B, C, N, K)
        pesi = F.softmax(self.punteggi(feature_vicini), dim=-1)
        aggregato = (pesi * feature_vicini).sum(dim=-1)  # (B, C, N)
        return self.mlp(aggregato)


class BloccoResidualeDilatato(nn.Module):
    """Blocco LFA: due passaggi LocSE + pooling attentivo con shortcut."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp_in = _mlp1d(d_in, d_out // 2)
        self.locse1 = CodificaSpazialeLocale(d_out // 2)
        self.pool1 = PoolingAttentivo(d_out, d_out // 2)
        self.locse2 = CodificaSpazialeLocale(d_out // 2)
        self.pool2 = PoolingAttentivo(d_out, d_out)
        self.mlp_out = _mlp1d(d_out, 2 * d_out, act=False)
        self.shortcut = _mlp1d(d_in, 2 * d_out, act=False)
        self.attivazione = nn.LeakyReLU(0.2)

    def forward(
        self, feature: torch.Tensor, xyz: torch.Tensor, idx_vicini: torch.Tensor
    ) -> torch.Tensor:
        f = self.mlp_in(feature)                                   # (B, d/2, N)
        geo = self.locse1(xyz, idx_vicini)                         # (B, d/2, N, K)
        f_vic = raccogli_vicini(f, idx_vicini)                     # (B, d/2, N, K)
        f = self.pool1(torch.cat([geo, f_vic], dim=1))             # (B, d/2, N)
        geo = self.locse2(xyz, idx_vicini)
        f_vic = raccogli_vicini(f, idx_vicini)
        f = self.pool2(torch.cat([geo, f_vic], dim=1))             # (B, d, N)
        return self.attivazione(self.mlp_out(f) + self.shortcut(feature))


class ReteSegmentazione(nn.Module):
    """Rete per la segmentazione semantica di nuvole di punti.

    Parametri
    ---------
    d_in: dimensione feature in ingresso per punto (3 = xyz; 6 = xyz+rgb)
    num_classi: numero di classi in uscita
    d_encoder: dimensioni d_out dei blocchi encoder
    """

    def __init__(
        self,
        d_in: int = 6,
        num_classi: int | None = None,
        d_encoder: tuple[int, ...] = (16, 64, 128, 256),
    ):
        if num_classi is None:
            from ..classi import NUM_CLASSI

            num_classi = NUM_CLASSI
        super().__init__()
        self.d_in = d_in
        self.num_classi = num_classi
        self.fc_inizio = _mlp1d(d_in, 8)

        self.blocchi = nn.ModuleList()
        d = 8
        dims_skip = []
        for d_out in d_encoder:
            self.blocchi.append(BloccoResidualeDilatato(d, d_out))
            d = 2 * d_out
            dims_skip.append(d)

        self.mlp_centro = _mlp1d(d, d)

        self.decoder = nn.ModuleList()
        for d_skip in reversed(dims_skip[:-1]):
            self.decoder.append(_mlp1d(d + d_skip, d_skip))
            d = d_skip
        self.decoder.append(_mlp1d(d + 8, 64))

        self.testa = nn.Sequential(
            _mlp1d(64, 64),
            _mlp1d(64, 32),
            nn.Dropout(0.5),
            nn.Conv1d(32, num_classi, 1),
        )

    def forward(self, ingressi: dict[str, list[torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        """Ingressi precomputati (vedi ``prepara_ingressi`` in inferenza.py):

        - ``feature``: (B, d_in, N) feature per punto al livello 0
        - ``xyz``: lista di (B, 3, N_l) coordinate per livello
        - ``idx_vicini``: lista di (B, N_l, K) KNN per livello
        - ``idx_sub``: lista di (B, N_{l+1}, K) vicini dei punti campionati
        - ``idx_interp``: lista di (B, N_l, 1) indice del punto grezzo piu'
          vicino, per l'upsampling del decoder

        Ritorna i logit (B, num_classi, N).
        """
        f = self.fc_inizio(ingressi["feature"])  # (B, 8, N)
        skip = [f]
        for livello, blocco in enumerate(self.blocchi):
            f = blocco(f, ingressi["xyz"][livello], ingressi["idx_vicini"][livello])
            # sottocampionamento casuale: max-pool sulle feature dei vicini
            # dei punti campionati (B, C, N_{l+1}, K) -> (B, C, N_{l+1})
            f = raccogli_vicini(f, ingressi["idx_sub"][livello]).max(dim=-1).values
            skip.append(f)

        f = self.mlp_centro(f)

        for livello, mlp in enumerate(self.decoder):
            idx = ingressi["idx_interp"][len(self.decoder) - 1 - livello]  # (B, N_l, 1)
            f = raccogli_vicini(f, idx).squeeze(-1)  # upsampling nearest
            f = mlp(torch.cat([f, skip[len(skip) - 2 - livello]], dim=1))

        return self.testa(f)
