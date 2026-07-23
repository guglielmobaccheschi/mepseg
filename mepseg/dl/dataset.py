"""Dataset PyTorch per il fine-tuning della rete su scansioni etichettate.

Formato atteso: gruppi di file nella stessa cartella (train/ o val/)
  - ``nome.npy`` (N, 3 float) oppure ``nome.ply``/``.las``/``.laz``/``.pcd``
  - ``nome_etichette.npy``  : array (N,) uint8 con i codici classe
                              (vedi :class:`mepseg.classi.ClasseMEP`)
  - ``nome_fiducia.npy``    : opzionale, array (N,) uint8 con la provenienza
                              dell'etichetta — 2 = confermata dall'utente in
                              Fase 4 (peso pieno), 1 = assegnata dalle regole
                              (peso ridotto), 0 = incerta (esclusa dalla
                              loss). Se il file manca, tutto vale 1.

I ritagli si generano con ``mepseg-esporta-dataset`` (o gli script di
lavoro) a partire dai run della pipeline geometrica + conferme di Fase 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Il modulo DL richiede PyTorch: pip install .[dl]"
    ) from exc

from .inferenza import RAPPORTO_SUB, prepara_ingressi

ESTENSIONI_NUVOLA = (".npy", ".ply", ".las", ".laz", ".pcd")
SUFFISSI_AUSILIARI = ("_etichette", "_fiducia")


class DatasetNuvoleEtichettate(Dataset):
    """Estrae blocchi casuali di punti da scansioni etichettate.

    Con ``augmenta=True`` ogni blocco riceve rotazione casuale attorno a z,
    specchiatura, jitter gaussiano 5 mm e scala globale ±5% — mai rotazioni
    fuori piano: la gravita' e' un'informazione (dist da soffitto/pavimento,
    verticalita') e va preservata.
    """

    def __init__(
        self,
        cartella: str | Path,
        punti_per_blocco: int = 16384,
        blocchi_per_scansione: int = 8,
        n_livelli: int = 4,
        k_vicini: int = 16,
        augmenta: bool = False,
    ):
        self.cartella = Path(cartella)
        self.punti_per_blocco = punti_per_blocco
        self.blocchi_per_scansione = blocchi_per_scansione
        self.n_livelli = n_livelli
        self.k_vicini = k_vicini
        self.augmenta = augmenta

        self.scansioni: list[Path] = []
        for percorso in sorted(self.cartella.iterdir()):
            if percorso.suffix.lower() not in ESTENSIONI_NUVOLA:
                continue
            if any(percorso.stem.endswith(s) for s in SUFFISSI_AUSILIARI):
                continue  # file di etichette/fiducia, non nuvole
            etichette = percorso.with_name(percorso.stem + "_etichette.npy")
            if etichette.exists():
                self.scansioni.append(percorso)
        if not self.scansioni:
            raise FileNotFoundError(
                f"Nessuna coppia nuvola/etichette trovata in {self.cartella}"
            )
        # i ritagli sono piccoli (decine di MB): si tengono in RAM invece di
        # rileggere il file a ogni blocco estratto
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _carica(self, percorso: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if percorso not in self._cache:
            if percorso.suffix.lower() == ".npy":
                punti = np.load(percorso).astype(np.float64)[:, :3]
            else:
                from ..io_nuvole import carica_nuvola

                punti = carica_nuvola(percorso).punti
            etichette = np.load(
                percorso.with_name(percorso.stem + "_etichette.npy")
            ).astype(np.int64)
            percorso_fiducia = percorso.with_name(percorso.stem + "_fiducia.npy")
            if percorso_fiducia.exists():
                fiducia = np.load(percorso_fiducia).astype(np.int64)
            else:
                fiducia = np.ones(len(punti), dtype=np.int64)
            if not (len(punti) == len(etichette) == len(fiducia)):
                raise ValueError(
                    f"{percorso.name}: punti/etichette/fiducia di lunghezze "
                    f"diverse ({len(punti)}/{len(etichette)}/{len(fiducia)})"
                )
            self._cache[percorso] = (punti, etichette, fiducia)
        return self._cache[percorso]

    def __len__(self) -> int:
        return len(self.scansioni) * self.blocchi_per_scansione

    def __getitem__(self, indice: int):
        percorso = self.scansioni[indice % len(self.scansioni)]
        punti_tutti, etichette_tutte, fiducia_tutta = self._carica(percorso)

        # blocco sferico attorno a un punto seme casuale
        seme = punti_tutti[np.random.randint(len(punti_tutti))]
        distanze = np.linalg.norm(punti_tutti - seme, axis=1)
        idx = np.argsort(distanze)[: self.punti_per_blocco]
        if len(idx) < self.punti_per_blocco:  # scansione piccola: ricampiona
            extra = np.random.choice(idx, self.punti_per_blocco - len(idx))
            idx = np.concatenate([idx, extra])

        multiplo = RAPPORTO_SUB ** self.n_livelli
        n_utile = (len(idx) // multiplo) * multiplo
        idx = idx[:n_utile]

        punti = punti_tutti[idx]
        punti = punti - punti.mean(axis=0)

        if self.augmenta:
            angolo = np.random.uniform(0.0, 2.0 * np.pi)
            cos_a, sin_a = np.cos(angolo), np.sin(angolo)
            rot = np.array(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
            )
            punti = punti @ rot.T
            if np.random.random() < 0.5:
                punti[:, 0] = -punti[:, 0]
            punti = punti * np.random.uniform(0.95, 1.05)
            punti = punti + np.random.normal(0.0, 0.005, size=punti.shape)

        ingressi = prepara_ingressi(
            punti, self.n_livelli, self.k_vicini, torch.device("cpu")
        )
        # prepara_ingressi aggiunge la dimensione batch: qui la togliamo,
        # il DataLoader la ricrea impilando i campioni.
        ingressi = {
            chiave: [t[0] for t in valore] for chiave, valore in ingressi.items()
        }
        ingressi["feature"] = torch.from_numpy(punti.T).float()
        bersaglio = torch.from_numpy(etichette_tutte[idx])
        fiducia = torch.from_numpy(fiducia_tutta[idx])
        return ingressi, bersaglio, fiducia
