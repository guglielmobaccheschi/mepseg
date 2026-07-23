"""Fine-tuning della rete di segmentazione sui ritagli etichettati.

Uso:
  python -m mepseg.dl.allenamento dataset_mep -o pesi/rete_mep_v1.pth \
      --epoche 80 --batch 4

La cartella dataset deve contenere ``train/`` e ``val/`` con le triple
``nome.npy`` (+ ``_etichette.npy`` + ``_fiducia.npy``) descritte in
:mod:`mepseg.dl.dataset`.

Scelte fisse del progetto (vedi documento di Fase 4):
- la loss pesa ogni punto per la PROVENIENZA della sua etichetta:
  fiducia 2 (confermata dall'utente) = peso pieno, fiducia 1 (dalle regole)
  = peso ridotto, fiducia 0 (incerta: generico, arredo, waste) = esclusa.
  La rete impara soprattutto da cio' che l'utente ha garantito, senza
  buttare il segnale debole delle regole;
- pesi di classe dall'inverso della frequenza (le luci sono <1% dei punti:
  senza pesi la rete impara "tutto e' apparecchiatura");
- augmentation solo attorno all'asse z (la gravita' e' un'informazione);
- il checkpoint migliore (mIoU su val) e' compatibile con
  ``inferenza.carica_modello`` e quindi con ``mepseg --pesi``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("mepseg")


def _sposta(ingressi: dict, device) -> dict:
    """Porta sul device la piramide di tensori preparata dal dataset."""
    esito = {}
    for chiave, valore in ingressi.items():
        if isinstance(valore, list):
            esito[chiave] = [t.to(device) for t in valore]
        else:
            esito[chiave] = valore.to(device)
    return esito


def calcola_pesi_classe(cartella_train: Path, num_classi: int) -> np.ndarray:
    """Peso per classe dall'inverso della frequenza sui punti supervisati
    (fiducia >= 1). Le classi assenti pesano 0: non possono essere apprese
    e non devono gonfiare la normalizzazione."""
    from .dataset import ESTENSIONI_NUVOLA, SUFFISSI_AUSILIARI

    conteggi = np.zeros(num_classi, dtype=np.float64)
    for percorso in sorted(cartella_train.iterdir()):
        if percorso.suffix.lower() not in ESTENSIONI_NUVOLA:
            continue
        if any(percorso.stem.endswith(s) for s in SUFFISSI_AUSILIARI):
            continue
        percorso_eti = percorso.with_name(percorso.stem + "_etichette.npy")
        if not percorso_eti.exists():
            continue
        etichette = np.load(percorso_eti).astype(np.int64)
        percorso_fid = percorso.with_name(percorso.stem + "_fiducia.npy")
        if percorso_fid.exists():
            supervisati = np.load(percorso_fid).astype(np.int64) >= 1
        else:
            supervisati = np.ones(len(etichette), dtype=bool)
        conteggi += np.bincount(
            etichette[supervisati], minlength=num_classi
        )[:num_classi]

    pesi = np.zeros(num_classi, dtype=np.float64)
    presenti = conteggi > 0
    pesi[presenti] = 1.0 / conteggi[presenti]
    # normalizzazione: peso medio 1 sulle classi presenti, cosi' la scala
    # della loss non dipende da quante classi ci sono nel dataset
    pesi[presenti] *= presenti.sum() / pesi[presenti].sum()
    return pesi


def principale(argv: list[str] | None = None) -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from ..classi import NOMI_CLASSI, NUM_CLASSI, ClasseMEP
    from .dataset import DatasetNuvoleEtichettate
    from .rete import ReteSegmentazione

    parser = argparse.ArgumentParser(
        prog="mepseg-allena",
        description="Fine-tuning della rete sui ritagli etichettati",
    )
    parser.add_argument("dataset", help="cartella con train/ e val/")
    parser.add_argument(
        "-o", "--output", default="pesi/rete_mep_v1.pth",
        help="checkpoint di destinazione (default: pesi/rete_mep_v1.pth)",
    )
    parser.add_argument("--epoche", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--decadimento-lr", type=float, default=0.95,
        help="fattore moltiplicativo del learning rate per epoca",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--punti-blocco", type=int, default=16384)
    parser.add_argument(
        "--blocchi", type=int, default=8, help="blocchi estratti per scansione"
    )
    parser.add_argument("--k-vicini", type=int, default=16)
    parser.add_argument(
        "--d-in", type=int, default=3,
        help="canali per punto: 3 = xyz, 6 = xyz+rgb (quando la cache avra' il colore)",
    )
    parser.add_argument(
        "--peso-fiducia-regole", type=float, default=0.4,
        help="peso dei punti etichettati SOLO dalle regole (fiducia 1)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seme", type=int, default=0)
    argomenti = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    torch.manual_seed(argomenti.seme)
    np.random.seed(argomenti.seme)
    device = torch.device(
        "cuda"
        if argomenti.device == "cuda"
        or (argomenti.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    log.info("Device: %s", device)

    cartella = Path(argomenti.dataset)
    dataset_train = DatasetNuvoleEtichettate(
        cartella / "train",
        punti_per_blocco=argomenti.punti_blocco,
        blocchi_per_scansione=argomenti.blocchi,
        k_vicini=argomenti.k_vicini,
        augmenta=True,
    )
    dataset_val = DatasetNuvoleEtichettate(
        cartella / "val",
        punti_per_blocco=argomenti.punti_blocco,
        blocchi_per_scansione=argomenti.blocchi,
        k_vicini=argomenti.k_vicini,
        augmenta=False,
    )
    log.info(
        "Dataset: %d scansioni train, %d val (blocchi da %d punti)",
        len(dataset_train.scansioni), len(dataset_val.scansioni),
        argomenti.punti_blocco,
    )
    # num_workers=0: i blocchi sono gia' in cache RAM e su Windows i worker
    # ricaricherebbero tutto in ogni processo
    loader_train = DataLoader(
        dataset_train, batch_size=argomenti.batch, shuffle=True, num_workers=0
    )
    loader_val = DataLoader(
        dataset_val, batch_size=argomenti.batch, shuffle=False, num_workers=0
    )

    pesi_classe = calcola_pesi_classe(cartella / "train", NUM_CLASSI)
    for codice, peso in enumerate(pesi_classe):
        if peso > 0:
            log.info(
                "  peso classe %-24s %.3f", NOMI_CLASSI[ClasseMEP(codice)], peso
            )
    pesi_classe_t = torch.from_numpy(pesi_classe).float().to(device)

    modello = ReteSegmentazione(d_in=argomenti.d_in, num_classi=NUM_CLASSI).to(device)
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=argomenti.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        ottimizzatore, gamma=argomenti.decadimento_lr
    )

    # peso per punto dalla provenienza dell'etichetta (indice = fiducia)
    pesi_fiducia = torch.tensor(
        [0.0, argomenti.peso_fiducia_regole, 1.0], device=device
    )

    percorso_out = Path(argomenti.output)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)
    config_rete = {
        "d_in": argomenti.d_in,
        "num_classi": NUM_CLASSI,
        "d_encoder": [16, 64, 128, 256],
    }

    def loss_pesata(logit, bersaglio, fiducia):
        # logit (B, C, N), bersaglio (B, N), fiducia (B, N)
        perdita = F.cross_entropy(
            logit, bersaglio, weight=pesi_classe_t, reduction="none"
        )
        peso = pesi_fiducia[fiducia.clamp(max=2)]
        totale = peso.sum()
        if totale <= 0:
            return None  # blocco interamente incerto: nessun segnale
        return (perdita * peso).sum() / totale

    miglior_miou = -1.0
    for epoca in range(1, argomenti.epoche + 1):
        inizio = time.time()
        modello.train()
        somma_loss, n_batch = 0.0, 0
        for ingressi, bersaglio, fiducia in loader_train:
            ingressi = _sposta(ingressi, device)
            bersaglio = bersaglio.to(device)
            fiducia = fiducia.to(device)
            perdita = loss_pesata(modello(ingressi), bersaglio, fiducia)
            if perdita is None:
                continue
            ottimizzatore.zero_grad()
            perdita.backward()
            ottimizzatore.step()
            somma_loss += float(perdita.item())
            n_batch += 1
        scheduler.step()

        # validazione: blocchi DETERMINISTICI (stesso seme a ogni epoca),
        # altrimenti la mIoU oscillerebbe per il solo campionamento
        np.random.seed(argomenti.seme)
        modello.eval()
        confusione = np.zeros((NUM_CLASSI, NUM_CLASSI), dtype=np.int64)
        with torch.no_grad():
            for ingressi, bersaglio, fiducia in loader_val:
                ingressi = _sposta(ingressi, device)
                pred = modello(ingressi).argmax(dim=1).cpu().numpy().ravel()
                vero = bersaglio.numpy().ravel()
                validi = fiducia.numpy().ravel() >= 1
                np.add.at(confusione, (vero[validi], pred[validi]), 1)

        intersezione = np.diag(confusione).astype(np.float64)
        unione = (
            confusione.sum(axis=0) + confusione.sum(axis=1) - np.diag(confusione)
        ).astype(np.float64)
        presenti = confusione.sum(axis=1) > 0
        iou = np.zeros(NUM_CLASSI)
        iou[unione > 0] = intersezione[unione > 0] / unione[unione > 0]
        miou = float(iou[presenti].mean()) if presenti.any() else 0.0

        log.info(
            "Epoca %3d/%d  loss %.4f  mIoU val %.3f  lr %.2e  (%.0f s)",
            epoca, argomenti.epoche,
            somma_loss / max(n_batch, 1), miou,
            scheduler.get_last_lr()[0], time.time() - inizio,
        )
        for codice in np.where(presenti)[0]:
            log.debug(
                "    IoU %-24s %.3f",
                NOMI_CLASSI[ClasseMEP(int(codice))], iou[codice],
            )

        if miou > miglior_miou:
            miglior_miou = miou
            torch.save(
                {
                    "stato_modello": modello.state_dict(),
                    "config": config_rete,
                    "epoca": epoca,
                    "miou_val": miou,
                    "iou_per_classe": {
                        NOMI_CLASSI[ClasseMEP(int(c))]: round(float(iou[c]), 4)
                        for c in np.where(presenti)[0]
                    },
                },
                percorso_out,
            )
            log.info(
                "  nuovo checkpoint migliore (mIoU %.3f): %s", miou, percorso_out
            )

    log.info("Fine: mIoU migliore %.3f in %s", miglior_miou, percorso_out)
    return 0


if __name__ == "__main__":
    sys.exit(principale())
