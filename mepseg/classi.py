"""Tassonomia delle classi MEP e palette colori associata.

Ogni punto della nuvola riceve una di queste etichette. I colori sono
pensati per essere distinguibili in CloudCompare / ReCap su sfondo scuro.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class ClasseMEP(IntEnum):
    """Codici di classificazione scritti nel campo scalare di output."""

    STRUTTURA = 0            # struttura generica non-MEP (fallback: le
                             # superfici riconosciute usano PAVIMENTO,
                             # SOFFITTO e PARETE)
    CONDOTTA_CIRCOLARE = 1   # canali aria a sezione circolare
    CONDOTTA_RETTANGOLARE = 2  # canali aria a sezione rettangolare
    TUBAZIONE = 3            # tubi idronici, gas, scarichi
    PASSERELLA_CAVI = 4      # canaline e passerelle portacavi
    LUCE = 5                 # corpi illuminanti
    SPRINKLER = 6            # teste e calate sprinkler
    TERMINALE_ARIA = 7       # diffusori, griglie, bocchette
    APPARECCHIATURA = 8      # pompe, UTA, quadri, unita' compatte
    MEP_GENERICO = 9         # elemento MEP non riconducibile alle classi sopra
    NON_MEP_ARREDO = 10      # arredo/oggetti REALI non impiantistici (armadi,
                             # scrivanie, scaffali): non struttura, non MEP,
                             # non da modellare — ma esistono davvero
    PAVIMENTO = 11           # superfici orizzontali calpestabili
    SOFFITTO = 12            # superfici orizzontali di copertura
    PARETE = 13              # superfici verticali strutturali
    PILASTRO = 14            # colonne strutturali (tonde o rettangolari)
                             # che attraversano il locale da pavimento a
                             # soffitto
    TRAVE = 15               # travi di sostegno dei piani superiori: bande
                             # allungate che interrompono il soffitto (il
                             # corpo della trave e' spesso in ombra: la
                             # banda vuota nel soffitto E i punti che vi
                             # pendono dentro sono la sua firma)
    WASTE = 16               # rumore di scansione: riflessi, code, nuvolette
                             # rarefatte, gruppi eliminati dall'utente in
                             # Fase 4. Non e' un oggetto reale (a differenza
                             # dell'arredo): da scartare e da escludere dalle
                             # iterazioni di riconoscimento successive


NOMI_CLASSI: dict[ClasseMEP, str] = {
    ClasseMEP.STRUTTURA: "Struttura / non-MEP",
    ClasseMEP.CONDOTTA_CIRCOLARE: "Condotta circolare",
    ClasseMEP.CONDOTTA_RETTANGOLARE: "Condotta rettangolare",
    ClasseMEP.TUBAZIONE: "Tubazione",
    ClasseMEP.PASSERELLA_CAVI: "Passerella cavi",
    ClasseMEP.LUCE: "Luce",
    ClasseMEP.SPRINKLER: "Sprinkler",
    ClasseMEP.TERMINALE_ARIA: "Terminale aria",
    ClasseMEP.APPARECCHIATURA: "Apparecchiatura",
    ClasseMEP.MEP_GENERICO: "Elemento MEP generico",
    ClasseMEP.NON_MEP_ARREDO: "Arredo",
    ClasseMEP.PAVIMENTO: "Pavimento",
    ClasseMEP.SOFFITTO: "Soffitto",
    ClasseMEP.PARETE: "Parete",
    ClasseMEP.PILASTRO: "Pilastro",
    ClasseMEP.TRAVE: "Trave",
    ClasseMEP.WASTE: "Waste",
}

# RGB 0-255 per classe.
COLORI_CLASSI: dict[ClasseMEP, tuple[int, int, int]] = {
    ClasseMEP.STRUTTURA: (128, 128, 128),
    ClasseMEP.CONDOTTA_CIRCOLARE: (0, 150, 255),
    ClasseMEP.CONDOTTA_RETTANGOLARE: (0, 60, 220),
    ClasseMEP.TUBAZIONE: (0, 200, 80),
    ClasseMEP.PASSERELLA_CAVI: (255, 140, 0),
    ClasseMEP.LUCE: (255, 220, 0),
    ClasseMEP.SPRINKLER: (255, 0, 60),
    ClasseMEP.TERMINALE_ARIA: (0, 230, 230),
    ClasseMEP.APPARECCHIATURA: (170, 0, 255),
    ClasseMEP.MEP_GENERICO: (255, 105, 180),
    ClasseMEP.NON_MEP_ARREDO: (139, 87, 42),
    ClasseMEP.PAVIMENTO: (105, 105, 105),
    ClasseMEP.SOFFITTO: (210, 210, 210),
    ClasseMEP.PARETE: (160, 160, 160),
    ClasseMEP.PILASTRO: (80, 80, 80),
    ClasseMEP.TRAVE: (110, 110, 140),
    # nero morbido: si legge come "spento" senza sparire sullo sfondo
    # scuro del viewer ne' abbagliare su sfondo chiaro in CloudCompare
    ClasseMEP.WASTE: (40, 40, 45),
}

NUM_CLASSI = len(ClasseMEP)


def tabella_colori() -> np.ndarray:
    """Ritorna una lookup table (NUM_CLASSI, 3) uint8 indicizzabile per codice."""
    lut = np.zeros((NUM_CLASSI, 3), dtype=np.uint8)
    for classe, rgb in COLORI_CLASSI.items():
        lut[int(classe)] = rgb
    return lut


def colora_etichette(etichette: np.ndarray) -> np.ndarray:
    """Converte un array di codici classe (N,) in colori RGB (N, 3) uint8."""
    return tabella_colori()[np.asarray(etichette, dtype=np.int64)]
