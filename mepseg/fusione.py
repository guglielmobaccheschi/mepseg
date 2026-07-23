"""Fusione delle etichette geometriche con le predizioni deep learning.

Strategia conservativa: la pipeline geometrica ha precedenza dove ha
riconosciuto una primitiva con confidenza (tubi, condotte, piani strutturali);
la rete neurale interviene solo dove la geometria ha risposto
"MEP generico", cioe' dove le regole non hanno trovato una forma nota.
"""
from __future__ import annotations

import numpy as np

from .classi import ClasseMEP


def fondi_etichette(
    etichette_geom: np.ndarray, etichette_dl: np.ndarray
) -> np.ndarray:
    if etichette_geom.shape != etichette_dl.shape:
        raise ValueError("Le due mappe di etichette devono avere la stessa forma")
    risultato = etichette_geom.copy()
    generici = etichette_geom == int(ClasseMEP.MEP_GENERICO)
    risultato[generici] = etichette_dl[generici]
    return risultato
