"""Test end-to-end sulla scena sintetica: verifica che ogni classe attesa
venga riconosciuta e che le stime dei diametri siano corrette.

Esecuzione:  python tests/test_pipeline.py   (oppure pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))
sys.path.insert(0, str(RADICE / "esempi"))


def test_scena_demo():
    from genera_scena_demo import genera
    from mepseg.classi import ClasseMEP
    from mepseg.io_nuvole import Nuvola
    from mepseg.pipeline import segmenta

    with open(RADICE / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    punti = genera()
    risultato = segmenta(Nuvola(punti), cfg)
    etichette = risultato.etichette
    totale = len(etichette)

    def frazione(classe: ClasseMEP) -> float:
        return float((etichette == int(classe)).mean())

    # la struttura domina la scena, distinta in pavimento/soffitto/pareti
    frazione_struttura = (
        frazione(ClasseMEP.PAVIMENTO)
        + frazione(ClasseMEP.SOFFITTO)
        + frazione(ClasseMEP.PARETE)
        + frazione(ClasseMEP.STRUTTURA)
    )
    assert frazione_struttura > 0.45, "struttura non riconosciuta"
    assert frazione(ClasseMEP.PAVIMENTO) > 0.10, "pavimento non riconosciuto"
    assert frazione(ClasseMEP.SOFFITTO) > 0.10, "soffitto non riconosciuto"
    assert frazione(ClasseMEP.PARETE) > 0.05, "pareti non riconosciute"
    # i due pilastri (tondo e rettangolare) devono avere la loro classe
    # (rilevati a livello strutturale, quindi come punti, non istanze)
    assert frazione(ClasseMEP.PILASTRO) > 0.01, "pilastri non riconosciuti"
    # ogni classe MEP attesa deve essere presente in quantita' plausibile
    assert frazione(ClasseMEP.CONDOTTA_CIRCOLARE) > 0.05
    assert frazione(ClasseMEP.CONDOTTA_RETTANGOLARE) > 0.05
    assert frazione(ClasseMEP.TUBAZIONE) > 0.03
    assert frazione(ClasseMEP.PASSERELLA_CAVI) > 0.01
    assert frazione(ClasseMEP.LUCE) > 0.005
    assert frazione(ClasseMEP.TERMINALE_ARIA) > 0.003
    assert frazione(ClasseMEP.APPARECCHIATURA) > 0.01
    assert (etichette == int(ClasseMEP.SPRINKLER)).sum() >= 100, "sprinkler persi"
    # quasi nulla deve restare non classificato
    assert frazione(ClasseMEP.MEP_GENERICO) < 0.02

    # regola di ancoraggio: l'armadietto aperto (mezzo cilindro verticale
    # appoggiato al pavimento) deve essere declassato ad arredo, non condotta
    arredi = [i for i in risultato.istanze if i.classe == ClasseMEP.NON_MEP_ARREDO]
    assert len(arredi) == 1, f"attesa 1 istanza arredo (armadietto), trovate {len(arredi)}"
    centro = arredi[0].feature.centro
    assert abs(centro[0] - 9.5) < 0.3 and abs(centro[1] - 1.4) < 0.4, (
        "l'istanza arredo non e' l'armadietto"
    )
    # ...ma tubi e condotte veri, ancorati alle pareti, non vanno declassati

    # stima diametri: la condotta circolare della scena ha raggio 0.30 m,
    # le tubazioni 0.04-0.06 m
    raggi_condotte = [
        i.feature.cilindro.raggio
        for i in risultato.istanze
        if i.classe == ClasseMEP.CONDOTTA_CIRCOLARE
    ]
    assert raggi_condotte and abs(raggi_condotte[0] - 0.30) < 0.02
    raggi_tubi = sorted(
        i.feature.cilindro.raggio
        for i in risultato.istanze
        if i.classe == ClasseMEP.TUBAZIONE
    )
    assert len(raggi_tubi) == 4  # 3 tubazioni + tubo sprinkler
    assert all(0.03 < r < 0.07 for r in raggi_tubi)

    # "trova simili": un modello estratto da una luce deve riconoscere la
    # luce stessa e NON una condotta circolare
    from mepseg.somiglianza import corrisponde, crea_modello

    luci = [i for i in risultato.istanze if i.classe == ClasseMEP.LUCE]
    condotte = [
        i for i in risultato.istanze if i.classe == ClasseMEP.CONDOTTA_CIRCOLARE
    ]
    assert luci and condotte
    modello = crea_modello(luci[0], "luce_test")
    assert modello["classe"] == "luce"
    tolleranze = {}
    assert corrisponde(modello, luci[0].feature, tolleranze), (
        "il modello non riconosce la propria istanza"
    )
    assert not corrisponde(modello, condotte[0].feature, tolleranze), (
        "il modello luce non deve combaciare con una condotta"
    )

    print("test_scena_demo superato:")
    for classe in ClasseMEP:
        n = int((etichette == int(classe)).sum())
        if n:
            print(f"  {classe.name:<22} {n:>9,} punti ({100.0 * n / totale:5.1f}%)")


if __name__ == "__main__":
    test_scena_demo()
