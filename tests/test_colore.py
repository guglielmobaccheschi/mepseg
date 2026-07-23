"""Taglio per discontinuita' di colore nel clustering a crescita.

Riproduce il guasto che la sola geometria non risolve e ne prova la cura:
una superficie geometricamente CONTINUA ma di due colori diversi (es. un
rack grigio e una passerella gialla che si toccano) viene fusa in un solo
cluster dalla crescita; col taglio per colore i due oggetti si separano.

E' la dimostrazione che l'RGB va nel CLUSTERING (dove nasce la fusione),
non nelle regole di classificazione (che girano dopo, per cluster, e non
possono dis-fondere un blob).

Esecuzione:  python tests/test_colore.py   (oppure pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from mepseg.crescita import clusterizza_crescita  # noqa: E402

CFG = {
    "eps": 0.08,
    "vicini_grafo": 8,
    "min_punti": 10,
    "angolo_liscio": 18.0,
    "salto_max": 0.04,
    "patch_minima": 30,
    "soglia_concava": 0.15,
    "min_frazione_convessa": 0.5,
    "offset_orientamento": 0.06,
    "normali_locali": False,   # normali fornite: test deterministico
    "taglio_colore": True,
    "soglia_colore": 60.0,
}

GRIGIO = (128, 128, 128)
GIALLO = (255, 220, 0)


def _slab_bicolore(passo=0.02, seme=0):
    """Lastra orizzontale continua x in [0,2], y in [0,1], z~0: meta'
    sinistra grigia, meta' destra gialla. Geometricamente un solo piano."""
    rng = np.random.default_rng(seme)
    xs = np.arange(0, 2, passo)
    ys = np.arange(0, 1, passo)
    gx, gy = np.meshgrid(xs, ys)
    punti = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    punti[:, 2] += rng.normal(0, 0.002, len(punti))  # rumore di scansione
    normali = np.tile([0.0, 0.0, 1.0], (len(punti), 1))
    colori = np.where(
        (punti[:, 0] < 1.0)[:, None], np.array(GRIGIO), np.array(GIALLO)
    ).astype(np.uint8)
    return punti, normali, colori


def _n_cluster(etichette):
    return int(etichette.max()) + 1 if etichette.size and etichette.max() >= 0 else 0


def test_senza_colore_la_lastra_si_fonde():
    # la malattia: senza colore la crescita vede un piano continuo -> 1 oggetto
    punti, normali, _ = _slab_bicolore()
    et = clusterizza_crescita(punti, normali, CFG, colori=None)
    assert _n_cluster(et) == 1, f"attesa fusione in 1 cluster, trovati {_n_cluster(et)}"


def test_con_colore_la_lastra_si_separa():
    # la cura: la discontinuita' di colore separa i due oggetti
    punti, normali, colori = _slab_bicolore()
    et = clusterizza_crescita(punti, normali, CFG, colori=colori)
    assert _n_cluster(et) == 2, f"atteso 2 cluster col taglio colore, trovati {_n_cluster(et)}"
    # ogni cluster deve essere di un solo colore (separazione pulita)
    for c in range(2):
        col_cluster = colori[et == c]
        primo = col_cluster[0]
        assert np.all(col_cluster == primo), "cluster con colori misti: taglio impreciso"


def test_colore_uniforme_non_frammenta():
    # controllo di non-regressione: una lastra di UN solo colore resta 1
    # cluster anche col taglio attivo (nessuna frammentazione spuria)
    punti, normali, colori = _slab_bicolore()
    colori[:] = np.array(GRIGIO, dtype=np.uint8)
    et = clusterizza_crescita(punti, normali, CFG, colori=colori)
    assert _n_cluster(et) == 1, "una lastra monocolore non va frammentata"


def test_preprocess_porta_il_colore():
    # plumbing: sottocampiona_e_prepara deve restituire il colore ridotto
    # allineato ai punti (Open3D lo media per voxel come le coordinate)
    from mepseg.preprocess import sottocampiona_e_prepara

    punti, _, colori = _slab_bicolore(passo=0.02)
    cfg = {"voxel": 0.03, "rimuovi_outlier": False}
    p_rid, n_rid, c_rid = sottocampiona_e_prepara(punti, cfg, colori)
    assert c_rid is not None, "il colore non e' stato propagato"
    assert len(c_rid) == len(p_rid), "colore non allineato ai punti ridotti"
    # entrambe le regioni di colore devono sopravvivere al voxel
    assert (c_rid[:, 0] > 200).any(), "regione gialla persa"
    assert (np.abs(c_rid[:, 0].astype(int) - 128) < 40).any(), "regione grigia persa"
    # senza colore in ingresso: nessun colore in uscita (nessuna regressione)
    p2, n2, c2 = sottocampiona_e_prepara(punti, cfg, None)
    assert c2 is None, "colore inventato da input senza colore"


def test_accumulatore_media_colore():
    # la parte delicata dello streaming: la media colore per voxel deve
    # restare corretta anche accumulando a blocchi (group-by incrementale)
    from mepseg.streaming import AccumulatoreVoxel

    acc = AccumulatoreVoxel(0.1)
    # voxel A (x~0): due punti rossi diversi; voxel B (x~0.5): verde puro
    acc.aggiungi(
        np.array([[0.01, 0, 0], [0.02, 0, 0], [0.5, 0, 0]]),
        np.array([[200, 0, 0], [100, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    # secondo blocco: un altro punto nel voxel A, colore nero
    acc.aggiungi(np.array([[0.03, 0, 0]]), np.array([[0, 0, 0]], dtype=np.uint8))
    punti = acc.risultato()
    colori = acc.risultato_colore()
    assert colori is not None and len(colori) == len(punti) == 2
    ia, ib = int(np.argmin(punti[:, 0])), int(np.argmax(punti[:, 0]))
    # voxel A: media di 200, 100, 0 sul rosso = 100
    assert abs(int(colori[ia, 0]) - 100) <= 1, colori[ia]
    assert colori[ia, 1] == 0 and colori[ia, 2] == 0
    # voxel B: verde puro, un solo punto
    assert tuple(int(v) for v in colori[ib]) == (0, 255, 0), colori[ib]


def test_accumulatore_senza_colore():
    from mepseg.streaming import AccumulatoreVoxel

    acc = AccumulatoreVoxel(0.1)
    acc.aggiungi(np.array([[0.01, 0, 0], [0.5, 0, 0]]))
    assert acc.risultato_colore() is None, "colore inventato senza input colore"


def _blob(colore, n=400, seme=0):
    from mepseg.primitive import calcola_feature

    rng = np.random.default_rng(seme)
    p = rng.uniform(0, 0.3, (n, 3))
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    col = np.tile(colore, (n, 1)).astype(np.uint8)
    return calcola_feature(p, nrm, 3.0, 0.0, {"voxel_eff": 0.03}, colori=col)


def test_feature_colore_calcolata():
    feat = _blob([200, 10, 10])
    assert feat.colore_dominante is not None
    assert abs(int(feat.colore_dominante[0]) - 200) <= 6, feat.colore_dominante
    assert feat.uniformita_colore is not None and feat.uniformita_colore > 0.95
    # senza colore: nessuna misura inventata
    from mepseg.primitive import calcola_feature

    p = np.random.default_rng(0).uniform(0, 0.3, (400, 3))
    nrm = np.tile([0.0, 0.0, 1.0], (400, 1))
    f2 = calcola_feature(p, nrm, 3.0, 0.0, {"voxel_eff": 0.03})
    assert f2.colore_dominante is None and f2.uniformita_colore is None


def test_guida_colore_spareggio():
    from mepseg.classi import ClasseMEP
    from mepseg.pipeline import Istanza
    from mepseg.regole import applica_guida_colore

    guida = [{"classe": "apparecchiatura", "rgb": [200, 0, 0]}]

    # generico rosso -> convertito ad apparecchiatura
    ist = Istanza(1, ClasseMEP.MEP_GENERICO, _blob([200, 10, 10]), np.arange(400))
    et = np.zeros(400, dtype=np.uint8)
    assert applica_guida_colore([ist], et, guida, {}) == 1
    assert ist.classe == ClasseMEP.APPARECCHIATURA
    assert (et == int(ClasseMEP.APPARECCHIATURA)).all()

    # generico verde -> NON combacia, resta generico
    ist_v = Istanza(2, ClasseMEP.MEP_GENERICO, _blob([10, 200, 10], seme=1), np.arange(400))
    assert applica_guida_colore([ist_v], np.zeros(400, np.uint8), guida, {}) == 0
    assert ist_v.classe == ClasseMEP.MEP_GENERICO

    # non-generico rosso -> il colore NON sovrascrive la geometria
    ist_c = Istanza(3, ClasseMEP.CONDOTTA_CIRCOLARE, _blob([200, 10, 10], seme=2), np.arange(400))
    assert applica_guida_colore([ist_c], np.zeros(400, np.uint8), guida, {}) == 0
    assert ist_c.classe == ClasseMEP.CONDOTTA_CIRCOLARE

    # guida vuota -> no-op (tool agnostico di default)
    ist2 = Istanza(4, ClasseMEP.MEP_GENERICO, _blob([200, 10, 10], seme=3), np.arange(400))
    assert applica_guida_colore([ist2], np.zeros(400, np.uint8), [], {}) == 0
    assert ist2.classe == ClasseMEP.MEP_GENERICO


if __name__ == "__main__":
    test_senza_colore_la_lastra_si_fonde()
    test_con_colore_la_lastra_si_separa()
    test_colore_uniforme_non_frammenta()
    test_preprocess_porta_il_colore()
    test_accumulatore_media_colore()
    test_accumulatore_senza_colore()
    test_feature_colore_calcolata()
    test_guida_colore_spareggio()
    print("test_colore: tutti i test superati")
