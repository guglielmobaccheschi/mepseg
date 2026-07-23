"""Test unitari della separazione per orientamento (mepseg/orientamento.py).

Casi sintetici mirati, senza pipeline completa:
  - due cilindri con assi ortogonali fusi in un cluster -> 2 sotto-istanze;
  - un cilindro solo -> resta intero (una direzione = niente da separare);
  - una lastra piana -> resta intera (asse indeterminato, nessuna
    frammentazione artificiosa);
  - due tubi PARALLELI ma lontani -> separati dalle componenti spaziali.

Esecuzione:  python tests/test_orientamento.py   (oppure pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from mepseg.orientamento import separa_per_orientamento  # noqa: E402

CFG = {
    "eps": 0.08,
    "tolleranza_angolo_gradi": 10.0,
    "min_punti_direzione": 150,
    "min_frazione_direzione": 0.15,
    "max_direzioni": 4,
    "pregate_orientamento": True,
    "sfericita_max_orientamento": 0.35,
}


def _cilindro(
    centro, asse, raggio=0.15, lunghezza=3.0, n=2000, seme=0
) -> tuple[np.ndarray, np.ndarray]:
    """Punti e normali di un guscio cilindrico lungo ``asse``."""
    rng = np.random.default_rng(seme)
    asse = np.asarray(asse, dtype=float)
    asse /= np.linalg.norm(asse)
    # base ortonormale attorno all'asse
    ausiliario = np.array([0.0, 0.0, 1.0])
    if abs(asse @ ausiliario) > 0.9:
        ausiliario = np.array([1.0, 0.0, 0.0])
    u = np.cross(asse, ausiliario)
    u /= np.linalg.norm(u)
    v = np.cross(asse, u)
    t = rng.uniform(-lunghezza / 2, lunghezza / 2, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    normali = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)
    punti = np.asarray(centro) + np.outer(t, asse) + raggio * normali
    punti += rng.normal(0, 0.003, punti.shape)  # rumore di scansione
    return punti, normali


def _lastra(centro, lato=2.0, n=2000, seme=0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seme)
    xy = rng.uniform(-lato / 2, lato / 2, (n, 2))
    punti = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(n)]) + np.asarray(centro)
    punti[:, 2] += rng.normal(0, 0.003, n)
    normali = np.tile([0.0, 0.0, 1.0], (n, 1))
    normali += rng.normal(0, 0.02, normali.shape)
    normali /= np.linalg.norm(normali, axis=1, keepdims=True)
    return punti, normali


def test_due_cilindri_ortogonali_si_separano():
    # una "croce": condotta lungo X e condotta lungo Y che si incrociano
    p1, n1 = _cilindro([0, 0, 0], [1, 0, 0], seme=1)
    p2, n2 = _cilindro([0, 0, 0.35], [0, 1, 0], seme=2)
    punti = np.vstack([p1, p2])
    normali = np.vstack([n1, n2])
    idx = np.arange(len(punti))

    gruppi = separa_per_orientamento(idx, punti, normali, CFG)
    assert len(gruppi) >= 2, "cilindri con assi diversi non separati"
    # ogni punto in un solo gruppo, nessun punto perso
    tutti = np.concatenate(gruppi)
    assert len(tutti) == len(idx)
    assert len(np.unique(tutti)) == len(idx)
    # i due gruppi maggiori devono essere ~puri (>=90% dallo stesso tubo)
    per_dimensione = sorted(gruppi, key=len, reverse=True)[:2]
    for g in per_dimensione:
        fraz_primo = float((g < len(p1)).mean())
        assert fraz_primo >= 0.9 or fraz_primo <= 0.1, (
            f"gruppo misto: {fraz_primo:.2f} dal primo cilindro"
        )


def test_cilindro_singolo_resta_intero():
    p, n = _cilindro([0, 0, 0], [1, 0, 0], seme=3)
    idx = np.arange(len(p))
    gruppi = separa_per_orientamento(idx, p, n, CFG)
    assert len(gruppi) == 1, "un solo cilindro non va frammentato"


def test_lastra_piana_resta_intera():
    p, n = _lastra([0, 0, 0], seme=4)
    idx = np.arange(len(p))
    gruppi = separa_per_orientamento(idx, p, n, CFG)
    assert len(gruppi) == 1, "una lastra piana non va frammentata"


def test_tubi_paralleli_lontani_separati_spazialmente():
    # stessa direzione, ma a 1 m di distanza: oggetti diversi
    p1, n1 = _cilindro([0, 0, 0], [1, 0, 0], seme=5)
    p2, n2 = _cilindro([0, 1.0, 0], [1, 0, 0], seme=6)
    # una terza direzione per far scattare la separazione (con una sola
    # direzione dominante il cluster resta intero per contratto)
    p3, n3 = _cilindro([0, 0.5, 0.6], [0, 1, 0], seme=7)
    punti = np.vstack([p1, p2, p3])
    normali = np.vstack([n1, n2, n3])
    idx = np.arange(len(punti))
    gruppi = separa_per_orientamento(idx, punti, normali, CFG)
    assert len(gruppi) >= 3, (
        "i due tubi paralleli lontani devono restare oggetti distinti"
    )


def _croce_compatta():
    # due cilindri CORTI e grossi incrociati: hanno due assi veri (come la
    # croce lunga), ma l'ingombro e' quasi cubico (spessore ~ lunghezza) ->
    # e' un blob compatto, non una condotta. Senza pre-gate si spezzerebbe.
    p1, n1 = _cilindro([0, 0, 0], [1, 0, 0], raggio=0.2, lunghezza=0.6,
                       n=1200, seme=10)
    p2, n2 = _cilindro([0, 0, 0.3], [0, 1, 0], raggio=0.2, lunghezza=0.6,
                       n=1200, seme=11)
    return np.vstack([p1, p2]), np.vstack([n1, n2])


def test_blob_compatto_non_si_spezza_col_pregate():
    punti, normali = _croce_compatta()
    idx = np.arange(len(punti))
    gruppi = separa_per_orientamento(idx, punti, normali, CFG)
    assert len(gruppi) == 1, "il pre-gate deve lasciare intero un blob compatto"


def test_pregate_disattivabile():
    # lo stesso blob compatto, col pre-gate spento, si separa: dimostra che
    # e' il gate — non l'auto-limitazione della RANSAC — a proteggerlo
    punti, normali = _croce_compatta()
    idx = np.arange(len(punti))
    cfg_off = dict(CFG, pregate_orientamento=False)
    gruppi = separa_per_orientamento(idx, punti, normali, cfg_off)
    assert len(gruppi) >= 2, "senza pre-gate la croce compatta si separa"


if __name__ == "__main__":
    test_due_cilindri_ortogonali_si_separano()
    test_cilindro_singolo_resta_intero()
    test_lastra_piana_resta_intera()
    test_tubi_paralleli_lontani_separati_spazialmente()
    test_blob_compatto_non_si_spezza_col_pregate()
    test_pregate_disattivabile()
    print("test_orientamento: tutti i test superati")
