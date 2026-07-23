"""Separazione per orientamento coerente (RANSAC sulle normali).

Secondo stadio del confronto con la letteratura (Li/Gan/Wang 2025, vedi
`C - Allineamento.md` §3.2): dentro un cluster che il clustering a crescita
ha fuso (piu' condotte/passerelle con ASSI DIVERSI saldate da contatti o
grovigli), i punti coerenti con una singola direzione dominante vengono
estratti come sotto-istanza, e si ripete sul residuo.

Interpretazione della "unit normal sphere" del paper (testo integrale non
disponibile, vedi §0 punto 3 di `C - Allineamento.md` — se i test sui casi
noti smentiscono, si aggiusta empiricamente qui):

  - le normali di una superficie CILINDRICA giacciono sul CERCHIO MASSIMO
    della sfera unitaria ortogonale all'asse del cilindro: |n . asse| ~ 0;
  - RANSAC sull'asse: due normali campionate a caso definiscono un asse
    candidato (il loro prodotto vettore); l'asse col massimo consenso
    e' la direzione dominante, raffinata come autovettore minimo della
    covarianza delle normali inlier;
  - i punti inlier formano il gruppo direzionale; si ripete sul residuo
    finche' resta un blob senza direzione dominante o troppo piccolo.

Dentro ogni gruppo direzionale le componenti spaziali sconnesse tornano
sotto-cluster separati (due tubi paralleli lontani non sono lo stesso
oggetto). Il residuo senza direzione resta un blocco a se': andra' alle
regole come qualunque altro cluster (tipicamente generico/apparecchiatura).

Pre-gate geometrico (il §3.2 chiede di agire sui *candidati condotta/
passerella*): al punto d'innesto le classi non esistono ancora, quindi il
"candidato condotta" si approssima con la geometria — la separazione parte
solo sui cluster SOTTILI (rapporto minima/massima estensione principale
sotto soglia: gusci tubolari e nastri, anche fusi a L/T/croce), mai sui
blob compatti e pieni (apparecchiature, armadi), che verrebbero
frammentati a torto. Discriminante sul MIN/MAX, non sull'elongazione
globale: una croce di due condotte non e' allungata globalmente.

ATTIVA di default (20/07: superata la disciplina "pipeline congelata" del
17/07 — nessun nuovo meccanismo entrava senza validazione preventiva per
posizione), protetta dal pre-gate sopra. Resta comunque da verificare
empiricamente sui casi aperti noti (istanza ex-id 56, top rack) col
confronto per posizione (`scripts/confronta_run.py`): lo strumento di
giudizio non e' cambiato, e' cambiato solo il fatto che non serve piu'
passarci PRIMA di accendere una feature.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("mepseg")


def _asse_dominante(
    normali: np.ndarray, tol_sin: float, rng: np.random.Generator,
    n_iter: int = 120,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """RANSAC: asse col massimo numero di normali sul suo cerchio massimo.

    Ritorna (asse, maschera inlier) oppure (None, None) se nessun campione
    valido (es. normali tutte quasi parallele: superficie piana, l'asse
    del "cilindro" e' indeterminato e non c'e' niente da separare).
    """
    n = len(normali)
    migliore_asse, migliore_conta = None, 0
    coppie = rng.integers(0, n, size=(n_iter, 2))
    for i, j in coppie:
        asse = np.cross(normali[i], normali[j])
        norma = np.linalg.norm(asse)
        if norma < 0.15:  # normali quasi parallele: asse mal condizionato
            continue
        asse /= norma
        conta = int((np.abs(normali @ asse) <= tol_sin).sum())
        if conta > migliore_conta:
            migliore_asse, migliore_conta = asse, conta
    if migliore_asse is None:
        return None, None
    # raffinamento: l'asse "vero" e' la direzione di minima varianza delle
    # normali inlier (autovettore minimo di M = media(n n^T), insensibile
    # al segno delle normali, che e' ambiguo)
    inlier = np.abs(normali @ migliore_asse) <= tol_sin
    m = normali[inlier]
    _, autovett = np.linalg.eigh(m.T @ m / len(m))
    asse = autovett[:, 0]
    inlier = np.abs(normali @ asse) <= tol_sin
    return asse, inlier


def _componenti_spaziali(
    punti: np.ndarray, eps: float, min_punti: int
) -> list[np.ndarray]:
    """Componenti connesse per distanza (raggio eps) dentro un gruppo
    direzionale: due tubi PARALLELI ma lontani condividono la direzione,
    non l'oggetto. Ritorna liste di indici locali."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(punti)
    if n < 2:
        return [np.arange(n)]
    albero = cKDTree(punti)
    coppie = albero.query_pairs(eps, output_type="ndarray")
    g = sparse.coo_matrix(
        (np.ones(len(coppie), dtype=np.uint8), (coppie[:, 0], coppie[:, 1])),
        shape=(n, n),
    )
    n_comp, comp = connected_components(g, directed=False)
    if n_comp <= 1:
        return [np.arange(n)]
    gruppi = [np.where(comp == c)[0] for c in range(n_comp)]
    grandi = [g_ for g_ in gruppi if len(g_) >= min_punti]
    piccoli = [g_ for g_ in gruppi if len(g_) < min_punti]
    if not grandi:
        return [np.arange(n)]
    if piccoli:
        # i frammenti confluiscono nella componente maggiore: non meritano
        # un'istanza propria e non vanno persi
        maggiore = int(np.argmax([len(g_) for g_ in grandi]))
        grandi[maggiore] = np.concatenate([grandi[maggiore]] + piccoli)
    return grandi


def _e_candidato_lineare(punti: np.ndarray, cfg: dict) -> bool:
    """Pre-gate geometrico: True se il cluster e' una struttura SOTTILE
    (condotta, tubo, passerella — anche fusa a L/T/croce), False se e' un
    blob compatto e pieno (apparecchiatura, armadio).

    Metrica: rapporto tra la minima e la massima estensione lungo gli assi
    principali (PCA). Un guscio tubolare o un nastro ha lo spessore molto
    minore della lunghezza -> rapporto piccolo; un blob 3D pieno ce l'ha
    vicino a 1. Il rapporto min/max (non max-su-secondo) e' cio' che lascia
    passare la croce di due condotte: e' spessa quanto un tubo ma lunga
    quanto due, quindi resta sotto soglia pur non essendo allungata.
    """
    if not cfg.get("pregate_orientamento", True):
        return True
    n = len(punti)
    if n < 3:
        return False
    centrati = punti - punti.mean(axis=0)
    cov = centrati.T @ centrati / n
    _, autovett = np.linalg.eigh(cov)
    proj = centrati @ autovett
    estensioni = proj.max(axis=0) - proj.min(axis=0)
    lungo = float(estensioni.max())
    if lungo <= 1e-9:
        return False
    sfericita = float(estensioni.min()) / lungo
    return sfericita <= float(cfg.get("sfericita_max_orientamento", 0.35))


def separa_per_orientamento(
    idx_cluster: np.ndarray, punti_r: np.ndarray, normali_r: np.ndarray,
    cfg: dict,
) -> list[np.ndarray]:
    """Spezza un cluster in sotto-istanze a orientamento coerente.

    Ritorna una lista di array di indici (>= 1): se non emergono almeno
    due direzioni dominanti DIVERSE, il cluster torna intero — la
    funzione non deve mai frammentare un oggetto legittimo con una sola
    direzione (ne' una superficie piana, dove l'asse e' indeterminato).
    """
    tol_gradi = float(cfg.get("tolleranza_angolo_gradi", 10.0))
    min_punti = max(int(cfg.get("min_punti_direzione", 150)), 30)
    min_fraz = float(cfg.get("min_frazione_direzione", 0.15))
    max_direzioni = int(cfg.get("max_direzioni", 4))
    eps = float(cfg.get("eps", 0.08))

    if len(idx_cluster) < 2 * min_punti:
        return [idx_cluster]
    # pre-gate geometrico (§3.2: agire sui candidati condotta/passerella):
    # i blob compatti e pieni non si separano per orientamento
    if not _e_candidato_lineare(punti_r[idx_cluster], cfg):
        return [idx_cluster]

    normali = normali_r[idx_cluster]
    tol_sin = float(np.sin(np.radians(tol_gradi)))
    # stesso principio del seed di Open3D nella pipeline: risultati
    # riproducibili tra un run e l'altro
    rng = np.random.default_rng(0)

    resto = np.arange(len(idx_cluster))
    gruppi: list[np.ndarray] = []
    assi: list[np.ndarray] = []
    while len(resto) >= min_punti and len(gruppi) < max_direzioni:
        asse, inlier = _asse_dominante(normali[resto], tol_sin, rng)
        if asse is None:
            break
        n_inlier = int(inlier.sum())
        if n_inlier < max(min_punti, int(min_fraz * len(idx_cluster))):
            break
        # un asse quasi parallelo a uno gia' estratto e' la STESSA
        # direzione (code della tolleranza): si fonde, non si duplica
        gemello = next(
            (k for k, a in enumerate(assi)
             if abs(float(a @ asse)) >= np.cos(np.radians(2 * tol_gradi))),
            None,
        )
        if gemello is not None:
            gruppi[gemello] = np.concatenate([gruppi[gemello], resto[inlier]])
        else:
            gruppi.append(resto[inlier])
            assi.append(asse)
        resto = resto[~inlier]

    if len(gruppi) < 2:
        return [idx_cluster]  # una sola direzione: niente da separare

    # il residuo senza direzione dominante resta un blocco a se' (blob
    # irregolare: valvole, grovigli) — alle regole come gli altri
    if len(resto) >= min_punti:
        gruppi.append(resto)
    elif len(resto):
        maggiore = int(np.argmax([len(g_) for g_ in gruppi]))
        gruppi[maggiore] = np.concatenate([gruppi[maggiore], resto])

    # dentro ogni gruppo direzionale: componenti spaziali separate
    esito: list[np.ndarray] = []
    for gruppo in gruppi:
        for comp in _componenti_spaziali(punti_r[idx_cluster[gruppo]], eps,
                                         min_punti):
            esito.append(idx_cluster[gruppo[comp]])
    if len(esito) > 1:
        log.info(
            "Orientamento: cluster da %s punti separato in %d sotto-istanze "
            "(%d direzioni dominanti, tolleranza %.0f gradi)",
            f"{len(idx_cluster):,}", len(esito), len(assi), tol_gradi,
        )
    return esito
