"""Clustering per continuita' di superficie (edge detection 3D).

Alternativa a DBSCAN per la fase 3 della pipeline. DBSCAN usa la sola
distanza euclidea: due oggetti che si toccano (rack contro muro, telaio di
finestra nel muro) diventano un cluster unico. Qui i collegamenti tra punti
vicini vengono filtrati dalle discontinuita' geometriche, l'equivalente dei
bordi in un'immagine:

  - bordo di SALTO: il passo tra i due punti ha una componente rilevante
    lungo la normale (superfici parallele affiancate, non la stessa
    superficie) -> il collegamento si taglia sempre;
  - bordo di PIEGA: le normali dei due punti divergono oltre soglia
    (spigolo). Uno spigolo pero' non separa necessariamente due oggetti:
    le facce di una scatola si incontrano in spigoli CONVESSI e devono
    restare insieme; due oggetti distinti si incontrano in giunzioni
    CONCAVE (il rack "rientra" rispetto al muro). Criterio alla LCCP:
    convesso -> fondi, concavo -> taglia.

L'orientamento delle normali (ambiguo dopo la stima) si risolve con il
test dello spazio libero: la normale punta verso il lato piu' vuoto,
cioe' quello da cui la superficie e' stata vista dallo scanner.

Le patch lisce piccole (anelli di condotte flessibili, frammenti) si
riattaccano alla patch adiacente con piu' contatto, per non frammentare
tubi sottili e gusci ondulati.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

log = logging.getLogger("mepseg")


def _normali_locali(punti: np.ndarray, cfg: dict) -> np.ndarray:
    """Normali a raggio stretto, dedicate al rilevamento dei bordi.

    Le normali del preprocess sono stimate su un raggio largo (4x voxel):
    ottime per le regole, ma "spalmano" gli spigoli su una fascia ampia e
    la piega tra due facce non supera mai la soglia d'angolo. Qui serve
    il contrario: raggio minimo, spigoli ripidi.
    """
    import open3d as o3d

    voxel = float(cfg.get("voxel_eff", 0.03))
    raggio = max(2.0 * voxel, 0.04)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(punti)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=raggio, max_nn=20)
    )
    return np.asarray(pcd.normals)


# famiglie geometriche delle patch (fase A dell'architettura per-superficie)
FAM_NESSUNA, FAM_PIANA_ORIZZ, FAM_PIANA_VERT, FAM_CILINDRICA, FAM_IRREGOLARE = (
    0, 1, 2, 3, 4,
)


def _famiglie_patch(
    normali: np.ndarray, patch: np.ndarray, n_patch: int, cfg: dict
) -> np.ndarray:
    """Famiglia geometrica di ogni patch dalla covarianza delle normali.

    La matrice M = media(n nᵀ) (insensibile al segno delle normali, che e'
    ambiguo) ha una firma diversa per ogni famiglia:
      - superficie PIANA: normali concentrate -> lambda1 ~ 1, resto ~ 0;
      - superficie CILINDRICA: normali distribuite sul cerchio ortogonale
        all'asse -> lambda3 ~ 0 ma lambda2 significativo;
      - IRREGOLARE (fasci di cavi, gomitoli): normali in tutte le
        direzioni -> anche lambda3 alto.
    Le piane si dividono in orizzontali/verticali dalla direzione dominante.
    """
    soglia_cil = float(cfg.get("soglia_cilindrica", 0.10))
    soglia_irr = float(cfg.get("soglia_irregolare", 0.05))

    # sei componenti uniche di n n^T accumulate per patch (group-by)
    conte = np.bincount(patch, minlength=n_patch).astype(np.float64)
    m = np.zeros((n_patch, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            s = np.bincount(
                patch, weights=normali[:, i] * normali[:, j], minlength=n_patch
            )
            m[:, i, j] = s
            m[:, j, i] = s
    m /= np.maximum(conte, 1.0)[:, None, None]

    autoval, autovett = np.linalg.eigh(m)  # crescenti
    l3, l2 = autoval[:, 0], autoval[:, 1]
    e1 = autovett[:, :, 2]  # direzione dominante delle normali

    famiglie = np.full(n_patch, FAM_NESSUNA, dtype=np.uint8)
    irregolare = l3 > soglia_irr
    cilindrica = ~irregolare & (l2 >= soglia_cil)
    piana = ~irregolare & ~cilindrica
    famiglie[irregolare] = FAM_IRREGOLARE
    famiglie[cilindrica] = FAM_CILINDRICA
    famiglie[piana & (np.abs(e1[:, 2]) >= 0.7)] = FAM_PIANA_ORIZZ
    famiglie[piana & (np.abs(e1[:, 2]) < 0.7)] = FAM_PIANA_VERT
    famiglie[conte < 5] = FAM_NESSUNA  # troppo piccole per giudicare
    return famiglie


def clusterizza_crescita(
    punti: np.ndarray, normali: np.ndarray, cfg: dict,
    info: dict | None = None, colori: np.ndarray | None = None,
) -> np.ndarray:
    """Ritorna etichette di cluster (-1 = rumore), come cluster_dbscan.

    Se ``info`` e' un dizionario, vi deposita "famiglie": la famiglia
    geometrica per punto (per l'export di debug e le fasi successive).

    Se ``colori`` (N, 3, uint8) e' fornito e ``taglio_colore`` e' attivo, una
    forte discontinuita' di colore tra due punti vicini severa il loro
    collegamento come un bordo di salto: due superfici geometricamente
    continue ma di colore diverso (rack grigio / passerella gialla /
    blindosbarra blu) diventano oggetti diversi. E' il segnale che la sola
    geometria non ha in una sala densa dello stesso materiale, dove il
    clustering fonde apparecchiature distinte.
    """
    n = len(punti)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    eps = float(cfg.get("eps", 0.08))
    k = int(cfg.get("vicini_grafo", 8))
    min_punti = max(int(cfg.get("min_punti", 10)), 1)
    angolo = float(cfg.get("angolo_liscio", 18.0))
    salto_max = float(cfg.get("salto_max", 0.04))
    patch_minima = int(cfg.get("patch_minima", 30))
    soglia_concava = float(cfg.get("soglia_concava", 0.15))
    min_fraz_convessa = float(cfg.get("min_frazione_convessa", 0.5))
    offset = float(cfg.get("offset_orientamento", 0.06))
    cos_liscio = np.cos(np.radians(angolo))
    # taglio per discontinuita' di colore (opzionale: richiede RGB)
    taglio_colore = colori is not None and bool(cfg.get("taglio_colore", True))
    soglia_colore = float(cfg.get("soglia_colore", 60.0))
    colori_f = colori.astype(np.float32) if taglio_colore else None

    # due giochi di normali con ruoli diversi: quelle LISCE del preprocess
    # (raggio largo) giudicano le famiglie — sulla fisarmonica delle
    # condotte flessibili la corrugazione si media e resta la firma
    # cilindrica; quelle AFFILATE (raggio stretto) rilevano i bordi, dove
    # serve il contrario (spigoli ripidi, niente media)
    normali_lisce = normali
    if cfg.get("normali_locali", True):
        normali = _normali_locali(punti, cfg)

    albero = cKDTree(punti)

    # --- 1. grafo k-NN con classificazione dei collegamenti ----------------
    blocchi_i, blocchi_j, blocchi_liscio = [], [], []
    blocco = 1_000_000
    kk = min(k + 1, n)
    for inizio in range(0, n, blocco):
        fine = min(inizio + blocco, n)
        dist, vic = albero.query(
            punti[inizio:fine], k=kk, distance_upper_bound=eps, workers=-1
        )
        if kk == 1:
            dist, vic = dist[:, None], vic[:, None]
        righe = np.repeat(np.arange(inizio, fine, dtype=np.int64), kk).reshape(
            -1, kk
        )
        valido = (vic < n) & (vic != righe) & np.isfinite(dist)
        i = righe[valido].astype(np.int32)
        j = vic[valido].astype(np.int32)

        passo = punti[j] - punti[i]
        salto = np.maximum(
            np.abs(np.einsum("ij,ij->i", passo, normali[i])),
            np.abs(np.einsum("ij,ij->i", passo, normali[j])),
        )
        tieni = salto <= salto_max  # i bordi di salto si scartano subito
        if colori_f is not None:
            # discontinuita' di colore = confine tra oggetti diversi anche su
            # superficie geometricamente continua: si severa come un salto
            dcol = np.linalg.norm(colori_f[j] - colori_f[i], axis=1)
            tieni &= dcol <= soglia_colore
        i, j = i[tieni], j[tieni]
        cosang = np.abs(np.einsum("ij,ij->i", normali[i], normali[j]))
        blocchi_i.append(i)
        blocchi_j.append(j)
        blocchi_liscio.append(cosang >= cos_liscio)

    arco_i = np.concatenate(blocchi_i)
    arco_j = np.concatenate(blocchi_j)
    liscio = np.concatenate(blocchi_liscio)
    del blocchi_i, blocchi_j, blocchi_liscio

    # --- 2. patch = componenti connesse dei soli collegamenti lisci --------
    g = sparse.coo_matrix(
        (np.ones(int(liscio.sum()), dtype=np.uint8),
         (arco_i[liscio], arco_j[liscio])),
        shape=(n, n),
    )
    n_patch, patch = connected_components(g, directed=False)

    # --- 2bis. famiglia geometrica di ogni patch (fase A) -------------------
    famiglie = _famiglie_patch(normali_lisce, patch, n_patch, cfg)
    if info is not None:
        info["famiglie"] = famiglie[patch]
    voxel = float(cfg.get("voxel_eff", 0.03))
    conteggi_patch = np.bincount(patch, minlength=n_patch)
    area_patch = conteggi_patch * voxel**2

    # --- 3. spigoli tra patch diverse: convessita' --------------------------
    piega = ~liscio & (patch[arco_i] != patch[arco_j])
    pi_, pj_ = arco_i[piega], arco_j[piega]

    merge_i, merge_j = [], []
    if len(pi_):
        # orientamento delle normali dei soli punti di bordo: la normale
        # punta verso il lato con piu' spazio libero (quello "visto")
        bordo = np.unique(np.concatenate([pi_, pj_]))
        p_b, n_b = punti[bordo], normali[bordo]
        d_avanti, _ = albero.query(p_b + offset * n_b, k=1, workers=-1)
        d_dietro, _ = albero.query(p_b - offset * n_b, k=1, workers=-1)
        segno = np.where(d_avanti >= d_dietro, 1.0, -1.0)
        incerto = np.abs(d_avanti - d_dietro) < 0.015
        segno_pieno = np.zeros(n)
        incerto_pieno = np.zeros(n, dtype=bool)
        segno_pieno[bordo] = segno
        incerto_pieno[bordo] = incerto

        d = punti[pi_] - punti[pj_]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ni = normali[pi_] * segno_pieno[pi_, None]
        nj = normali[pj_] * segno_pieno[pj_, None]
        conv = np.einsum("ij,ij->i", ni - nj, d)
        # concavo solo se l'orientamento di entrambe le normali e' affidabile;
        # nel dubbio si fonde (comportamento DBSCAN, ci pensa poi lo
        # spezza-blob della pipeline)
        concavo = (conv < -soglia_concava) & ~incerto_pieno[pi_] & ~incerto_pieno[pj_]

        # voto per coppia di patch: si fondono se la maggioranza dei
        # contatti al confine non e' concava
        pa = patch[pi_].astype(np.int64)
        pb = patch[pj_].astype(np.int64)
        alto, basso = np.maximum(pa, pb), np.minimum(pa, pb)
        chiave = alto * n_patch + basso
        uniche, inversa = np.unique(chiave, return_inverse=True)
        contatti = np.bincount(inversa)
        concavi = np.bincount(inversa, weights=concavo.astype(np.float64))
        fondi = (1.0 - concavi / contatti) >= min_fraz_convessa

        # --- 3bis. compatibilita' semantica (fase B): la convessita' da
        # sola non basta dove l'orientamento e' incerto (zone dense di
        # cavi). Due patch di famiglie "incompatibili" non sono mai lo
        # stesso oggetto, comunque si tocchino:
        #   - un cilindro esteso non si fonde con una lastra grande ne'
        #     con un groviglio (la condotta che sfiora il tappeto di cavi
        #     o la passerella resta condotta);
        #   - un groviglio non si fonde con una lastra grande (il tappeto
        #     di cavi non ingloba pezzi di controsoffitto o ante).
        # Le lastre piccole (staffe, flange, sportelli) restano libere di
        # fondersi con tutto: sono i dettagli degli oggetti stessi.
        area_mista = float(cfg.get("area_max_fusione_mista", 1.5))
        p_a = (uniche // n_patch).astype(np.int64)
        p_b = (uniche % n_patch).astype(np.int64)
        fam_a, fam_b = famiglie[p_a], famiglie[p_b]
        piano_grande_a = (
            ((fam_a == FAM_PIANA_ORIZZ) | (fam_a == FAM_PIANA_VERT))
            & (area_patch[p_a] >= area_mista)
        )
        piano_grande_b = (
            ((fam_b == FAM_PIANA_ORIZZ) | (fam_b == FAM_PIANA_VERT))
            & (area_patch[p_b] >= area_mista)
        )
        cil_a = (fam_a == FAM_CILINDRICA) & (conteggi_patch[p_a] >= patch_minima)
        cil_b = (fam_b == FAM_CILINDRICA) & (conteggi_patch[p_b] >= patch_minima)
        irr_a = (fam_a == FAM_IRREGOLARE) & (area_patch[p_a] >= area_mista)
        irr_b = (fam_b == FAM_IRREGOLARE) & (area_patch[p_b] >= area_mista)
        # contro i cilindri il veto scatta anche per grovigli MEDI: basta
        # un ciuffo di cavi da qualche centinaio di punti per inghiottire
        # una condotta (valvole e staffe piccole restano sotto patch_minima
        # e continuano ad attaccarsi liberamente)
        irr_media_a = (fam_a == FAM_IRREGOLARE) & (conteggi_patch[p_a] >= patch_minima)
        irr_media_b = (fam_b == FAM_IRREGOLARE) & (conteggi_patch[p_b] >= patch_minima)
        incompatibili = (
            (cil_a & (piano_grande_b | irr_media_b))
            | (cil_b & (piano_grande_a | irr_media_a))
            | (irr_a & piano_grande_b)
            | (irr_b & piano_grande_a)
        )
        fondi &= ~incompatibili

        merge_i.append(p_a[fondi])
        merge_j.append(p_b[fondi])

    # --- 4. patch piccole: al vicino con piu' contatto ----------------------
    # ...ma solo se il contatto e' VERO: passi corti (~la risoluzione del
    # voxel), non collegamenti di grafo lunghi fino a eps. Senza questo
    # vincolo lo speckle a mezz'aria si agganciava alle superfici a 5-8 cm
    # e inquinava OBB e densita' delle istanze; le patch piccole legittime
    # (anelli di flessibili, staffe) toccano il resto a distanza di voxel.
    contatto_max = float(cfg.get("contatto_max_patch", 2.5)) * voxel
    dim_patch = np.bincount(patch, minlength=n_patch)
    piccole = dim_patch < patch_minima
    if len(pi_) and piccole.any():
        passo_corto = (
            np.linalg.norm(punti[pi_] - punti[pj_], axis=1) <= contatto_max
        )
        pa = patch[pi_].astype(np.int64)
        pb = patch[pj_].astype(np.int64)
        m = piccole[pa] & passo_corto  # contatti veri da una patch piccola
        if m.any():
            chiave = pa[m] * n_patch + pb[m]
            uniche, inversa = np.unique(chiave, return_inverse=True)
            contatti = np.bincount(inversa)
            src = uniche // n_patch
            dst = uniche % n_patch
            # per ogni patch piccola, il vicino col massimo contatto
            ordine = np.lexsort((contatti, src))
            ultimo = np.append(src[ordine][1:] != src[ordine][:-1], True)
            scelti = ordine[ultimo]
            merge_i.append(src[scelti])
            merge_j.append(dst[scelti])

    # --- 5. oggetti = componenti connesse del grafo delle patch -------------
    if merge_i:
        mi = np.concatenate(merge_i)
        mj = np.concatenate(merge_j)
        g_patch = sparse.coo_matrix(
            (np.ones(len(mi), dtype=np.uint8), (mi, mj)),
            shape=(n_patch, n_patch),
        )
        _, gruppo = connected_components(g_patch, directed=False)
    else:
        gruppo = np.arange(n_patch)

    etichette = gruppo[patch]

    # --- 6. rumore e rinumerazione compatta ---------------------------------
    dimensioni = np.bincount(etichette)
    valide = dimensioni >= min_punti
    log.info(
        "Crescita: %d patch lisce -> %d oggetti (%d sotto min_punti -> rumore)",
        n_patch, int(valide.sum()), int((~valide).sum()),
    )
    mappa = np.full(len(dimensioni), -1, dtype=np.int64)
    mappa[valide] = np.arange(int(valide.sum()))
    return mappa[etichette]
