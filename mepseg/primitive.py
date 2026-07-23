"""Fitting di primitive geometriche e calcolo delle feature per cluster.

Le feature calcolate qui alimentano il motore a regole (:mod:`mepseg.regole`):
descrittori di forma da PCA, bounding box orientata e fitting di cilindro
(asse stimato dalle normali, cerchio ai minimi quadrati sulla sezione).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitCilindro:
    valido: bool = False
    raggio: float = 0.0
    inlier_ratio: float = 0.0
    coerenza_radiale: float = 0.0  # frazione di normali dirette radialmente
    sezione_su_raggio: float = 0.0  # estensione min della sezione / raggio
    concentrazione_normali: float = 1.0  # frazione nei 2 bin angolari dominanti
    asse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    centro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lunghezza: float = 0.0
    residui: np.ndarray | None = None  # distanza radiale assoluta per punto


@dataclass
class FeatureCluster:
    n_punti: int
    centro: np.ndarray
    dimensioni: np.ndarray        # lati OBB ordinati decrescenti (3,)
    linearita: float              # (l1-l2)/l1
    planarita: float              # (l2-l3)/l1
    dispersione: float            # l3/l1
    verticalita_asse: float       # |z| della direzione principale
    quota_top: float              # z massima del cluster
    quota_bottom: float           # z minima del cluster
    dist_da_soffitto: float       # quota_soffitto - quota_top
    dist_da_pavimento: float      # quota_bottom - quota_pavimento
    cilindro: FitCilindro
    sotto_tubazione: bool = False  # True per sotto-cluster staccati da un tubo
    estensione_z_robusta: float = 0.0  # z al 95° - z al 5° percentile: spessore
                                       # verticale del grosso del cluster, cieco
                                       # ad appendici sparse (pendini, cavi)
    densita_superficie: float = 1.0  # punti effettivi / punti attesi se il
                                     # guscio dell'OBB fosse campionato pieno
                                     # alla risoluzione del voxel: un oggetto
                                     # reale scansionato sta sopra ~0.1, il
                                     # rumore sparso molto sotto
    frazione_z_alta: float = 1.0     # massima frazione di punti in una fascia
                                     # di 15 cm agli estremi (alto O basso):
                                     # una lampada ha il corpo concentrato in
                                     # una fascia (in alto se a plafone, in
                                     # basso se sospesa coi tiranti sopra);
                                     # lastre inclinate e nuvole di riflesso
                                     # hanno massa distribuita
    # colore (solo se la nuvola ha RGB): misure sempre calcolate, mai un
    # colore->classe cablato. Il colore dominante e' la mediana per canale
    # (robusta a pochi punti fuori tinta); l'uniformita' e' la frazione di
    # punti vicini al dominante — alta su una superficie di colore coerente
    # (oggetto reale), bassa su un raggruppamento cromaticamente misto
    colore_dominante: np.ndarray | None = None  # RGB (3,) uint8, o None
    uniformita_colore: float | None = None      # 0-1, o None senza RGB


def _pca(punti: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (autovalori decrescenti, autovettori per colonna)."""
    centrati = punti - punti.mean(axis=0)
    cov = centrati.T @ centrati / max(len(punti), 1)
    autoval, autovett = np.linalg.eigh(cov)
    ordine = np.argsort(autoval)[::-1]
    return autoval[ordine].clip(min=0.0), autovett[:, ordine]


def _dimensioni_obb(punti: np.ndarray, autovett: np.ndarray) -> np.ndarray:
    """Lati della bounding box orientata secondo gli assi PCA, decrescenti."""
    proiezioni = (punti - punti.mean(axis=0)) @ autovett
    lati = proiezioni.max(axis=0) - proiezioni.min(axis=0)
    return np.sort(lati)[::-1]


def _fit_cerchio_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit algebrico di cerchio (metodo di Kasa): ritorna (centro (2,), raggio)."""
    a = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centro = sol[:2]
    raggio_q = sol[2] + centro @ centro
    return centro, float(np.sqrt(max(raggio_q, 0.0)))


def _fit_cerchio_ransac(
    xy: np.ndarray, raggio_massimo: float, iterazioni: int = 150
) -> tuple[np.ndarray, float]:
    """RANSAC per il cerchio di sezione: robusto alle appendici (calate
    sprinkler, valvole, staffe) che distruggono il fit ai minimi quadrati.

    Ritorna (centro, raggio); il candidato con piu' inlier viene raffinato
    con un fit Kasa sugli inlier stessi.
    """
    rng = np.random.default_rng(0)
    campione = xy
    if len(xy) > 3000:  # sottocampiona per velocita': basta per lo scoring
        campione = xy[rng.choice(len(xy), 3000, replace=False)]

    miglior_punteggio, miglior_centro, miglior_raggio = -1.0, None, 0.0
    for _ in range(iterazioni):
        tre = campione[rng.choice(len(campione), 3, replace=False)]
        # circumcentro dei tre punti
        a2 = 2.0 * (tre[1] - tre[0])
        b2 = 2.0 * (tre[2] - tre[0])
        det = a2[0] * b2[1] - a2[1] * b2[0]
        if abs(det) < 1e-12:
            continue
        d1 = (tre[1] ** 2 - tre[0] ** 2).sum()
        d2 = (tre[2] ** 2 - tre[0] ** 2).sum()
        centro = np.array(
            [(d1 * b2[1] - d2 * a2[1]) / det, (d2 * a2[0] - d1 * b2[0]) / det]
        )
        raggio = float(np.linalg.norm(tre[0] - centro))
        if raggio <= 1e-4 or raggio > raggio_massimo:
            continue
        # tolleranza limitata: circonferenze enormi quasi rettilinee non
        # devono vincere raccogliendo inlier con tolleranze larghissime
        tol = float(np.clip(0.15 * raggio, 0.01, 0.04))
        scarti = campione - centro
        residui = np.abs(np.linalg.norm(scarti, axis=1) - raggio)
        inlier = residui < tol
        n_inlier = int(inlier.sum())
        if n_inlier < 3:
            continue
        # lo score premia anche la copertura angolare: un arco gigante quasi
        # rettilineo puo' inglobare un intero tubo piccolo nella sua banda di
        # tolleranza, ma i suoi inlier occupano un settore angolare minuscolo.
        angoli = np.arctan2(scarti[inlier, 1], scarti[inlier, 0])
        bins_occupati = len(np.unique((angoli / (2 * np.pi / 18)).astype(np.int64)))
        punteggio = n_inlier * (bins_occupati / 18.0)
        if punteggio > miglior_punteggio:
            miglior_punteggio, miglior_centro, miglior_raggio = (
                punteggio, centro, raggio,
            )

    if miglior_centro is None:
        return _fit_cerchio_2d(xy)

    # raffinamento: due giri di Kasa sugli inlier del modello corrente
    centro, raggio = miglior_centro, miglior_raggio
    for _ in range(2):
        tol = float(np.clip(0.15 * raggio, 0.01, 0.04))
        residui = np.abs(np.linalg.norm(xy - centro, axis=1) - raggio)
        inlier = residui < tol
        if inlier.sum() < 10:
            break
        centro, raggio = _fit_cerchio_2d(xy[inlier])
        if raggio <= 1e-4 or raggio > raggio_massimo:
            return miglior_centro, miglior_raggio
    return centro, raggio


def fitta_cilindro(punti: np.ndarray, normali: np.ndarray, cfg: dict) -> FitCilindro:
    """Fitting di cilindro su un cluster.

    L'asse e' l'autovettore con autovalore minimo della matrice di scatter
    delle normali (per un cilindro le normali giacciono nel piano ortogonale
    all'asse). I punti vengono proiettati sulla sezione e vi si adatta un
    cerchio; l'inlier ratio misura quanto il cluster e' davvero cilindrico.
    """
    if len(punti) < 20:
        return FitCilindro()

    # scatter delle normali (invariante al segno n / -n)
    m = normali.T @ normali
    autoval, autovett = np.linalg.eigh(m)
    asse = autovett[:, 0]  # autovalore minimo
    asse /= np.linalg.norm(asse)

    # base ortonormale del piano di sezione
    ausiliario = np.array([1.0, 0.0, 0.0])
    if abs(asse @ ausiliario) > 0.9:
        ausiliario = np.array([0.0, 1.0, 0.0])
    u = np.cross(asse, ausiliario)
    u /= np.linalg.norm(u)
    v = np.cross(asse, u)

    centrati = punti - punti.mean(axis=0)
    xy = np.column_stack([centrati @ u, centrati @ v])
    raggio_massimo = float(cfg.get("raggio_massimo", 1.5))
    centro_2d, raggio = _fit_cerchio_ransac(xy, raggio_massimo)
    if raggio <= 1e-4 or raggio > raggio_massimo:
        return FitCilindro()

    dist_radiale = np.linalg.norm(xy - centro_2d, axis=1)
    residui = np.abs(dist_radiale - raggio)
    tolleranza = max(
        float(cfg.get("tolleranza_min", 0.01)),
        float(cfg.get("tolleranza_relativa", 0.15)) * raggio,
    )
    inlier_ratio = float((residui < tolleranza).mean())

    # consistenza della sezione: un vero cilindro (anche visto da un solo
    # lato, quindi mezzo guscio) ha un'estensione di sezione almeno pari al
    # raggio in entrambe le direzioni; una passerella "abbraccia" il cerchio
    # ma ha sezione molto piu' bassa del raggio fittato.
    inlier_sez = residui < tolleranza
    xy_sez = xy[inlier_sez] if inlier_sez.sum() >= 10 else xy
    estensioni_sez = xy_sez.max(axis=0) - xy_sez.min(axis=0)
    sezione_su_raggio = float(estensioni_sez.min() / max(raggio, 1e-9))

    # coerenza radiale: su un vero cilindro la normale di ogni punto e'
    # diretta verso l'asse; una passerella o una scatola possono "abbracciare"
    # un cerchio con pochi residui, ma le loro normali non sono radiali.
    direzione_radiale = (xy - centro_2d) / (dist_radiale[:, None] + 1e-12)
    normali_2d = np.column_stack([normali @ u, normali @ v])
    modulo_2d = np.linalg.norm(normali_2d, axis=1)
    allineamento = np.abs((normali_2d * direzione_radiale).sum(axis=1)) / (
        modulo_2d + 1e-12
    )
    # i punti con normale quasi parallela all'asse (tappi, appendici) non
    # possono essere coerenti: modulo_2d basso li esclude automaticamente.
    coerenza_radiale = float(((allineamento > 0.8) & (modulo_2d > 0.3)).mean())

    # distribuzione angolare delle normali di sezione: su un cilindro le
    # normali spazzano un arco continuo; su un box (condotta rettangolare)
    # si concentrano in 2-3 direzioni discrete. Un rettangolo con lati
    # simili puo' superare inlier e coerenza radiale, ma non questo test.
    valide = modulo_2d > 0.3
    if valide.sum() >= 10:
        angoli_normali = np.mod(
            np.arctan2(normali_2d[valide, 1], normali_2d[valide, 0]), np.pi
        )
        # doppio istogramma, il secondo sfasato di mezzo bin: una direzione
        # discreta che cade sul BORDO di un bin si spalma su due bin
        # adiacenti e sfuggirebbe al top-2 (un box con facce allineate agli
        # assi del fit misurava 0.55 invece di ~0.83)
        mezzo_bin = np.pi / 24
        concentrazione_normali = 0.0
        for sfasamento in (0.0, mezzo_bin):
            angoli_s = np.mod(angoli_normali + sfasamento, np.pi)
            istogramma, _ = np.histogram(angoli_s, bins=12, range=(0.0, np.pi))
            top2 = np.sort(istogramma)[-2:].sum()
            concentrazione_normali = max(
                concentrazione_normali,
                float(top2 / max(istogramma.sum(), 1)),
            )
    else:
        concentrazione_normali = 1.0

    quota_assiale = centrati @ asse
    lunghezza = float(quota_assiale.max() - quota_assiale.min())
    centro_3d = punti.mean(axis=0) + u * centro_2d[0] + v * centro_2d[1]

    return FitCilindro(
        valido=True,
        raggio=raggio,
        inlier_ratio=inlier_ratio,
        coerenza_radiale=coerenza_radiale,
        sezione_su_raggio=sezione_su_raggio,
        concentrazione_normali=concentrazione_normali,
        asse=asse,
        centro=centro_3d,
        lunghezza=lunghezza,
        residui=residui,
    )


# distanza RGB (0-255) entro cui un punto e' "dello stesso colore" del
# dominante, per il conto dell'uniformita' (tollera ombre/gradienti)
SOGLIA_UNIFORMITA_COLORE = 40.0


def _feature_colore(
    punti: np.ndarray, colori: np.ndarray | None
) -> tuple[np.ndarray | None, float | None]:
    """Colore dominante (mediana per canale) e uniformita' del cluster."""
    if colori is None or len(colori) != len(punti) or len(punti) == 0:
        return None, None
    col = np.asarray(colori, dtype=np.float32)
    dominante = np.median(col, axis=0)
    dist = np.linalg.norm(col - dominante, axis=1)
    uniformita = float((dist <= SOGLIA_UNIFORMITA_COLORE).mean())
    return dominante.round().clip(0, 255).astype(np.uint8), uniformita


def calcola_feature(
    punti: np.ndarray,
    normali: np.ndarray,
    quota_soffitto: float,
    quota_pavimento: float,
    cfg_cilindro: dict,
    sotto_tubazione: bool = False,
    colori: np.ndarray | None = None,
) -> FeatureCluster:
    autoval, autovett = _pca(punti)
    l1 = max(autoval[0], 1e-12)
    dimensioni = _dimensioni_obb(punti, autovett)
    quota_top = float(punti[:, 2].max())
    quota_bottom = float(punti[:, 2].min())
    # densita' rispetto al guscio dell'OBB: quanti punti ci si aspetterebbe
    # se le facce della scatola fossero superfici reali campionate al voxel
    voxel = max(float(cfg_cilindro.get("voxel_eff", 0.03)), 1e-3)
    a, b, c = np.maximum(dimensioni, voxel)
    area_guscio = 2.0 * (a * b + b * c + a * c)
    punti_attesi = area_guscio / voxel**2
    densita = float(len(punti) / max(punti_attesi, 1.0))
    return FeatureCluster(
        n_punti=len(punti),
        centro=punti.mean(axis=0),
        dimensioni=dimensioni,
        linearita=float((autoval[0] - autoval[1]) / l1),
        planarita=float((autoval[1] - autoval[2]) / l1),
        dispersione=float(autoval[2] / l1),
        verticalita_asse=float(abs(autovett[2, 0])),
        quota_top=quota_top,
        quota_bottom=quota_bottom,
        dist_da_soffitto=float(quota_soffitto - quota_top),
        dist_da_pavimento=float(quota_bottom - quota_pavimento),
        cilindro=fitta_cilindro(punti, normali, cfg_cilindro),
        sotto_tubazione=sotto_tubazione,
        estensione_z_robusta=float(
            np.percentile(punti[:, 2], 95) - np.percentile(punti[:, 2], 5)
        ),
        densita_superficie=densita,
        frazione_z_alta=float(
            max(
                (punti[:, 2] >= quota_top - 0.15).mean(),
                (punti[:, 2] <= quota_bottom + 0.15).mean(),
            )
        ),
        **dict(zip(("colore_dominante", "uniformita_colore"),
                   _feature_colore(punti, colori))),
    )
