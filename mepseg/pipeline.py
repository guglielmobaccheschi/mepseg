"""Pipeline principale di segmentazione MEP.

Fasi:
  1. caricamento e sottocampionamento voxel + normali
  2. rimozione elementi strutturali (RANSAC piani)
  3. clustering euclideo (DBSCAN) dei punti residui
  4. feature geometriche + classificazione a regole per cluster
  5. raffinamento: distacco di appendici dai cluster cilindrici
     (es. sprinkler appesi alle tubazioni) e riclassificazione
  6. (opzionale) fusione con le predizioni della rete neurale
  7. propagazione delle etichette alla nuvola a piena risoluzione
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .classi import ClasseMEP
from .io_nuvole import Nuvola
from .preprocess import propaga_etichette, sottocampiona_e_prepara
from .primitive import FeatureCluster, calcola_feature
from .regole import classifica_cluster
from .strutture import rimuovi_strutture

log = logging.getLogger("mepseg")


@dataclass
class Istanza:
    id: int
    classe: ClasseMEP
    feature: FeatureCluster
    indici: np.ndarray = None  # indici dei punti (nuvola ridotta)


@dataclass
class RisultatoSegmentazione:
    etichette: np.ndarray            # (N,) codici classe sulla nuvola originale
    etichette_ridotte: np.ndarray    # (M,) sulla nuvola sottocampionata
    punti_ridotti: np.ndarray        # (M, 3)
    istanze: list[Istanza] = field(default_factory=list)
    quota_soffitto: float = 0.0
    quota_pavimento: float = 0.0
    piani_struttura: list[dict] = field(default_factory=list)
    # famiglia geometrica per punto (nuvola ridotta), se il clustering
    # "crescita" e' attivo: 0 nessuna/struttura, 1 piana orizzontale,
    # 2 piana verticale, 3 cilindrica, 4 irregolare (export di debug)
    famiglie_superfici: np.ndarray = None


def _clusterizza(punti: np.ndarray, cfg: dict) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(punti)
    etichette = np.asarray(
        pcd.cluster_dbscan(
            eps=float(cfg.get("eps", 0.08)),
            min_points=int(cfg.get("min_punti", 40)),
            print_progress=False,
        )
    )
    return etichette


def _sottocluster(punti: np.ndarray, eps: float, min_punti: int) -> np.ndarray:
    """DBSCAN leggero (scikit-learn) per il raffinamento delle appendici."""
    from sklearn.cluster import DBSCAN

    if len(punti) < min_punti:
        return np.full(len(punti), -1)
    return DBSCAN(eps=eps, min_samples=min_punti).fit_predict(punti)


def _spezza_per_erosione(
    idx: np.ndarray, punti_r: np.ndarray, cfg_cluster: dict,
    erosione: float | None = None,
) -> tuple[list, np.ndarray]:
    """Spezza un mega-cluster erodendone l'impronta in pianta.

    Le file di rack sono bande spesse; i "ponti" che le saldano tra loro
    (fasci di cavi, attraversamenti) sono sottili. Erodendo l'impronta i
    ponti si rompono: ogni punto viene poi assegnato alla componente
    superstite piu' vicina — ma solo entro ``raggio_briciole``: i punti
    piu' lontani da qualunque blocco sono l'alone di speckle che il
    clustering ha trascinato dentro il blob, non parti di un oggetto.
    Ritorna (lista dei blocchi >= 1, indici degli scarti fuori raggio).
    """
    from scipy import ndimage

    cella = float(cfg_cluster.get("cella_erosione", 0.15))
    raggio_briciole = float(cfg_cluster.get("raggio_briciole", 0.5))
    if erosione is None:
        erosione = float(cfg_cluster.get("erosione_spezza", 0.3))
    min_punti = max(int(cfg_cluster.get("min_punti", 10)), 10)
    nessuno = np.empty(0, dtype=idx.dtype)

    p = punti_r[idx]
    origine = p[:, :2].min(axis=0)
    ij = np.floor((p[:, :2] - origine) / cella).astype(np.int64)
    forma = ij.max(axis=0) + 1
    if forma[0] * forma[1] > 50_000_000:
        return [idx], nessuno
    occupate = np.zeros(forma, dtype=bool)
    occupate[ij[:, 0], ij[:, 1]] = True

    iterazioni = max(1, int(round(erosione / cella)))
    erose = ndimage.binary_erosion(occupate, iterations=iterazioni)
    etichette, n_comp = ndimage.label(erose, structure=np.ones((3, 3), int))
    if n_comp <= 1:
        return [idx], nessuno

    # ogni cella (anche quelle erose via) -> componente piu' vicina, con
    # guinzaglio: oltre il raggio la cella non appartiene a nessun blocco.
    # Il raggio DEVE scalare con l'erosione usata: l'erosione mangia il
    # bordo dei blocchi per l'intera profondita' di erosione, e quel bordo
    # e' carne viva da restituire; l'alone di speckle sta molto piu' in la'
    guinzaglio = max(raggio_briciole, erosione + 2 * cella)
    dist_celle, (ii, jj) = ndimage.distance_transform_edt(
        etichette == 0, return_indices=True
    )
    etichette_piene = etichette[ii, jj]
    etichette_piene[dist_celle > guinzaglio / cella] = 0
    comp_punto = etichette_piene[ij[:, 0], ij[:, 1]]
    scarti = idx[comp_punto == 0]

    blocchi, piccoli = [], []
    for c in range(1, n_comp + 1):
        idx_c = idx[comp_punto == c]
        (blocchi if len(idx_c) >= min_punti else piccoli).append(idx_c)
    if not blocchi:
        return [idx], nessuno
    if piccoli:
        # le briciole VICINE confluiscono nel blocco maggiore
        maggiore = int(np.argmax([len(b) for b in blocchi]))
        blocchi[maggiore] = np.concatenate([blocchi[maggiore]] + piccoli)
    return blocchi, scarti


def _spezza_per_vuoti(
    idx: np.ndarray, punti_r: np.ndarray, cfg_cluster: dict
) -> list:
    """Spezza un cluster snello con VUOTI PERIODICI lungo l'asse.

    Una fila di neon collegati da un cavo o saldati dal clustering diventa
    un unico elemento allungato che le regole scambiano per passerella o
    condotta. Ma una fila ha una firma inconfondibile: 3+ blocchi densi
    separati da intervalli quasi vuoti. Un oggetto continuo occluso produce
    al piu' 1-2 interruzioni. Ritorna la lista dei segmenti (>= 1)."""
    gap_min = float(cfg_cluster.get("gap_min_vuoti", 0.15))
    min_seg = int(cfg_cluster.get("min_segmenti_vuoti", 3))
    sezione_max = float(cfg_cluster.get("sezione_max_vuoti", 0.8))
    min_punti = max(int(cfg_cluster.get("min_punti", 10)), 10)

    p = punti_r[idx]
    if len(p) < min_seg * min_punti:
        return [idx]
    centrati = p - p.mean(axis=0)
    cov = centrati.T @ centrati / len(p)
    autoval, autovett = np.linalg.eigh(cov)
    asse = autovett[:, np.argmax(autoval)]
    t = centrati @ asse
    estensione = float(t.max() - t.min())
    if estensione < 0.8 or estensione > 12.0:
        return [idx]
    # snello: sezione trasversale contenuta
    trasversale = centrati - np.outer(t, asse)
    if float(np.linalg.norm(trasversale, axis=1).max()) > sezione_max:
        return [idx]

    passo = 0.05
    bins = np.floor((t - t.min()) / passo).astype(np.int64)
    occupato = np.zeros(int(bins.max()) + 1, dtype=bool)
    occupato[bins] = True
    # segmenti = sequenze di bin occupati separate da vuoti >= gap_min
    gap_bins = max(1, int(round(gap_min / passo)))
    segmenti, inizio, vuoti = [], None, 0
    for k, occ in enumerate(occupato):
        if occ:
            if inizio is None:
                inizio = k
            vuoti = 0
        elif inizio is not None:
            vuoti += 1
            if vuoti >= gap_bins:
                segmenti.append((inizio, k - vuoti + 1))
                inizio = None
    if inizio is not None:
        segmenti.append((inizio, len(occupato)))
    if len(segmenti) < min_seg:
        return [idx]
    esito = []
    for a, b in segmenti:
        m = (bins >= a) & (bins < b)
        if m.sum() >= min_punti:
            esito.append(idx[m])
    return esito if len(esito) >= min_seg else [idx]


def _spezza_cluster_giganti(
    liste_cluster: list, punti_r: np.ndarray, strutture, cfg_cluster: dict
) -> tuple[list, list]:
    """Spezza i cluster DBSCAN troppo estesi per essere un solo oggetto.

    In un locale denso (file di rack, passerelle, cavi) DBSCAN salda tutto
    in un unico "mega-cluster" grande quanto il locale: le luci e le
    passerelle appese al soffitto ne vengono inghiottite e finiscono
    classificate col blob. Da ogni cluster con ingombro in pianta oltre la
    soglia si stacca la fascia sotto il soffitto LOCALE e la si
    ri-clusterizza: i sotto-cluster tornano al normale percorso di
    classificazione (luci, passerelle, condotte), il resto rimane un unico
    cluster (tipicamente apparecchiatura).
    """
    lato_max = float(cfg_cluster.get("lato_max_spezza", 8.0))
    fascia = float(cfg_cluster.get("fascia_soffitto_spezza", 1.2))
    lato_min_erosione = float(cfg_cluster.get("lato_min_erosione", 4.0))
    eps = float(cfg_cluster.get("eps", 0.08))
    min_punti = max(int(cfg_cluster.get("min_punti", 10)), 10)
    soffitti = [o for o in strutture.orizzontali if o.tipo == "soffitto"]
    scarti_totali: list[np.ndarray] = []  # alone dei blob, fuori guinzaglio

    # quote dei pavimenti accettati (fuse se vicine): un blob che si
    # sviluppa su piu' livelli attraverso i cavedi va prima tagliato per
    # livello, l'erosione in pianta da sola non puo' separarlo
    quote_pav = []
    for o in sorted(
        (o for o in strutture.orizzontali if o.tipo == "pavimento"),
        key=lambda o: o.quota,
    ):
        if not quote_pav or o.quota - quote_pav[-1] > 0.5:
            quote_pav.append(o.quota)

    def spezza_se_blob(idx_blocco: np.ndarray) -> list:
        """Taglio per livello + erosione ricorsiva in pianta, ma solo per i
        blob estesi in ENTRAMBE le direzioni: i nastri lunghi e stretti
        (condotte, passerelle) sono oggetti singoli legittimi."""

        def è_blob(b: np.ndarray) -> bool:
            p2 = punti_r[b][:, :2]
            lati2 = p2.max(axis=0) - p2.min(axis=0)
            return float(lati2.min()) > lato_min_erosione

        if not è_blob(idx_blocco):
            return [idx_blocco]

        # 1. taglio per livello (sopra ogni quota pavimento, con margine)
        da_erodere = [idx_blocco]
        if len(quote_pav) > 1:
            z = punti_r[idx_blocco][:, 2]
            livello = np.searchsorted(np.array(quote_pav) + 0.15, z)
            gruppi = [
                idx_blocco[livello == l]
                for l in np.unique(livello)
            ]
            gruppi = [g for g in gruppi if len(g) >= min_punti]
            if len(gruppi) > 1:
                da_erodere = gruppi

        # 2. erosione ricorsiva: se un blocco resta un blob, si riprova
        # con un raggio di erosione raddoppiato (fino a 4x)
        erosione_base = float(cfg_cluster.get("erosione_spezza", 0.3))
        blocchi, coda = [], [(b, erosione_base) for b in da_erodere]
        while coda:
            b, er = coda.pop()
            if not è_blob(b):
                blocchi.append(b)
                continue
            sub, scarti = _spezza_per_erosione(b, punti_r, cfg_cluster, er)
            if len(sub) == 1:
                if er >= 4 * erosione_base - 1e-9:
                    blocchi.append(b)  # non si spezza piu': tienilo intero
                else:
                    coda.append((b, 2 * er))
            else:
                if len(scarti):
                    scarti_totali.append(scarti)
                coda.extend((s, er) for s in sub)
        if len(blocchi) > 1:
            p2 = punti_r[idx_blocco][:, :2]
            lati2 = p2.max(axis=0) - p2.min(axis=0)
            log.info(
                "Mega-cluster residuo (%.0f x %.0f m in pianta, %s punti): "
                "spezzato per livello+erosione in %d blocchi",
                lati2[0], lati2[1], f"{len(idx_blocco):,}", len(blocchi),
            )
        return blocchi

    esito = []
    for idx in liste_cluster:
        p = punti_r[idx]
        lati_xy = p[:, :2].max(axis=0) - p[:, :2].min(axis=0)
        if float(max(lati_xy)) <= lato_max:
            esito.append(idx)
            continue

        # soffitto locale per punto: la quota piu' bassa tra i soffitti
        # accettati che sovrastano il punto in pianta
        soffitto_loc = np.full(len(p), strutture.quota_soffitto)
        for o in soffitti:
            x0, y0, x1, y1 = o.bbox_xy
            m = (
                (p[:, 0] >= x0 - 0.5) & (p[:, 0] <= x1 + 0.5)
                & (p[:, 1] >= y0 - 0.5) & (p[:, 1] <= y1 + 0.5)
                & (p[:, 2] <= o.quota + 0.3)
                & (o.quota < soffitto_loc)
            )
            soffitto_loc[m] = o.quota

        in_fascia = p[:, 2] >= soffitto_loc - fascia
        n_fascia = int(in_fascia.sum())
        if n_fascia < min_punti or n_fascia == len(idx):
            esito.extend(spezza_se_blob(idx))
            continue

        idx_fascia = idx[in_fascia]
        sub_ids = _clusterizza(
            punti_r[idx_fascia], {"eps": eps, "min_punti": min_punti}
        )
        n_sub = int(sub_ids.max()) + 1 if sub_ids.size else 0
        # il rumore della fascia resta col corpo del cluster
        resto = np.concatenate([idx[~in_fascia], idx_fascia[sub_ids == -1]])
        if len(resto) >= min_punti:
            esito.extend(spezza_se_blob(resto))
        for s in range(n_sub):
            # anche i sotto-cluster della fascia possono essere "tappeti"
            # continui di cavi/passerelle grandi quanto la sala
            esito.extend(spezza_se_blob(idx_fascia[sub_ids == s]))
        log.info(
            "Cluster gigante (%.0f x %.0f m in pianta, %s punti): staccata "
            "la fascia soffitto -> %d sotto-cluster",
            lati_xy[0], lati_xy[1], f"{len(idx):,}", n_sub,
        )
    return esito, scarti_totali


def segmenta(nuvola: Nuvola, cfg: dict) -> RisultatoSegmentazione:
    """Esegue la pipeline completa su una :class:`Nuvola`."""
    import open3d as o3d

    # risultati riproducibili: il RANSAC dei piani di Open3D altrimenti
    # cambia leggermente a ogni esecuzione, e i cluster borderline
    # potrebbero cambiare classe da un run all'altro
    try:
        o3d.utility.random.seed(0)
    except AttributeError:  # versioni di Open3D senza seed globale
        pass

    log.info("Punti in ingresso: %d", len(nuvola))

    # 1. preprocess ---------------------------------------------------------
    punti_r, normali_r, colori_r = sottocampiona_e_prepara(
        nuvola.punti, cfg["preprocess"], nuvola.colori
    )
    log.info("Punti dopo sottocampionamento: %d", len(punti_r))
    if colori_r is not None:
        log.info("Colore RGB disponibile: attivo il taglio per colore nel clustering")

    etichette_r = np.full(len(punti_r), int(ClasseMEP.MEP_GENERICO), dtype=np.uint8)

    # 2. strutture ----------------------------------------------------------
    voxel_eff = max(
        float(cfg["preprocess"].get("voxel", 0.03)),
        float(cfg["preprocess"].get("voxel_streaming", 0.0)),
    ) or 0.03
    cfg_strutture = dict(cfg["strutture"])
    cfg_strutture["voxel_eff"] = voxel_eff
    strutture = rimuovi_strutture(punti_r, normali_r, cfg_strutture)
    # ogni componente strutturale accettata porta il proprio tipo:
    # PAVIMENTO, SOFFITTO o PARETE (STRUTTURA resta come fallback generico)
    etichette_r[strutture.maschera_struttura] = strutture.etichette_struttura[
        strutture.maschera_struttura
    ]
    log.info(
        "Struttura: %d punti (%d piani), soffitto a %.2f m, pavimento a %.2f m",
        int(strutture.maschera_struttura.sum()),
        len(strutture.piani),
        strutture.quota_soffitto,
        strutture.quota_pavimento,
    )

    maschera_mep = ~strutture.maschera_struttura
    cfg_strutture_pipe = cfg.get("strutture", {})

    # 2bis. Fase 0 architettonica (rilevamento progressivo): sotto il
    # pavimento OSSERVATO della propria colonna non esiste nulla di reale
    # — solo le immagini speculari degli oggetti sopra, riflesse dal
    # pavimento tecnico lucido. La mappa dei pavimenti (per livello: le
    # quote di piani diversi non si mischiano) permette il taglio per
    # punto: i riflessi diventano waste ed ESCONO dal pool di ricerca MEP
    # prima del completamento pareti (un muro riflesso e' coplanare al
    # muro vero: verrebbe assorbito) e del clustering.
    mappa_pav = None
    if cfg_strutture_pipe.get("mappa_pavimenti", True):
        from .strutture import costruisci_mappa_pavimenti

        mappa_pav = costruisci_mappa_pavimenti(
            punti_r, etichette_r, cfg_strutture_pipe
        )
    if mappa_pav is not None:
        tolleranza = float(
            cfg_strutture_pipe.get("tolleranza_sotto_pavimento", 0.20)
        )
        quota_colonna = mappa_pav.quota_minima_colonna(punti_r[:, :2])
        with np.errstate(invalid="ignore"):
            riflessi = punti_r[:, 2] < quota_colonna - tolleranza
        # i pavimenti definiscono la mappa e i soffitti veri (intradossi
        # del livello sotto) possono stare sotto un pavimento superiore:
        # non si toccano
        riflessi &= (etichette_r != int(ClasseMEP.PAVIMENTO)) & (
            etichette_r != int(ClasseMEP.SOFFITTO)
        )
        if riflessi.any():
            etichette_r[riflessi] = int(ClasseMEP.WASTE)
            maschera_mep &= ~riflessi
            log.info(
                "Fase 0 architettonica: %s punti sotto il pavimento della "
                "propria colonna -> waste (riflessi)",
                f"{int(riflessi.sum()):,}",
            )

    # 2ter. completamento pareti PER PUNTO: le fasce di muro sopra i rack
    # vengono bocciate dai test di accettazione (base occlusa) e finirebbero
    # saldate nei cluster degli oggetti adiacenti, dove nessuna regola
    # per-cluster puo' piu' separarle. Un punto coplanare a una parete
    # accettata (stesso piano, normale allineata) e' muro: si etichetta
    # subito e non partecipa al clustering.
    if cfg_strutture_pipe.get("completa_pareti", True):
        soglia_cp = float(cfg_strutture_pipe.get("dist_recupero_residui", 0.05))
        n_completati = 0
        idx_cand = np.where(maschera_mep)[0]
        p_cand = punti_r[idx_cand]
        n_cand = normali_r[idx_cand]
        for pi in strutture.piani:
            if pi["tipo"] != "parete" or "d" not in pi:
                continue
            norm_p = np.asarray(pi["normale"])
            bbox_p = np.asarray(pi["bbox"])
            vicino = (
                (p_cand >= bbox_p[:3] - 3.0) & (p_cand <= bbox_p[3:] + 3.0)
            ).all(axis=1)
            vicino &= np.abs(p_cand @ norm_p + float(pi["d"])) <= soglia_cp
            # allineamento severo: su un guscio curvo tangente al muro solo
            # una banda sottilissima supera 0.95, un muro vero e' >= 0.98 —
            # con 0.8 il completamento mutilava le condotte rasenti
            vicino &= np.abs(n_cand @ norm_p) >= float(
                cfg_strutture_pipe.get("allineamento_completa_pareti", 0.95)
            )
            if vicino.any():
                idx_v = idx_cand[vicino]
                etichette_r[idx_v] = int(ClasseMEP.PARETE)
                maschera_mep[idx_v] = False
                n_completati += int(vicino.sum())
        if n_completati:
            log.info(
                "Completamento pareti: %s punti coplanari a pareti accettate",
                f"{n_completati:,}",
            )

    # 3. clustering dei punti non strutturali -------------------------------
    idx_mep = np.where(maschera_mep)[0]
    istanze: list[Istanza] = []
    famiglie_superfici = None
    if len(idx_mep) > 0:
        cfg_cluster = dict(cfg["clustering"])
        # eps deve scalare con il voxel, altrimenti a voxel grossi la
        # connettivita' si rompe e i cluster si frammentano
        voxel = max(
            float(cfg["preprocess"].get("voxel", 0.03)),
            float(cfg["preprocess"].get("voxel_streaming", 0.0)),
        )
        cfg_cluster["eps"] = max(float(cfg_cluster.get("eps", 0.08)), 2.5 * voxel)
        cfg_cluster["voxel_eff"] = voxel or 0.03

        # sbarramento di evidenza PER PUNTO: un punto senza un minimo di
        # vicinato (contro TUTTA la nuvola: un impianto attaccato al muro
        # ha i vicini del muro) non appartiene a nessuna superficie
        # scansionata — e' un fantasma multipath. Fuori dal grafo PRIMA del
        # clustering: da dentro, le catene di speckle si saldano agli
        # oggetti veri e nessuna regola a valle le separa piu'.
        if cfg_cluster.get("evidenza_punto", True) and len(idx_mep):
            from scipy.spatial import cKDTree

            raggio_ev = float(cfg_cluster.get("raggio_evidenza", 2.5)) * (
                voxel or 0.03
            )
            min_vicini = int(cfg_cluster.get("min_vicini_evidenza", 4))
            albero_ev = cKDTree(punti_r)
            dist_ev, _ = albero_ev.query(
                punti_r[idx_mep], k=min_vicini + 1,
                distance_upper_bound=raggio_ev, workers=-1,
            )
            fantasma = ~np.isfinite(dist_ev[:, min_vicini])
            if fantasma.any():
                etichette_r[idx_mep[fantasma]] = int(ClasseMEP.WASTE)
                idx_mep = idx_mep[~fantasma]
                log.info(
                    "Evidenza per punto: %s punti senza vicinato (fantasmi "
                    "multipath) -> waste",
                    f"{int(fantasma.sum()):,}",
                )

        metodo = str(cfg_cluster.get("metodo", "dbscan")).lower()
        if metodo.startswith("crescita"):
            from .crescita import clusterizza_crescita

            info_crescita: dict = {}
            cluster_ids = clusterizza_crescita(
                punti_r[idx_mep], normali_r[idx_mep], cfg_cluster,
                info=info_crescita,
                colori=colori_r[idx_mep] if colori_r is not None else None,
            )
            if "famiglie" in info_crescita:
                famiglie_superfici = np.zeros(len(punti_r), dtype=np.uint8)
                famiglie_superfici[idx_mep] = info_crescita["famiglie"]
        else:
            cluster_ids = _clusterizza(punti_r[idx_mep], cfg_cluster)
        n_cluster = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
        log.info("Cluster MEP trovati: %d", n_cluster)

        # i cluster grandi quanto un locale (file di rack saldate da cavi e
        # passerelle) inghiottono le luci a soffitto: si spezzano prima di
        # classificare
        liste_cluster = [idx_mep[cluster_ids == cid] for cid in range(n_cluster)]
        liste_cluster, scarti_blob = _spezza_cluster_giganti(
            liste_cluster, punti_r, strutture, cfg_cluster
        )
        # i punti oltre il guinzaglio da qualunque blocco superstite non
        # ereditano la classe di un blocco lontano: sono le zone del blob
        # che l'erosione non ha saputo strutturare (tappeti di cavi,
        # margini) — restano MEP generico, il bucket onesto per il
        # finder/DL (i fantasmi veri li ha gia' tolti l'evidenza per punto)
        if scarti_blob:
            idx_scarti = np.concatenate(scarti_blob)
            etichette_r[idx_scarti] = int(ClasseMEP.MEP_GENERICO)
            log.info(
                "Spezza-blob: %s punti fuori guinzaglio -> MEP generico "
                "(nessuna eredita' dai blocchi lontani)",
                f"{len(idx_scarti):,}",
            )
        # file di elementi ripetuti saldate in un unico nastro (neon
        # collegati da un cavo): si spezzano ai vuoti periodici, poi le
        # regole di contesto (serialita', catene) ricompongono il senso
        if cfg_cluster.get("spezza_vuoti", True):
            liste_cluster = [
                seg
                for idx_c in liste_cluster
                for seg in _spezza_per_vuoti(idx_c, punti_r, cfg_cluster)
            ]
        # separazione per orientamento (RANSAC sulle normali, letteratura
        # Li/Gan/Wang 2025): istanze con assi diversi fuse in un cluster
        # (top rack + condotta, ex-id 56) si separano per direzione
        # dominante PRIMA delle regole. ATTIVA di default (20/07), protetta
        # dal pre-gate geometrico in orientamento.py (solo cluster sottili)
        if cfg_cluster.get("separa_orientamento", True):
            from .orientamento import separa_per_orientamento

            liste_cluster = [
                seg
                for idx_c in liste_cluster
                for seg in separa_per_orientamento(
                    idx_c, punti_r, normali_r, cfg_cluster
                )
            ]

        cfg_cil = cfg.get("cilindro", {})
        cfg_cil = dict(cfg_cil)
        cfg_cil["voxel_eff"] = voxel or 0.03
        cfg_regole = cfg.get("regole", {})
        eps = float(cfg_cluster["eps"])
        min_punti_sub = max(int(cfg_cluster.get("min_punti", 10)), 10)
        prossimo_id = 1

        # recupero dei residui strutturali: un cluster sottile che giace sul
        # piano di una parete/soffitto gia' accettati (serramenti, pezzi di
        # muro bocciati dai test di copertura per porte e occlusioni) e'
        # struttura, non un impianto
        from .strutture import CLASSE_PER_TIPO

        soglia_res = float(cfg_strutture.get("dist_recupero_residui", 0.05))
        fraz_res = float(cfg_strutture.get("frazione_recupero_residui", 0.8))
        spess_res = float(cfg_strutture.get("spessore_max_residuo", 0.12))
        piani_geom = [
            (
                np.asarray(pi["normale"]),
                float(pi["d"]),
                np.asarray(pi["bbox"]),
                CLASSE_PER_TIPO[pi["tipo"]],
            )
            for pi in strutture.piani
            # niente soffitti: le plafoniere a filo appena liberate
            # dall'estrazione sporgenze sono coplanari al soffitto e
            # verrebbero riassorbite
            if "d" in pi and pi["tipo"] in ("parete", "pavimento")
        ]

        def residuo_strutturale(p: np.ndarray, feat) -> int | None:
            """Codice classe strutturale se il cluster giace su un piano
            accettato, altrimenti None."""
            if feat.dimensioni[2] > spess_res or not piani_geom:
                return None
            for norm_p, d_p, bbox_p, codice in piani_geom:
                # per le pareti il bbox si estende lungo il piano: la fascia
                # di muro sopra i rack e' coplanare alla parte accettata ma
                # fuori dal suo ingombro (il gate vero e' la coplanarita')
                margine = 2.0 if codice == CLASSE_PER_TIPO["parete"] else 0.3
                dentro = (
                    (p >= bbox_p[:3] - margine) & (p <= bbox_p[3:] + margine)
                ).all(axis=1)
                if dentro.mean() < fraz_res:
                    continue
                dist = np.abs(p @ norm_p + d_p)
                if ((dist <= soglia_res) & dentro).mean() >= fraz_res:
                    return codice
            return None

        fonti_pavimento = {"diretto": 0, "mappa": 0, "globale": 0}

        def quote_locali(p: np.ndarray) -> tuple[float, float]:
            """Soffitto/pavimento locali: il piano orizzontale accettato piu'
            vicino sopra (sotto) il cluster E sovrapposto in pianta. Nelle
            scene multi-locale/multi-livello ogni ambiente ha il suo
            riferimento, senza prendere in prestito il soffitto di un'ala
            diversa dell'edificio."""
            top, bottom = float(p[:, 2].max()), float(p[:, 2].min())
            x0, y0 = float(p[:, 0].min()) - 0.5, float(p[:, 1].min()) - 0.5
            x1, y1 = float(p[:, 0].max()) + 0.5, float(p[:, 1].max()) + 0.5
            sopra, sopra_pav, sotto = [], [], []
            for piano in strutture.orizzontali:
                bx0, by0, bx1, by1 = piano.bbox_xy
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue  # nessuna sovrapposizione in pianta
                # riferimenti TIPIZZATI: un soffitto non puo' fare da
                # pavimento locale (una luce a filo soffitto risulterebbe
                # "sotto il pavimento"). In direzione opposta vale il
                # ripiego fisico: se sopra il cluster non c'e' un soffitto
                # accettato, l'intradosso del PAVIMENTO del piano superiore
                # e' il soffitto del locale.
                if piano.quota >= top - 0.3:
                    (sopra if piano.tipo == "soffitto" else sopra_pav).append(
                        piano.quota
                    )
                if piano.tipo == "pavimento" and piano.quota <= bottom + 0.3:
                    sotto.append(piano.quota)
            soffitto = (
                min(sopra) if sopra
                else (min(sopra_pav) if sopra_pav else strutture.quota_soffitto)
            )
            # pavimento: (1) osservato nel locale; (2) EREDITATO dalla
            # mappa dei pavimenti — dove il pavimento e' occluso (sale
            # rack) risponde la colonna piu' vicina in cui e' stato visto
            # davvero, sul livello giusto; (3) fallback globale
            if sotto:
                pavimento = max(sotto)
                fonti_pavimento["diretto"] += 1
            else:
                dalla_mappa = None
                if mappa_pav is not None:
                    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
                    dalla_mappa = mappa_pav.pavimento_locale(cx, cy, bottom + 0.3)
                if dalla_mappa is not None:
                    pavimento = dalla_mappa
                    fonti_pavimento["mappa"] += 1
                else:
                    pavimento = strutture.quota_pavimento
                    fonti_pavimento["globale"] += 1
            return soffitto, pavimento

        # cluster sotto la soglia di istanza: restano MEP generico (rumore
        # onesto, bersaglio del finder/DL) senza diventare istanze — niente
        # micro-frammenti in tabella e nelle regole. La soglia e' separata
        # da min_punti (che governa la meccanica del clustering) e NON si
        # applica alle appendici dei tubi (le teste sprinkler sono piccole).
        min_punti_istanza = int(cfg_cluster.get("min_punti_istanza", 100))
        n_micro = 0

        def col_di(idx):
            """Colore per-punto del sotto-insieme (o None senza RGB)."""
            return colori_r[idx] if colori_r is not None else None

        n_recuperati = 0
        for idx_cluster in liste_cluster:
            if len(idx_cluster) < min_punti_istanza:
                etichette_r[idx_cluster] = int(ClasseMEP.MEP_GENERICO)
                n_micro += 1
                continue
            p = punti_r[idx_cluster]
            n = normali_r[idx_cluster]
            quota_soff, quota_pav = quote_locali(p)
            feat = calcola_feature(
                p, n, quota_soff, quota_pav, cfg_cil, colori=col_di(idx_cluster)
            )

            codice_struttura = residuo_strutturale(p, feat)
            if codice_struttura is not None:
                etichette_r[idx_cluster] = codice_struttura
                n_recuperati += 1
                continue

            # 5. raffinamento: se il cluster e' in buona parte un cilindro
            # esteso (tubo, condotta), stacca i punti lontani dalla superficie
            # (calate sprinkler, valvole, staffe) prima di classificare, poi
            # riclassifica le appendici separatamente.
            idx_appendici = np.array([], dtype=np.int64)
            cil = feat.cilindro
            if (
                cil.valido
                and cil.residui is not None
                and cil.inlier_ratio >= 0.50
                and cil.coerenza_radiale >= 0.50
                and cil.sezione_su_raggio
                >= float(cfg_regole.get("min_sezione_su_raggio", 0.75))
                and cil.concentrazione_normali
                <= float(cfg_regole.get("max_concentrazione_normali", 0.55))
                and cil.lunghezza
                >= float(cfg_regole.get("lunghezza_min_cilindro", 0.4))
                and cil.raggio <= float(cfg_regole.get("raggio_max_condotta", 0.60))
            ):
                soglia = max(0.03, 0.5 * cil.raggio)
                fuori = cil.residui > soglia
                if min_punti_sub <= fuori.sum() < 0.5 * len(idx_cluster):
                    feat_core = calcola_feature(
                        punti_r[idx_cluster[~fuori]],
                        normali_r[idx_cluster[~fuori]],
                        quota_soff,
                        quota_pav,
                        cfg_cil,
                        colori=col_di(idx_cluster[~fuori]),
                    )
                    # il distacco ha senso solo se il nucleo ripulito e'
                    # davvero un tubo/condotta; altrimenti si annulla (evita
                    # di mutilare condotte rettangolari o apparecchiature).
                    if classifica_cluster(feat_core, cfg_regole) in (
                        ClasseMEP.TUBAZIONE,
                        ClasseMEP.CONDOTTA_CIRCOLARE,
                    ):
                        idx_appendici = idx_cluster[fuori]
                        idx_cluster = idx_cluster[~fuori]
                        feat = feat_core

            classe = classifica_cluster(feat, cfg_regole)

            etichette_r[idx_cluster] = int(classe)
            istanze.append(Istanza(prossimo_id, classe, feat, idx_cluster))
            prossimo_id += 1

            if len(idx_appendici) > 0:
                sub_ids = _sottocluster(punti_r[idx_appendici], eps, min_punti_sub)
                for sid in range(int(sub_ids.max()) + 1 if sub_ids.size else 0):
                    idx_sub = idx_appendici[sub_ids == sid]
                    feat_sub = calcola_feature(
                        punti_r[idx_sub],
                        normali_r[idx_sub],
                        quota_soff,
                        quota_pav,
                        cfg_cil,
                        sotto_tubazione=True,
                        colori=col_di(idx_sub),
                    )
                    classe_sub = classifica_cluster(feat_sub, cfg_regole)
                    etichette_r[idx_sub] = int(classe_sub)
                    istanze.append(Istanza(prossimo_id, classe_sub, feat_sub, idx_sub))
                    prossimo_id += 1
                # rumore residuo delle appendici: eredita la classe del tubo
                idx_rumore = idx_appendici[sub_ids == -1]
                etichette_r[idx_rumore] = int(classe)

        if n_micro:
            log.info(
                "Micro-cluster sotto %d punti: %d lasciati a MEP generico "
                "(nessuna istanza)", min_punti_istanza, n_micro,
            )
        if n_recuperati:
            log.info(
                "Residui strutturali recuperati: %d cluster coplanari a "
                "piani accettati (serramenti, frammenti di muro)",
                n_recuperati,
            )
        if fonti_pavimento["mappa"] or fonti_pavimento["globale"]:
            log.info(
                "Pavimento locale dei cluster: %d osservato, %d dalla mappa "
                "dei pavimenti, %d dal fallback globale",
                fonti_pavimento["diretto"], fonti_pavimento["mappa"],
                fonti_pavimento["globale"],
            )

        # punti rumore DBSCAN: restano MEP generico
        idx_rumore = idx_mep[cluster_ids == -1]
        etichette_r[idx_rumore] = int(ClasseMEP.MEP_GENERICO)

        # 5b. regole di contesto: coerenza topologica degli impianti
        cfg_contesto = cfg.get("contesto", {})
        if cfg_contesto.get("luci_seriali", True):
            from .contesto import riclassifica_luci_seriali

            riclassifica_luci_seriali(istanze, punti_r, etichette_r, cfg_contesto)
        if cfg_contesto.get("ancoraggio", True):
            from .contesto import applica_ancoraggio

            applica_ancoraggio(istanze, punti_r, etichette_r, cfg_contesto)
        # DOPO l'ancoraggio: la catena ripara anche i segmenti declassati
        # ad arredo perche' il taglio gli aveva staccato la continuazione
        if cfg_contesto.get("catene", True):
            from .contesto import unifica_catene

            unifica_catene(istanze, punti_r, etichette_r, cfg_contesto)

        # 5b-bis. spareggio per colore (opt-in): dove la geometria si e'
        # arresa (MEP generico) e la nuvola ha RGB, il colore dominante puo'
        # combaciare con una convenzione dell'impianto fornita dall'utente
        # (canaline gialle, blindosbarre blu). Solo su generico, mai sovrascrive
        # la geometria; vuoto di default (il tool resta agnostico sul colore)
        guida_colore = cfg.get("regole", {}).get("colori_guida") or []
        if guida_colore:
            from .regole import applica_guida_colore

            applica_guida_colore(
                istanze, etichette_r, guida_colore, cfg.get("regole", {})
            )

        # 5c. modelli confermati dall'utente: le istanze rimaste dubbie che
        # combaciano con la firma geometrica di un oggetto gia' confermato
        # (stessa plafoniera, stesso rack...) ne ereditano la classe. Le
        # conferme di Fase 4 partecipano come modelli aggiuntivi (run di
        # allineamento) e i rifiuti bloccano le conversioni.
        cfg_som = cfg.get("somiglianza", {}) or {}
        modelli = list(cfg_som.get("modelli") or [])
        conferme = list(cfg_som.get("conferme") or [])
        rifiuti = list(cfg_som.get("rifiuti") or [])
        if conferme:
            from .somiglianza import modelli_da_conferme

            modelli += modelli_da_conferme(conferme)
        if modelli:
            from .somiglianza import applica_modelli

            applica_modelli(
                modelli, istanze, etichette_r, cfg_som, rifiuti=rifiuti
            )

        # 5d. pin delle decisioni dell'utente: DOPO regole, contesto e
        # modelli, perche' l'utente ha l'ultima parola su tutto (le istanze
        # eliminate diventano WASTE, le confermate tengono la loro classe)
        if conferme or rifiuti:
            from .somiglianza import applica_pin

            applica_pin(conferme, istanze, etichette_r, rifiuti=rifiuti)

    # 6. fusione opzionale con deep learning --------------------------------
    cfg_dl = cfg.get("dl", {})
    percorso_pesi = cfg_dl.get("pesi")
    if percorso_pesi:
        try:
            from .dl.inferenza import segmenta_dl
            from .fusione import fondi_etichette

            etichette_dl = segmenta_dl(punti_r, percorso_pesi, cfg_dl)
            etichette_r = fondi_etichette(etichette_r, etichette_dl)
            log.info("Fusione con le predizioni della rete completata")
        except ImportError as exc:
            log.warning("Modulo DL non disponibile (%s): salto la fusione", exc)

    # 7. propagazione a piena risoluzione ------------------------------------
    etichette = propaga_etichette(punti_r, etichette_r, nuvola.punti)

    return RisultatoSegmentazione(
        etichette=etichette,
        etichette_ridotte=etichette_r,
        punti_ridotti=punti_r,
        istanze=istanze,
        quota_soffitto=strutture.quota_soffitto,
        quota_pavimento=strutture.quota_pavimento,
        piani_struttura=strutture.piani,
        famiglie_superfici=famiglie_superfici,
    )
