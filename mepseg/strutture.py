"""Rilevamento e rimozione degli elementi strutturali (pavimenti, soffitti, pareti).

Estrazione iterativa di piani con RANSAC (Open3D), poi analisi PER
COMPONENTI CONNESSE: in una nuvola multi-locale un singolo piano RANSAC
raccoglie superfici coplanari di piu' stanze (i soffitti di tutto un
livello, i fondi di condotte parallele, i fronti di file di rack). Ogni
componente contigua viene quindi valutata separatamente con criteri
dimensionali e di coerenza delle normali.

Ogni componente orizzontale accettata registra la propria quota e il
proprio ingombro in pianta: la pipeline li usa per assegnare a ogni
cluster il soffitto/pavimento LOCALE del suo ambiente.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .classi import ClasseMEP

log = logging.getLogger("mepseg")

# codice classe per ogni tipo di componente strutturale accettata
CLASSE_PER_TIPO = {
    "pavimento": int(ClasseMEP.PAVIMENTO),
    "soffitto": int(ClasseMEP.SOFFITTO),
    "parete": int(ClasseMEP.PARETE),
    "pilastro": int(ClasseMEP.PILASTRO),
    "trave": int(ClasseMEP.TRAVE),
}


def _rileva_travi(
    punti: np.ndarray,
    maschera_struttura: np.ndarray,
    etichette_struttura: np.ndarray,
    orizzontali: list,
    cfg: dict,
) -> np.ndarray:
    """Travi di sostegno: bande vuote e allungate che interrompono il soffitto.

    Il corpo della trave e' quasi sempre in ombra per lo scanner (coperto
    dagli impianti o troppo radente), quindi nella nuvola la trave appare
    come un CORRIDOIO VUOTO nel soffitto, largo e regolare — la firma
    complementare a quella dei pilastri (che sono presenza continua, qui
    e' assenza a forma di striscia). I punti non strutturali che pendono
    dentro il corridoio (fazzoletti di intradosso, staffe, frange) oggi
    finiscono in MEP generico: sono la trave. Ritorna la loro maschera.
    """
    from scipy import ndimage

    cella = float(cfg.get("cella_trave", 0.15))
    largh_min = float(cfg.get("larghezza_min_trave", 0.2))
    largh_max = float(cfg.get("larghezza_max_trave", 1.0))
    lung_min = float(cfg.get("lunghezza_min_trave", 1.2))
    allung_min = float(cfg.get("allungamento_min_trave", 2.5))
    profondita = float(cfg.get("profondita_trave", 0.6))

    esito = np.zeros(len(punti), dtype=bool)
    soffitto_mask = etichette_struttura == int(ClasseMEP.SOFFITTO)
    if not soffitto_mask.any():
        return esito

    # livelli: quote dei soffitti accettati, fuse se vicine
    quote = []
    for o in sorted(
        (o for o in orizzontali if o.tipo == "soffitto"), key=lambda o: o.quota
    ):
        if not quote or o.quota - quote[-1][-1] > 0.5:
            quote.append([o.quota])
        else:
            quote[-1].append(o.quota)
    livelli = [float(np.mean(q)) for q in quote]

    p_soff = punti[soffitto_mask]
    idx_liberi = np.where(~maschera_struttura)[0]
    p_liberi = punti[idx_liberi]

    for quota in livelli:
        m_liv = np.abs(p_soff[:, 2] - quota) <= 0.3
        if m_liv.sum() < 500:
            continue
        xy = p_soff[m_liv][:, :2]
        origine = xy.min(axis=0)
        ij = np.floor((xy - origine) / cella).astype(np.int64)
        forma = ij.max(axis=0) + 1
        if forma[0] * forma[1] > 50_000_000:
            continue
        griglia = np.zeros(forma, dtype=bool)
        griglia[ij[:, 0], ij[:, 1]] = True
        # chiusura morfologica: i micro-buchi da campionamento (1 cella)
        # non sono travi; un corridoio vero e' largo almeno 2 celle
        griglia = ndimage.binary_closing(
            griglia, structure=np.ones((3, 3), bool)
        )
        # buchi interni alla copertura del soffitto di questo livello
        pieni = ndimage.binary_fill_holes(griglia)
        buchi = pieni & ~griglia
        etich, n_comp = ndimage.label(buchi, structure=np.ones((3, 3), int))
        if n_comp == 0:
            continue
        ok = np.zeros(n_comp + 1, dtype=bool)
        for c, sl in enumerate(ndimage.find_objects(etich), start=1):
            if sl is None:
                continue
            lati = sorted(
                ((sl[0].stop - sl[0].start) * cella,
                 (sl[1].stop - sl[1].start) * cella)
            )
            # striscia: stretta, lunga, allungata (un vano/cavedio compatto
            # o un'ombra tonda non sono travi)
            ok[c] = (
                largh_min <= lati[0] <= largh_max
                and lati[1] >= lung_min
                and lati[1] >= allung_min * lati[0]
            )
        if not ok.any():
            continue
        corridoio = ok[etich]
        # margine di un paio di celle: le frange della trave sbordano
        corridoio = ndimage.binary_dilation(
            corridoio, structure=np.ones((3, 3), bool), iterations=1
        )
        # punti liberi che pendono nel corridoio, poco sotto il soffitto
        m_z = (p_liberi[:, 2] >= quota - profondita) & (
            p_liberi[:, 2] <= quota + 0.2
        )
        if not m_z.any():
            continue
        ij_l = np.floor((p_liberi[m_z][:, :2] - origine) / cella).astype(np.int64)
        dentro = (
            (ij_l[:, 0] >= 0) & (ij_l[:, 0] < forma[0])
            & (ij_l[:, 1] >= 0) & (ij_l[:, 1] < forma[1])
        )
        colpiti = np.zeros(int(m_z.sum()), dtype=bool)
        colpiti[dentro] = corridoio[ij_l[dentro, 0], ij_l[dentro, 1]]
        esito[idx_liberi[np.where(m_z)[0][colpiti]]] = True
    return esito


def _rileva_pilastri(
    punti: np.ndarray,
    maschera_struttura: np.ndarray,
    etichette_struttura: np.ndarray,
    orizzontali: list,
    quota_soffitto: float,
    quota_pavimento: float,
    cfg: dict,
) -> np.ndarray:
    """Individua le colonne strutturali sui punti non ancora assegnati.

    Un pilastro occupa la sua impronta in pianta CON CONTINUITA' dal
    pavimento al soffitto: nessun oggetto MEP o arredo lo fa (i rack si
    fermano sotto il soffitto, le calate non arrivano a terra). Rilevarlo
    qui, prima del clustering, evita che venga saldato nel mega-cluster
    del locale. Interamente vettorializzato: nessun ciclo sulle celle.
    Ritorna la maschera bool dei punti pilastro.
    """
    from scipy import ndimage

    cella = float(cfg.get("cella_pilastro", 0.15))
    copertura_min = float(cfg.get("copertura_min_pilastro", 0.85))
    lato_min = float(cfg.get("lato_min_pilastro", 0.2))
    lato_max = float(cfg.get("lato_max_pilastro", 1.2))
    allungamento_max = float(cfg.get("allungamento_max_pilastro", 2.5))
    passo_z = 0.25

    idx_attivi = np.where(~maschera_struttura)[0]
    esito = np.zeros(len(punti), dtype=bool)
    if len(idx_attivi) == 0:
        return esito
    p = punti[idx_attivi]

    origine_xy = p[:, :2].min(axis=0)
    ij = np.floor((p[:, :2] - origine_xy) / cella).astype(np.int64)
    forma = ij.max(axis=0) + 1
    if forma[0] * forma[1] > 50_000_000:  # scena enorme: griglia irragionevole
        return esito
    n_celle = int(forma[0] * forma[1])
    id_cella = ij[:, 0] * int(forma[1]) + ij[:, 1]

    # quota media e conteggio per cella (group-by numpy)
    conteggio = np.bincount(id_cella, minlength=n_celle)
    z_media = np.zeros(n_celle)
    occupate_xy = conteggio > 0
    z_media[occupate_xy] = (
        np.bincount(id_cella, weights=p[:, 2], minlength=n_celle)[occupate_xy]
        / conteggio[occupate_xy]
    )

    # soffitto/pavimento locali per cella: per ogni piano orizzontale
    # accettato si aggiorna in blocco la fascia di celle nel suo bbox
    centri_i = (np.arange(forma[0]) + 0.5) * cella + origine_xy[0]
    centri_j = (np.arange(forma[1]) + 0.5) * cella + origine_xy[1]
    soff = np.full(n_celle, quota_soffitto)
    pav = np.full(n_celle, quota_pavimento)
    for o in orizzontali:
        x0, y0, x1, y1 = o.bbox_xy
        mi = (centri_i >= x0 - 0.5) & (centri_i <= x1 + 0.5)
        mj = (centri_j >= y0 - 0.5) & (centri_j <= y1 + 0.5)
        m = (np.outer(mi, mj)).ravel()
        if o.tipo == "soffitto":
            m &= (o.quota > z_media + 0.3) & (o.quota < soff)
            soff[m] = o.quota
        elif o.tipo == "pavimento":
            m &= (o.quota < z_media - 0.3) & (o.quota > pav)
            pav[m] = o.quota

    z0 = pav + 0.3
    z1 = soff - 0.3
    altezza = z1 - z0

    # copertura verticale: fette da 25 cm occupate tra z0 e z1 della cella
    z0_punto = z0[id_cella]
    dentro = (p[:, 2] >= z0_punto) & (p[:, 2] <= z1[id_cella])
    fetta = ((p[:, 2] - z0_punto) / passo_z).astype(np.int64)
    n_fette_max = int(np.ceil(altezza.max() / passo_z)) + 1 if len(altezza) else 1
    chiavi_cf = id_cella[dentro] * n_fette_max + fetta[dentro]
    celle_delle_fette = np.unique(chiavi_cf) // n_fette_max
    fette_occupate = np.bincount(celle_delle_fette, minlength=n_celle)

    n_fette = np.maximum(np.ceil(altezza / passo_z), 1)
    valida = (
        (conteggio >= 10)
        & (altezza >= 1.5)
        & (fette_occupate / n_fette >= copertura_min)
    )
    griglia = valida.reshape(forma)

    componenti, n_comp = ndimage.label(griglia, structure=np.ones((3, 3), int))
    if n_comp == 0:
        return esito

    # veto anti-tramezzo: una parete accettata vicina in pianta suggerisce
    # che la "colonna" e' solo un residuo di muro (porte e occlusioni la
    # spezzano in frammenti dal footprint compatto)
    xy_pareti = punti[etichette_struttura == int(ClasseMEP.PARETE)][:, :2]
    albero_pareti = None
    if len(xy_pareti) > 0:
        from scipy.spatial import cKDTree

        albero_pareti = cKDTree(xy_pareti[:: max(1, len(xy_pareti) // 200_000)])

    id_cella_punto = componenti[ij[:, 0], ij[:, 1]]
    for c in range(1, n_comp + 1):
        celle_c = np.argwhere(componenti == c)
        lati = np.sort((celle_c.max(axis=0) - celle_c.min(axis=0) + 1) * cella)
        if not (lato_min <= lati[1] <= lato_max):
            continue
        if lati[1] > allungamento_max * max(lati[0], cella):
            continue
        if albero_pareti is not None:
            centri_c = (celle_c + 0.5) * cella + origine_xy
            vicina = albero_pareti.query(centri_c, k=1)[0] < 0.3
            if vicina.mean() > 0.5:
                continue
        esito[idx_attivi[id_cella_punto == c]] = True
    return esito


def _sporgenze_sotto_soffitto(
    xy: np.ndarray, z: np.ndarray, cfg: dict
) -> np.ndarray:
    """Punti di un soffitto accettato che sporgono verso il basso.

    La soglia RANSAC (2 cm) fa inglobare al piano del soffitto le plafoniere
    a filo o incassate, che stanno qualche centimetro SOTTO la superficie
    dominante. Si mappa la quota media per cella in pianta: le celle
    nettamente piu' basse della quota dominante, se formano componenti
    COMPATTE (una plafoniera, non un'ala di controsoffitto ribassato),
    vengono restituite al pool MEP. Ritorna la maschera bool dei punti da
    rilasciare.
    """
    from scipy import ndimage

    cella = 0.15
    sporgenza_min = float(cfg.get("sporgenza_min_soffitto", 0.03))
    lato_max = float(cfg.get("lato_max_sporgenza", 3.0))

    ij = np.floor(xy / cella).astype(np.int64)
    ij -= ij.min(axis=0)
    forma = ij.max(axis=0) + 1
    if forma[0] * forma[1] > 50_000_000:
        return np.zeros(len(z), dtype=bool)
    n_celle = int(forma[0] * forma[1])
    id_cella = ij[:, 0] * int(forma[1]) + ij[:, 1]

    conte = np.bincount(id_cella, minlength=n_celle)
    z_somma = np.bincount(id_cella, weights=z, minlength=n_celle)
    occupate = conte > 0
    z_cella = np.zeros(n_celle)
    z_cella[occupate] = z_somma[occupate] / conte[occupate]

    # quota dominante LOCALE del soffitto: massimo in una finestra mobile
    # (~2 m) sulla griglia. Il percentile globale falliva nelle ali dove
    # il soffitto sta tutto a una quota diversa dal resto della componente:
    # li' le plafoniere non "sporgevano" mai rispetto alla quota globale.
    finestra = max(
        3, int(round(float(cfg.get("finestra_quota_locale", 2.0)) / cella))
    )
    griglia_z = np.where(occupate, z_cella, -np.inf).reshape(forma)
    dominante_loc = ndimage.maximum_filter(griglia_z, size=finestra).ravel()
    bassa = occupate & (z_cella <= dominante_loc - sporgenza_min)
    if not bassa.any():
        return np.zeros(len(z), dtype=bool)

    etichette, n_comp = ndimage.label(
        bassa.reshape(forma), structure=np.ones((3, 3), int)
    )
    ok = np.zeros(n_comp + 1, dtype=bool)
    for c, sl in enumerate(ndimage.find_objects(etichette), start=1):
        if sl is None:
            continue
        lati = (
            (sl[0].stop - sl[0].start) * cella,
            (sl[1].stop - sl[1].start) * cella,
        )
        # compatta: una lastra ribassata grande (controsoffitto) resta soffitto
        ok[c] = max(lati) <= lato_max
    griglia_rilascio = ok[etichette]
    # dilatazione: l'apparecchio esce INTERO, non a cocci — il bordo della
    # plafoniera sfuma verso la quota del soffitto e senza dilatazione
    # resterebbe assorbito, lasciando frammenti sotto le soglie di evidenza
    dilata = int(cfg.get("dilata_sporgenze", 1))
    if dilata > 0 and griglia_rilascio.any():
        griglia_rilascio = ndimage.binary_dilation(
            griglia_rilascio, structure=np.ones((3, 3), bool), iterations=dilata
        )
    cella_da_rilasciare = griglia_rilascio.ravel()
    return cella_da_rilasciare[id_cella]


@dataclass
class PianoOrizzontale:
    quota: float
    bbox_xy: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    tipo: str                                   # "soffitto" | "pavimento"
    n_punti: int


@dataclass
class RisultatoStrutture:
    maschera_struttura: np.ndarray  # bool (N,) True = punto strutturale
    quota_soffitto: float           # globale (fallback)
    quota_pavimento: float
    piani: list[dict]               # metadati delle componenti accettate
    orizzontali: list[PianoOrizzontale] = field(default_factory=list)
    # (N,) codice ClasseMEP per punto: PAVIMENTO/SOFFITTO/PARETE dove
    # maschera_struttura e' True, 0 altrove
    etichette_struttura: np.ndarray = None

    @property
    def quote_orizzontali(self) -> list[float]:
        return sorted(p.quota for p in self.orizzontali)


@dataclass
class MappaPavimenti:
    """Elevation map dei pavimenti OSSERVATI, per livello (Fase 0).

    Invece di "quota pavimento del locale = un numero", una griglia 2D
    (x, y) -> quota del pavimento realmente visto in quella colonna,
    riempita per prossimita' dalle celle coperte (entro un raggio: oltre,
    nessun riferimento e nessuna decisione). Tenuta PER LIVELLO perche' in
    un edificio multi-piano le quote di piani diversi non vanno mischiate.
    """

    origine: np.ndarray             # (2,) angolo min in pianta
    cella: float
    forma: tuple[int, int]
    # dal livello piu' basso: {"quota": mediana, "mappa": (H, W) float,
    # NaN dove il livello non ha copertura entro il raggio}
    livelli: list[dict] = field(default_factory=list)

    def _indici(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((xy - self.origine) / self.cella).astype(np.int64)
        return (
            np.clip(ij[:, 0], 0, self.forma[0] - 1),
            np.clip(ij[:, 1], 0, self.forma[1] - 1),
        )

    def quota_minima_colonna(self, xy: np.ndarray) -> np.ndarray:
        """Quota del pavimento piu' BASSO osservato nella colonna di ogni
        punto (NaN dove nessun livello ha copertura): sotto questa quota
        non esiste nulla di reale, solo immagini speculari."""
        i, j = self._indici(xy)
        quote = np.full(len(xy), np.nan)
        for livello in self.livelli:  # dal basso: il primo valido vince
            valori = livello["mappa"][i, j]
            prendi = np.isnan(quote) & ~np.isnan(valori)
            quote[prendi] = valori[prendi]
        return quote

    def pavimento_locale(self, x: float, y: float, z_massimo: float) -> float | None:
        """Pavimento di riferimento per un cluster: il livello PIU' ALTO il
        cui pavimento, in quella colonna, sta sotto il cluster. E' l'eredita'
        dalle zone certe: dove il pavimento del locale e' occluso (sale
        rack) risponde la mappa costruita dove e' stato davvero visto."""
        i, j = self._indici(np.array([[x, y]]))
        for livello in reversed(self.livelli):
            valore = float(livello["mappa"][i[0], j[0]])
            if not np.isnan(valore) and valore <= z_massimo:
                return valore
        return None


def costruisci_mappa_pavimenti(
    punti: np.ndarray, etichette: np.ndarray, cfg: dict
) -> MappaPavimenti | None:
    """Costruisce la :class:`MappaPavimenti` dai punti gia' etichettati
    PAVIMENTO (Fase 0 architettonica, dopo l'accettazione dei piani)."""
    from scipy import ndimage

    from .classi import ClasseMEP

    cella = float(cfg.get("cella_mappa", 0.5))
    raggio = float(cfg.get("raggio_riempimento_mappa", 8.0))
    fusione = float(cfg.get("fusione_livelli_mappa", 0.5))

    maschera_pav = etichette == int(ClasseMEP.PAVIMENTO)
    if not maschera_pav.any():
        return None
    origine = punti[:, :2].min(axis=0)
    forma_arr = (
        np.floor((punti[:, :2].max(axis=0) - origine) / cella).astype(np.int64) + 1
    )
    if int(forma_arr[0]) * int(forma_arr[1]) > 30_000_000:
        return None  # estensione irragionevole: niente mappa
    forma = (int(forma_arr[0]), int(forma_arr[1]))

    z_pav = punti[maschera_pav, 2]
    xy_pav = punti[maschera_pav, :2]
    ij = np.floor((xy_pav - origine) / cella).astype(np.int64)

    # livelli: quote dei pavimenti fuse dove vicine (< fusione), separate
    # ai salti — piani diversi dell'edificio
    ordinate = np.sort(z_pav)
    tagli = np.where(np.diff(ordinate) > fusione)[0]
    confini = (
        [float(ordinate[0]) - 1e-6]
        + [float(ordinate[t] + ordinate[t + 1]) / 2 for t in tagli]
        + [float(ordinate[-1]) + 1e-6]
    )

    livelli = []
    for basso, alto in zip(confini[:-1], confini[1:]):
        del_livello = (z_pav > basso) & (z_pav <= alto)
        if int(del_livello.sum()) < 30:
            continue
        mappa = np.full(forma, np.inf)
        np.minimum.at(
            mappa, (ij[del_livello, 0], ij[del_livello, 1]), z_pav[del_livello]
        )
        coperta = np.isfinite(mappa)
        dist, (ii, jj) = ndimage.distance_transform_edt(
            ~coperta, return_indices=True
        )
        piena = mappa[ii, jj]
        piena[dist > raggio / cella] = np.nan
        livelli.append(
            {"quota": float(np.median(z_pav[del_livello])), "mappa": piena}
        )
    if not livelli:
        return None
    log.info(
        "Mappa pavimenti: %d livelli (quote %s), cella %.2f m",
        len(livelli),
        ", ".join(f"{lv['quota']:.2f}" for lv in livelli),
        cella,
    )
    return MappaPavimenti(
        origine=origine, cella=cella, forma=forma, livelli=livelli
    )


def _componenti_su_griglia(
    proiezioni: np.ndarray, cella: float
) -> tuple[np.ndarray, int]:
    """Componenti connesse (8-vicinato) su griglia 2D delle proiezioni.

    Ritorna (id_componente per punto (1..n), numero componenti).
    """
    from scipy import ndimage

    ij = np.floor(proiezioni / cella).astype(np.int64)
    ij -= ij.min(axis=0)
    forma = ij.max(axis=0) + 1
    griglia = np.zeros(forma, dtype=bool)
    griglia[ij[:, 0], ij[:, 1]] = True
    etichette, n = ndimage.label(griglia, structure=np.ones((3, 3), dtype=int))
    return etichette[ij[:, 0], ij[:, 1]], int(n)


def rimuovi_strutture(
    punti: np.ndarray, normali: np.ndarray, cfg: dict
) -> RisultatoStrutture:
    import open3d as o3d

    n_totale = len(punti)
    soglia_dist = float(cfg.get("soglia_distanza_piano", 0.02))
    # soglia basata sull'AREA fisica, non sulla frazione dei punti totali:
    # in una nuvola multi-locale il soffitto di una stanza e' una frazione
    # piccola del totale ma la sua area resta significativa.
    voxel_eff = max(float(cfg.get("voxel_eff", 0.03)), 0.005)
    min_punti_piano = max(
        int(cfg.get("min_punti_piano_min", 1500)),
        int(float(cfg.get("min_area_piano", 2.0)) / voxel_eff**2),
    )
    max_piani = int(cfg.get("max_piani", 60))
    min_lato_orizz = float(cfg.get("min_lato_orizzontale", 1.5))
    min_area_piano = float(cfg.get("min_area_piano", 2.0))
    min_altezza_parete = float(cfg.get("min_altezza_parete", 1.8))
    min_area_parete = float(cfg.get("min_area_parete", 3.0))
    min_coerenza = float(cfg.get("min_coerenza_normali", 0.70))
    min_copertura_parete = float(cfg.get("min_copertura_parete", 0.35))
    cella = float(cfg.get("cella_componenti", 0.30))
    tolleranza_quota = float(cfg.get("tolleranza_incontro_quota", 0.6))

    z_min_scena = float(np.percentile(punti[:, 2], 2))
    z_max_scena = float(np.percentile(punti[:, 2], 98))
    z_medio = 0.5 * (z_min_scena + z_max_scena)

    # campione della nuvola per decidere il TIPO delle superfici orizzontali
    # con l'evidenza LOCALE (cosa c'e' appena sopra/sotto), non con la meta'
    # dell'intervallo z della scena: se il ritaglio include il tetto o
    # rumore alto, la meta' scena sale e tutti i soffitti interni
    # diventerebbero "pavimenti" (successo sul run FiberCop 1 intero:
    # 0 soffitti accettati, regole a valle azzerate)
    if n_totale > 1_500_000:
        rng_tipo = np.random.default_rng(0)
        punti_camp = punti[rng_tipo.choice(n_totale, 1_500_000, replace=False)]
    else:
        punti_camp = punti

    def _tipo_orizzontale(comp: np.ndarray, quota: float) -> str:
        """'pavimento' o 'soffitto' dall'evidenza locale: gli oggetti
        APPOGGIANO sui pavimenti (banda piena appena sopra, vuota appena
        sotto: li' c'e' la soletta) e PENDONO dai soffitti (viceversa).
        La banda stretta (8-20 cm) non scavalca l'intercapedine del solaio,
        cosi' la superficie gemella dall'altro lato non inquina il conto.
        A evidenza pari o scarsa, ripiego sulla quota media della scena."""
        x0, y0 = float(comp[:, 0].min()), float(comp[:, 1].min())
        x1, y1 = float(comp[:, 0].max()), float(comp[:, 1].max())
        dentro = (
            (punti_camp[:, 0] >= x0) & (punti_camp[:, 0] <= x1)
            & (punti_camp[:, 1] >= y0) & (punti_camp[:, 1] <= y1)
        )
        dz = punti_camp[dentro, 2] - quota
        n_sopra = int(((dz >= 0.08) & (dz <= 0.20)).sum())
        n_sotto = int(((dz <= -0.08) & (dz >= -0.20)).sum())
        if n_sopra >= 2 * n_sotto and n_sopra > 20:
            return "pavimento"
        if n_sotto >= 2 * n_sopra and n_sotto > 20:
            return "soffitto"
        return "soffitto" if quota > z_medio else "pavimento"

    indici_attivi = np.arange(n_totale)
    maschera_struttura = np.zeros(n_totale, dtype=bool)
    etichette_struttura = np.zeros(n_totale, dtype=np.uint8)
    piani_accettati: list[dict] = []
    orizzontali: list[PianoOrizzontale] = []

    for _ in range(max_piani):
        if len(indici_attivi) < min_punti_piano:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(punti[indici_attivi])
        modello, inlier = pcd.segment_plane(
            distance_threshold=soglia_dist, ransac_n=3, num_iterations=1000
        )
        if len(inlier) < min_punti_piano:
            break

        idx_globali = indici_attivi[np.asarray(inlier)]
        p_piano = punti[idx_globali]
        normale = np.asarray(modello[:3])
        normale /= np.linalg.norm(normale)
        orizzontale = abs(normale[2]) > 0.9
        verticale = abs(normale[2]) < 0.2

        if orizzontale or verticale:
            # coerenza normali per punto (i piani "fantasma" che tagliano
            # tubi coplanari hanno normali sparse)
            allineamento = np.abs(normali[idx_globali] @ normale) > 0.85

            # assi del piano e proiezioni 2D
            centrati = p_piano - p_piano.mean(axis=0)
            cov = centrati.T @ centrati / len(p_piano)
            autoval, autovett = np.linalg.eigh(cov)
            assi = autovett[:, np.argsort(autoval)[::-1][:2]]
            proiezioni = centrati @ assi

            comp_ids, n_comp = _componenti_su_griglia(proiezioni, cella)
            for c in range(1, n_comp + 1):
                m = comp_ids == c
                n_c = int(m.sum())
                if n_c < int(cfg.get("min_punti_componente", 200)):
                    continue
                coerenza_c = float(allineamento[m].mean())
                if coerenza_c < min_coerenza:
                    continue
                proi_c = proiezioni[m]
                lati = proi_c.max(axis=0) - proi_c.min(axis=0)
                celle_c = len(
                    np.unique(
                        np.floor(proi_c[:, 0] / cella).astype(np.int64) * 1_000_003
                        + np.floor(proi_c[:, 1] / cella).astype(np.int64)
                    )
                )
                area_c = celle_c * cella**2
                z_c = p_piano[m][:, 2]
                quota_c = float(z_c.mean())
                accettata, tipo = False, ""

                if orizzontale:
                    if min(lati) >= min_lato_orizz and area_c >= min_area_piano:
                        tipo = _tipo_orizzontale(p_piano[m], quota_c)
                        accettata = True
                elif verticale:
                    estensione_z = float(z_c.max() - z_c.min())
                    if estensione_z >= min_altezza_parete and area_c >= min_area_parete:
                        # una parete vera "incontra" un pavimento in basso e
                        # un soffitto in alto NEL SUO INTORNO in pianta: i
                        # fronti di una fila di rack non arrivano al soffitto.
                        xy_c = p_piano[m][:, :2]
                        bbox_c = (
                            xy_c[:, 0].min() - 1.0, xy_c[:, 1].min() - 1.0,
                            xy_c[:, 0].max() + 1.0, xy_c[:, 1].max() + 1.0,
                        )
                        vicine = [
                            o for o in orizzontali
                            if not (
                                o.bbox_xy[2] < bbox_c[0] or o.bbox_xy[0] > bbox_c[2]
                                or o.bbox_xy[3] < bbox_c[1] or o.bbox_xy[1] > bbox_c[3]
                            )
                        ]
                        if vicine:
                            base_ok = any(
                                abs(float(z_c.min()) - o.quota) <= tolleranza_quota
                                for o in vicine
                            )
                            cima_ok = any(
                                abs(float(z_c.max()) - o.quota)
                                <= tolleranza_quota + 0.2
                                for o in vicine
                            )
                            # parete occlusa in alto (ombre di armadi/rack):
                            # accettata se copre comunque gran parte
                            # dell'altezza del locale
                            soffitti_vicini = [
                                o.quota for o in vicine if o.quota > float(z_c.min())
                            ]
                            pavimenti_vicini = [
                                o.quota for o in vicine if o.quota < float(z_c.max())
                            ]
                            copre_altezza = False
                            if soffitti_vicini and pavimenti_vicini:
                                altezza_locale = max(soffitti_vicini) - min(
                                    pavimenti_vicini
                                )
                                copre_altezza = (
                                    estensione_z >= 0.75 * altezza_locale
                                )
                            accettata = base_ok and (cima_ok or copre_altezza)
                        else:
                            # nessuna quota nota nell'intorno: fallback
                            # conservativo, la parete deve attraversare
                            # l'intera altezza della scena ed essere densa
                            copertura_c = area_c / max(lati[0] * lati[1], 1e-9)
                            accettata = (
                                float(z_c.min()) <= z_min_scena + 0.8
                                and float(z_c.max()) >= z_max_scena - 0.8
                                and copertura_c >= min_copertura_parete
                            )
                        tipo = "parete"

                if accettata:
                    idx_comp = idx_globali[m]
                    # anti-assorbimento: le plafoniere a filo inglobate dal
                    # piano soffitto tornano al pool MEP
                    if tipo == "soffitto" and cfg.get("estrai_sporgenze", True):
                        rilascio = _sporgenze_sotto_soffitto(
                            p_piano[m][:, :2], z_c, cfg
                        )
                        n_ril = int(rilascio.sum())
                        if 0 < n_ril < 0.4 * len(idx_comp):
                            idx_comp = idx_comp[~rilascio]
                    maschera_struttura[idx_comp] = True
                    etichette_struttura[idx_comp] = CLASSE_PER_TIPO[tipo]
                    if tipo in ("soffitto", "pavimento"):
                        xy_c = p_piano[m][:, :2]
                        orizzontali.append(
                            PianoOrizzontale(
                                quota=quota_c,
                                bbox_xy=(
                                    float(xy_c[:, 0].min()), float(xy_c[:, 1].min()),
                                    float(xy_c[:, 0].max()), float(xy_c[:, 1].max()),
                                ),
                                tipo=tipo,
                                n_punti=n_c,
                            )
                        )
                    p_c = p_piano[m]
                    piani_accettati.append(
                        {
                            "tipo": tipo,
                            "normale": normale.tolist(),
                            "quota_media": quota_c,
                            "n_punti": n_c,
                            "estensione": [float(lati[0]), float(lati[1])],
                            "coerenza_normali": round(coerenza_c, 3),
                            "area": round(area_c, 2),
                            # geometria completa per il recupero dei residui
                            # coplanari in pipeline (finestre, pezzi di muro)
                            "d": float(-normale @ p_c.mean(axis=0)),
                            "bbox": [
                                *p_c.min(axis=0).tolist(),
                                *p_c.max(axis=0).tolist(),
                            ],
                        }
                    )

        # il piano viene comunque tolto dal pool di ricerca (anche le parti
        # non accettate) per non ritrovarlo al giro successivo; i punti non
        # accettati restano non-struttura e verranno classificati dopo.
        maschera_rimasti = np.ones(len(indici_attivi), dtype=bool)
        maschera_rimasti[np.asarray(inlier)] = False
        indici_attivi = indici_attivi[maschera_rimasti]

    quote_soffitti = [o.quota for o in orizzontali if o.tipo == "soffitto"]
    quote_pavimenti = [o.quota for o in orizzontali if o.tipo == "pavimento"]
    quota_soffitto = max(quote_soffitti) if quote_soffitti else float(
        np.percentile(punti[:, 2], 99.5)
    )
    quota_pavimento = min(quote_pavimenti) if quote_pavimenti else float(
        np.percentile(punti[:, 2], 0.5)
    )

    # pilastri: colonne che attraversano il locale da pavimento a soffitto,
    # rilevate PRIMA del clustering (altrimenti finiscono saldate nei
    # cluster degli oggetti adiacenti)
    maschera_pilastri = _rileva_pilastri(
        punti, maschera_struttura, etichette_struttura, orizzontali,
        quota_soffitto, quota_pavimento, cfg,
    )
    if maschera_pilastri.any():
        maschera_struttura |= maschera_pilastri
        etichette_struttura[maschera_pilastri] = CLASSE_PER_TIPO["pilastro"]

    # travi: corridoi vuoti a striscia nei soffitti accettati; i punti che
    # vi pendono dentro sono struttura, non impianti (e sottrarli ai cluster
    # MEP evita che facciano da ponte tra oggetti di sale diverse)
    if cfg.get("rileva_travi", True):
        maschera_travi = _rileva_travi(
            punti, maschera_struttura, etichette_struttura, orizzontali, cfg
        )
        if maschera_travi.any():
            maschera_struttura |= maschera_travi
            etichette_struttura[maschera_travi] = CLASSE_PER_TIPO["trave"]

    return RisultatoStrutture(
        maschera_struttura,
        quota_soffitto,
        quota_pavimento,
        piani_accettati,
        orizzontali,
        etichette_struttura,
    )
