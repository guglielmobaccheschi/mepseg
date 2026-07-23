"""Regole di contesto: coerenza topologica degli impianti.

Gli impianti sono una rete, non oggetti isolati:

1. **Luci seriali**: i corpi illuminanti tubolari (neon) sono geometricamente
   identici a tubazioni corte, ma si presentano in serie di elementi simili,
   paralleli, vicini al soffitto. Una "fila di tubazioni corte" e' una fila
   di luci.
2. **Ancoraggio**: una tubazione o condotta reale prosegue in un altro
   segmento, tocca un'apparecchiatura o penetra una struttura. Un cilindro
   "fluttuante" con entrambe le estremita' nel vuoto non e' un impianto
   (es. armadietti con ante aperte, arredi curvi) e viene declassato ad
   Arredo/non-MEP.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from .classi import ClasseMEP

log = logging.getLogger("mepseg")

# classi i cui punti costituiscono un ancoraggio valido per un'estremita'
CLASSI_ANCORA = {
    int(ClasseMEP.STRUTTURA),
    int(ClasseMEP.PAVIMENTO),
    int(ClasseMEP.SOFFITTO),
    int(ClasseMEP.PARETE),
    int(ClasseMEP.PILASTRO),
    int(ClasseMEP.APPARECCHIATURA),
    int(ClasseMEP.TUBAZIONE),
    int(ClasseMEP.CONDOTTA_CIRCOLARE),
    int(ClasseMEP.CONDOTTA_RETTANGOLARE),
    int(ClasseMEP.PASSERELLA_CAVI),
    int(ClasseMEP.TERMINALE_ARIA),
}

# ancore valide per l'estremita' ALTA di un elemento verticale: un montante
# vero penetra il soffitto o prosegue in altri impianti. Il pavimento non
# puo' ancorare una cima, e l'apparecchiatura nemmeno: rack e armadi stanno
# appoggiati sul pavimento senza toccare il soffitto, e qualunque oggetto
# posato SOPRA un arredo ne "ancorerebbe" la cima per sbaglio.
CLASSI_ANCORA_ALTA = {
    int(ClasseMEP.SOFFITTO),
    int(ClasseMEP.PARETE),
    int(ClasseMEP.PILASTRO),
    int(ClasseMEP.TUBAZIONE),
    int(ClasseMEP.CONDOTTA_CIRCOLARE),
    int(ClasseMEP.CONDOTTA_RETTANGOLARE),
    int(ClasseMEP.PASSERELLA_CAVI),
    int(ClasseMEP.TERMINALE_ARIA),
}

CLASSI_DA_ANCORARE = {
    ClasseMEP.TUBAZIONE,
    ClasseMEP.CONDOTTA_CIRCOLARE,
    ClasseMEP.CONDOTTA_RETTANGOLARE,
}


def _asse_principale(punti: np.ndarray) -> np.ndarray:
    centrati = punti - punti.mean(axis=0)
    cov = centrati.T @ centrati / len(punti)
    autoval, autovett = np.linalg.eigh(cov)
    return autovett[:, np.argmax(autoval)]


# classi che possono essere riclassificate come luce se in serie; le LUCI
# gia' classificate contano come membri della fila ma non vengono toccate
CLASSI_CANDIDATE_NEON = {ClasseMEP.TUBAZIONE, ClasseMEP.MEP_GENERICO}


def riclassifica_luci_seriali(
    istanze: list, punti_r: np.ndarray, etichette_r: np.ndarray, cfg: dict
) -> int:
    """Converte in LUCE le file di elementi lineari corti, paralleli e simili.

    La firma delle luci e' la RIPETIZIONE, non la forma esatta: i neon nudi
    sono cilindri sottili, ma le lampade lineari a sospensione hanno un corpo
    scatolare a sezione quasi quadrata che non fitta ne' il cilindro ne' il
    "piatto". Candidati: tubazioni e MEP generici corti, a sezione contenuta,
    orizzontali e vicini al soffitto. Le luci gia' riconosciute dalle regole
    per-cluster votano per il minimo di ripetizioni senza essere toccate.

    Ritorna il numero di istanze riclassificate.
    """
    raggio_max = float(cfg.get("raggio_max_neon", 0.10))
    sezione_max = float(cfg.get("sezione_max_neon", 0.35))
    lunghezza_max = float(cfg.get("lunghezza_max_neon", 2.5))
    lunghezza_min = float(cfg.get("lunghezza_min_neon", 0.3))
    dist_soffitto_max = float(cfg.get("dist_soffitto_neon", 1.2))
    dist_max = float(cfg.get("dist_max_neon", 6.0))
    min_ripetizioni = int(cfg.get("min_ripetizioni", 3))

    def descrittore(ist) -> tuple[np.ndarray, float] | None:
        """(asse, lunghezza) se l'istanza ha la geometria di una luce
        lineare, altrimenti None."""
        f = ist.feature
        cil = f.cilindro
        if cil.valido and cil.raggio <= raggio_max and cil.lunghezza > 0:
            asse, lunghezza = cil.asse, float(cil.lunghezza)
        else:
            # corpo scatolare: sezione OBB contenuta (i lati sono ordinati
            # in senso decrescente, quindi basta controllare il secondo)
            if float(f.dimensioni[1]) > sezione_max:
                return None
            asse = _asse_principale(punti_r[ist.indici])
            lunghezza = float(f.dimensioni[0])
        if not (lunghezza_min <= lunghezza <= lunghezza_max):
            return None
        if abs(float(asse[2])) > 0.35:  # una luce lineare e' orizzontale
            return None
        if f.dist_da_soffitto > dist_soffitto_max:
            return None
        # anti-riflesso: il bagliore produce nuvole rade con massa
        # distribuita in verticale. Una lampada vera (anche sospesa, anche
        # col riflesso sotto) ha il CORPO concentrato nella fascia alta.
        if f.estensione_z_robusta > float(
            cfg.get("estensione_z_max_neon", 0.30)
        ) and f.frazione_z_alta < float(cfg.get("frazione_alta_min_neon", 0.5)):
            return None
        if f.densita_superficie < float(cfg.get("densita_min_neon", 0.08)):
            return None
        return asse, lunghezza

    membri = []  # (istanza, asse, lunghezza, riclassificabile)
    for ist in istanze:
        if ist.classe in CLASSI_CANDIDATE_NEON:
            riclassificabile = True
        elif ist.classe == ClasseMEP.LUCE:
            riclassificabile = False
        else:
            continue
        d = descrittore(ist)
        if d is not None:
            membri.append((ist, d[0], d[1], riclassificabile))
    if len(membri) < min_ripetizioni:
        return 0

    # grafo di compatibilita': assi paralleli, lunghezze e quote simili,
    # membri vicini tra loro (una "fila" e' locale, non tutto l'edificio)
    n = len(membri)
    adiacenza = [[] for _ in range(n)]
    for i in range(n):
        _, asse_i, lung_i, _ = membri[i]
        centro_i = membri[i][0].feature.centro
        for j in range(i + 1, n):
            _, asse_j, lung_j, _ = membri[j]
            centro_j = membri[j][0].feature.centro
            rapporto = lung_i / max(lung_j, 1e-6)
            if (
                abs(float(asse_i @ asse_j)) > 0.9
                and 0.7 < rapporto < 1.43
                and abs(float(centro_i[2]) - float(centro_j[2])) < 0.5
                and float(np.linalg.norm(centro_i[:2] - centro_j[:2])) <= dist_max
            ):
                adiacenza[i].append(j)
                adiacenza[j].append(i)

    # componenti connesse del grafo
    visitati = [False] * n
    riclassificate = 0
    for seme in range(n):
        if visitati[seme]:
            continue
        gruppo, coda = [], [seme]
        visitati[seme] = True
        while coda:
            k = coda.pop()
            gruppo.append(k)
            for v in adiacenza[k]:
                if not visitati[v]:
                    visitati[v] = True
                    coda.append(v)
        if len(gruppo) >= min_ripetizioni:
            for k in gruppo:
                ist, _, _, riclassificabile = membri[k]
                if not riclassificabile:
                    continue  # e' gia' una luce: vota e basta
                ist.classe = ClasseMEP.LUCE
                etichette_r[ist.indici] = int(ClasseMEP.LUCE)
                riclassificate += 1
    if riclassificate:
        log.info(
            "Contesto: %d elementi lineari in serie riclassificati come luci",
            riclassificate,
        )
    return riclassificate


# classi che possono entrare in una catena assiale e riceverne la classe
CLASSI_CATENA = {
    ClasseMEP.TUBAZIONE,
    ClasseMEP.CONDOTTA_CIRCOLARE,
    ClasseMEP.CONDOTTA_RETTANGOLARE,
    ClasseMEP.PASSERELLA_CAVI,
    ClasseMEP.APPARECCHIATURA,
    ClasseMEP.MEP_GENERICO,
    ClasseMEP.NON_MEP_ARREDO,
}
# classi "informative" che possono vincere il voto di catena
CLASSI_VOTO_CATENA = {
    ClasseMEP.TUBAZIONE,
    ClasseMEP.CONDOTTA_CIRCOLARE,
    ClasseMEP.CONDOTTA_RETTANGOLARE,
    ClasseMEP.PASSERELLA_CAVI,
}


def unifica_catene(
    istanze: list, punti_r: np.ndarray, etichette_r: np.ndarray, cfg: dict
) -> int:
    """Continuita' assiale: segmenti collineari di uno stesso corri-dotto
    (blindosbarra, condotta, passerella) spezzati dal clustering votano una
    classe unica.

    Ogni segmento, classificato da solo, puo' cadere in una classe diversa
    (un pezzo denso -> apparecchiatura, uno corto -> generico, uno con le
    estremita' "nel vuoto" -> arredo, proprio perche' il taglio gli ha
    staccato la continuazione). Una catena di segmenti snelli, paralleli,
    allineati e di sezione simile e' UN oggetto lineare: vince la classe
    lineare col maggior numero di punti. Ritorna i segmenti riclassificati.
    """
    gap_max = float(cfg.get("gap_max_catena", 0.6))
    offset_max = float(cfg.get("offset_lat_catena", 0.35))
    parallelismo = float(cfg.get("parallelismo_catena", 0.9))
    rapporto_sez = float(cfg.get("rapporto_sezione_catena", 2.5))
    sezione_max = float(cfg.get("sezione_max_catena", 0.8))

    membri = []
    for ist in istanze:
        if ist.classe not in CLASSI_CATENA:
            continue
        f = ist.feature
        lung, sez = float(f.dimensioni[0]), float(f.dimensioni[1])
        # solo elementi SNELLI: i blocchi rack (sezione larga) non sono
        # segmenti di corri-dotto
        if sez > sezione_max or lung < 2.0 * sez or lung < 0.3:
            continue
        cil = f.cilindro
        asse = cil.asse if cil.valido else _asse_principale(punti_r[ist.indici])
        membri.append((ist, np.asarray(asse, float), lung, max(sez, 0.02)))
    if len(membri) < 2:
        return 0

    centri = np.array([m[0].feature.centro for m in membri])
    albero = cKDTree(centri)
    coppie = albero.query_pairs(4.0, output_type="ndarray")

    n = len(membri)
    adiacenza = [[] for _ in range(n)]
    for i, j in coppie:
        _, asse_i, lung_i, sez_i = membri[i]
        _, asse_j, lung_j, sez_j = membri[j]
        if abs(float(asse_i @ asse_j)) < parallelismo:
            continue
        if max(sez_i, sez_j) / min(sez_i, sez_j) > rapporto_sez:
            continue
        delta = centri[j] - centri[i]
        lungo = float(delta @ asse_i)
        laterale = float(np.linalg.norm(delta - lungo * asse_i))
        if laterale > offset_max:
            continue
        gap = abs(lungo) - 0.5 * (lung_i + lung_j)
        if gap > gap_max:
            continue
        adiacenza[i].append(j)
        adiacenza[j].append(i)

    visitati = [False] * n
    riclassificati = 0
    for seme in range(n):
        if visitati[seme]:
            continue
        gruppo, coda = [], [seme]
        visitati[seme] = True
        while coda:
            k = coda.pop()
            gruppo.append(k)
            for v in adiacenza[k]:
                if not visitati[v]:
                    visitati[v] = True
                    coda.append(v)
        if len(gruppo) < 2:
            continue
        voti: dict = {}
        for k in gruppo:
            ist = membri[k][0]
            if ist.classe in CLASSI_VOTO_CATENA:
                voti[ist.classe] = voti.get(ist.classe, 0) + ist.feature.n_punti
        if not voti:
            continue  # catena tutta generico/arredo: niente da propagare
        vincente = max(voti, key=voti.get)
        for k in gruppo:
            ist = membri[k][0]
            if ist.classe != vincente:
                ist.classe = vincente
                etichette_r[ist.indici] = int(vincente)
                riclassificati += 1
    if riclassificati:
        log.info(
            "Contesto: %d segmenti collineari unificati alla classe della "
            "propria catena", riclassificati,
        )
    return riclassificati


def applica_ancoraggio(
    istanze: list, punti_r: np.ndarray, etichette_r: np.ndarray, cfg: dict
) -> int:
    """Declassa ad Arredo/non-MEP i tubi/condotte con entrambe le estremita'
    nel vuoto. Ritorna il numero di istanze declassate."""
    raggio_anc = float(cfg.get("raggio_ancoraggio", 0.4))
    spessore_fetta = float(cfg.get("spessore_fetta_estremita", 0.2))
    max_campioni = int(cfg.get("campioni_per_estremita", 25))

    candidate = [ist for ist in istanze if ist.classe in CLASSI_DA_ANCORARE]
    if not candidate:
        return 0

    # id istanza per punto: per escludere l'istanza stessa dalla ricerca
    id_istanza = np.full(len(punti_r), -1, dtype=np.int32)
    for ist in istanze:
        id_istanza[ist.indici] = ist.id
    albero = cKDTree(punti_r)

    def estremita_ancorata(
        campioni: np.ndarray, proprio_id: int, raggio: float, classi_valide: set
    ) -> bool:
        if len(campioni) > max_campioni:
            passo = len(campioni) // max_campioni
            campioni = campioni[::passo]
        vicini = albero.query_ball_point(campioni, raggio, workers=-1)
        for lista in vicini:
            for v in lista:
                if id_istanza[v] != proprio_id and int(etichette_r[v]) in classi_valide:
                    return True
        return False

    declassate = 0
    for ist in candidate:
        p = punti_r[ist.indici]
        cil = ist.feature.cilindro
        asse = cil.asse if cil.valido else _asse_principale(p)
        if asse[2] < 0:
            asse = -asse  # orientato verso l'alto: "fetta alta" = cima
        # raggio di ricerca proporzionato alla sezione dell'elemento
        raggio = max(raggio_anc, 1.5 * float(ist.feature.dimensioni[2]))
        t = p @ asse
        fetta_bassa = p[t <= t.min() + spessore_fetta]
        fetta_alta = p[t >= t.max() - spessore_fetta]
        # un elemento VERTICALE (montante) penetra pavimento E soffitto:
        # un "cilindro" appoggiato a terra col vertice libero (armadietto,
        # boiler d'arredo) e' ancorato solo in basso e viene declassato.
        # Per la cima valgono solo soffitto, pareti e altri impianti: rack
        # e armadi toccano il pavimento ma mai il soffitto.
        verticale = abs(float(asse[2])) > 0.7
        ancora_bassa = estremita_ancorata(fetta_bassa, ist.id, raggio, CLASSI_ANCORA)
        ancora_alta = estremita_ancorata(
            fetta_alta, ist.id, raggio,
            CLASSI_ANCORA_ALTA if verticale else CLASSI_ANCORA,
        )
        ancorata = (
            (ancora_bassa and ancora_alta) if verticale
            else (ancora_bassa or ancora_alta)
        )
        if not ancorata:
            ist.classe = ClasseMEP.NON_MEP_ARREDO
            etichette_r[ist.indici] = int(ClasseMEP.NON_MEP_ARREDO)
            declassate += 1
    if declassate:
        log.info(
            "Contesto: %d tubi/condotte 'fluttuanti' declassati ad arredo/non-MEP",
            declassate,
        )
    return declassate
