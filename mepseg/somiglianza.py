"""Ricerca per somiglianza geometrica ("trova simili").

Gli impianti sono fatti di oggetti STANDARD ripetuti: la stessa plafoniera,
lo stesso rack, lo stesso diffusore compaiono decine di volte in un edificio.
Quando un'istanza viene confermata dall'utente, la sua firma geometrica
(dimensioni, sviluppo verticale, quota rispetto a soffitto e pavimento,
orientamento) diventa un MODELLO riutilizzabile: le istanze dubbie (MEP
generico) che combaciano col modello ereditano la sua classe.

I modelli vivono in un file JSON accanto al progetto e si accumulano nel
tempo; da CLI: ``--salva-modello ID:NOME[:CLASSE]`` per estrarre un modello
da un'istanza del run corrente, ``--modelli percorso.json`` per applicarli.

Fase 4 — conferme dell'utente (``conferme.json`` accanto all'output, da CLI
``--conferme percorso.json``). Distinzione voluta: i MODELLI sono la libreria
riusabile tra nuvole diverse, le CONFERME sono legate alla singola nuvola.
Una conferma vale tre volte: PIN dell'istanza specifica (l'utente ha l'ultima
parola su regole, contesto e modelli), modello positivo per propagare la
classe alle istanze simili nel run di allineamento, futura etichetta di
training a fiducia piena. I RIFIUTI (azione "Elimina" nella GUI) marcano il
rumore: l'istanza eliminata diventa WASTE nei run successivi e nessun
modello converte un'istanza la cui firma combacia con un rifiuto (il
rifiuto vince sempre sui modelli; solo il pin di una conferma esplicita
vince sul rifiuto).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from .classi import ClasseMEP, NOMI_CLASSI

log = logging.getLogger("mepseg")

VERSIONE_FILE = 1
VERSIONE_CONFERME = 1

# ri-identificazione della STESSA istanza fisica tra due run: il clustering
# puo' spostare qualche punto di bordo, ma centro e ingombro restano quelli
TOL_CENTRO_PIN = 0.3   # m sul centro OBB
TOL_DIM_PIN = 0.2      # relativa sui lati OBB
TOL_DIM_PIN_ABS = 0.05  # m: pavimento assoluto per gli oggetti piccoli

# classi le cui istanze possono essere riclassificate da un modello: solo
# quelle "dubbie" — mai sovrascrivere una classe decisa dalle regole
CLASSI_BERSAGLIO_DEFAULT = ["mep_generico"]


def _categoria_orientamento(verticalita: float) -> str:
    if verticalita > 0.7:
        return "verticale"
    if verticalita < 0.35:
        return "orizzontale"
    return "obliquo"


def firma_da_feature(f) -> dict:
    """Firma geometrica di un'istanza (da :class:`FeatureCluster`)."""
    return {
        "dimensioni": [round(float(v), 4) for v in f.dimensioni],
        "estensione_z": round(float(f.estensione_z_robusta), 4),
        "dist_soffitto": round(float(f.dist_da_soffitto), 4),
        "dist_pavimento": round(float(f.dist_da_pavimento), 4),
        "verticalita": round(float(f.verticalita_asse), 4),
        "centro": [round(float(v), 3) for v in f.centro],
    }


def firma_da_riga(riga: dict) -> dict:
    """Stessa firma, ma da una riga del report JSON (percorso della GUI:
    il report ha gia' tutti i campi, non serve rifare il run)."""
    return {
        "dimensioni": [riga["dim_1"], riga["dim_2"], riga["dim_3"]],
        "estensione_z": riga["estensione_z"],
        "dist_soffitto": riga["dist_soffitto"],
        "dist_pavimento": riga["dist_pavimento"],
        "verticalita": riga["verticalita"],
        "centro": [riga["centro_x"], riga["centro_y"], riga["centro_z"]],
    }


def crea_modello(istanza, nome: str, classe: ClasseMEP | None = None,
                 origine: str = "") -> dict:
    """Estrae la firma geometrica di un'istanza come modello riusabile."""
    modello = firma_da_feature(istanza.feature)
    modello.update({
        "nome": nome,
        "classe": (classe if classe is not None else istanza.classe).name.lower(),
        "origine": {
            "file": origine,
            "istanza": int(istanza.id),
            "data": date.today().isoformat(),
        },
    })
    return modello


def corrisponde(modello: dict, feature, cfg: dict) -> bool:
    """True se la firma dell'istanza combacia col modello.

    Tolleranza sulle dimensioni: relativa + un margine assoluto, perche' le
    scansioni parziali accorciano/assottigliano gli oggetti reali rispetto
    al campione confermato.
    """
    tol_rel = float(cfg.get("tolleranza_dim_rel", 0.4))
    tol_abs = float(cfg.get("tolleranza_dim_abs", 0.08))
    tol_quota = float(cfg.get("tolleranza_quota", 0.6))

    for dim_ist, dim_mod in zip(feature.dimensioni, modello["dimensioni"]):
        if abs(float(dim_ist) - dim_mod) > max(tol_rel * dim_mod, tol_abs):
            return False

    # sviluppo verticale: limite superiore (esclude lastre inclinate e aste
    # appese che condividono l'OBB col modello ma non la giacitura)
    est_max = modello["estensione_z"] + max(
        tol_rel * modello["estensione_z"], tol_abs
    )
    if float(feature.estensione_z_robusta) > est_max:
        return False

    if _categoria_orientamento(float(feature.verticalita_asse)) != (
        _categoria_orientamento(modello["verticalita"])
    ):
        return False

    # contesto in quota: un oggetto "da soffitto" va cercato al soffitto,
    # uno "a pavimento" al pavimento. I due casi si escludono: per un
    # oggetto a soffitto la distanza dal pavimento locale non e'
    # significativa (puo' perfino risultare negativa se sotto di lui c'e'
    # un piano orizzontale ribassato) e viceversa.
    a_soffitto = modello["dist_soffitto"] <= 1.5
    if a_soffitto:
        if abs(float(feature.dist_da_soffitto) - modello["dist_soffitto"]) > tol_quota:
            return False
    elif modello["dist_pavimento"] <= 0.3:
        if abs(float(feature.dist_da_pavimento) - modello["dist_pavimento"]) > tol_quota:
            return False
    return True


def applica_modelli(
    modelli: list[dict], istanze: list, etichette_r, cfg: dict,
    rifiuti: list[dict] | None = None,
) -> int:
    """Riclassifica le istanze dubbie che combaciano con un modello.

    Un modello puo' restringere i propri bersagli col campo ``bersagli``
    (le correzioni dell'utente bersagliano anche la classe sbagliata di
    partenza); i ``rifiuti`` sono esempi negativi e vincono sempre: la
    firma che combacia con un rifiuto non viene mai convertita, nemmeno
    se combacia con un modello positivo. Ritorna il numero di istanze
    riclassificate.
    """
    nomi_default = cfg.get("classi_bersaglio") or CLASSI_BERSAGLIO_DEFAULT
    bersaglio_default = {ClasseMEP[str(n).strip().upper()] for n in nomi_default}
    rifiuti = rifiuti or []
    # memoizzato per istanza: il confronto coi rifiuti non dipende dal modello
    esito_rifiuto: dict[int, bool] = {}

    def rifiutata(ist) -> bool:
        if ist.id not in esito_rifiuto:
            esito_rifiuto[ist.id] = any(
                corrisponde(r["firma"], ist.feature, cfg) for r in rifiuti
            )
        return esito_rifiuto[ist.id]

    totale = 0
    n_bloccate = 0
    for modello in modelli:
        classe = ClasseMEP[str(modello["classe"]).strip().upper()]
        if modello.get("bersagli"):
            bersaglio = {
                ClasseMEP[str(n).strip().upper()] for n in modello["bersagli"]
            }
        else:
            bersaglio = bersaglio_default
        convertite = 0
        for ist in istanze:
            if ist.classe not in bersaglio or ist.classe == classe:
                continue
            if corrisponde(modello, ist.feature, cfg):
                if rifiutata(ist):
                    n_bloccate += 1
                    continue
                ist.classe = classe
                etichette_r[ist.indici] = int(classe)
                convertite += 1
        if convertite:
            log.info(
                "Somiglianza: %d istanze -> %s (modello '%s')",
                convertite, NOMI_CLASSI[classe], modello["nome"],
            )
        totale += convertite
    if n_bloccate:
        log.info(
            "Somiglianza: %d conversioni bloccate dai rifiuti dell'utente",
            n_bloccate,
        )
    return totale


# ---------------------------------------------------------------------------
# Fase 4: conferme, rifiuti e pin (conferme.json accanto all'output)
# ---------------------------------------------------------------------------

def stessa_istanza(firma: dict, feature) -> bool:
    """True se ``feature`` e' con ogni probabilita' la STESSA istanza fisica
    descritta dalla firma: centro entro TOL_CENTRO_PIN, lati OBB entro il
    20%. La classe d'origine e' volutamente ignorata (tra un run e l'altro
    la classe puo' cambiare: e' proprio cio' che il pin corregge)."""
    dist = sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(feature.centro, firma["centro"])
    ) ** 0.5
    if dist > TOL_CENTRO_PIN:
        return False
    for dim_ist, dim_f in zip(feature.dimensioni, firma["dimensioni"]):
        if abs(float(dim_ist) - dim_f) > max(TOL_DIM_PIN * dim_f, TOL_DIM_PIN_ABS):
            return False
    return True


def firme_coincidono(f1: dict, f2: dict) -> bool:
    """Confronto firma-firma con gli stessi criteri del pin (per evitare
    voci duplicate in conferme.json sulla stessa istanza fisica)."""
    dist = sum(
        (float(a) - float(b)) ** 2 for a, b in zip(f1["centro"], f2["centro"])
    ) ** 0.5
    if dist > TOL_CENTRO_PIN:
        return False
    for d1, d2 in zip(f1["dimensioni"], f2["dimensioni"]):
        if abs(float(d1) - float(d2)) > max(TOL_DIM_PIN * float(d2), TOL_DIM_PIN_ABS):
            return False
    return True


def applica_pin(
    conferme: list[dict], istanze: list, etichette_r,
    rifiuti: list[dict] | None = None,
) -> int:
    """Ri-applica le decisioni dell'utente alle istanze specifiche.

    Da chiamare DOPO tutte le regole e i modelli: l'utente ha l'ultima
    parola su tutto. Le istanze ELIMINATE (rifiuti) diventano WASTE —
    rumore da scartare, fuori dalle iterazioni di riconoscimento — poi le
    conferme pinnano la loro classe. Ritorna il numero di istanze
    riportate alla decisione dell'utente."""

    def istanza_bersaglio(firma: dict):
        candidate = [i for i in istanze if stessa_istanza(firma, i.feature)]
        if not candidate:
            return None
        centro = firma["centro"]
        return min(
            candidate,
            key=lambda i: sum(
                (float(a) - float(b)) ** 2 for a, b in zip(i.feature.centro, centro)
            ),
        )

    n_pin = 0
    for r in rifiuti or []:
        ist = istanza_bersaglio(r["firma"])
        if ist is None:
            log.warning(
                "Elimina: nessuna istanza corrispondente in questo run "
                "(centro %s)", r["firma"]["centro"],
            )
            continue
        if ist.classe != ClasseMEP.WASTE:
            ist.classe = ClasseMEP.WASTE
            etichette_r[ist.indici] = int(ClasseMEP.WASTE)
            n_pin += 1
    for c in conferme:
        classe = ClasseMEP[str(c["classe"]).strip().upper()]
        ist = istanza_bersaglio(c["firma"])
        if ist is None:
            log.warning(
                "Pin '%s' (%s): nessuna istanza corrispondente in questo run",
                c.get("nome") or "?", NOMI_CLASSI[classe],
            )
            continue
        if ist.classe != classe:
            ist.classe = classe
            etichette_r[ist.indici] = int(classe)
            n_pin += 1
    if conferme or rifiuti:
        log.info(
            "Pin: %d conferme e %d eliminazioni applicate, %d istanze "
            "riportate alla decisione dell'utente",
            len(conferme), len(rifiuti or []), n_pin,
        )
    return n_pin


def modelli_da_conferme(conferme: list[dict]) -> list[dict]:
    """Le conferme diventano modelli positivi per la propagazione.

    Una CORREZIONE ("questa 'condotta' e' in realta' un armadio") bersaglia
    anche la classe sbagliata di partenza: le altre istanze con la stessa
    firma e lo stesso errore vengono corrette in blocco."""
    modelli = []
    for c in conferme:
        m = dict(c["firma"])
        m["nome"] = c.get("nome") or f"conferma_{c['classe']}"
        m["classe"] = c["classe"]
        if c.get("origine") == "correzione" and c.get("classe_precedente"):
            m["bersagli"] = sorted(
                {c["classe_precedente"], *CLASSI_BERSAGLIO_DEFAULT}
            )
        modelli.append(m)
    return modelli


def crea_conferma(
    firma: dict, classe: str, nome: str = "", origine: str = "conferma",
    classe_precedente: str | None = None,
) -> dict:
    voce = {
        "firma": firma,
        "classe": str(classe).strip().lower(),
        "nome": nome or "",
        "origine": origine,
        "quando": datetime.now().isoformat(timespec="seconds"),
    }
    if classe_precedente:
        voce["classe_precedente"] = str(classe_precedente).strip().lower()
    return voce


def crea_rifiuto(firma: dict, classe_rifiutata: str) -> dict:
    return {
        "firma": firma,
        "classe_rifiutata": str(classe_rifiutata).strip().lower(),
        "quando": datetime.now().isoformat(timespec="seconds"),
    }


def carica_conferme(percorso: str | Path) -> dict:
    """Carica conferme.json; struttura sempre completa anche se il file
    manca (prima conferma di una nuvola nuova)."""
    percorso = Path(percorso)
    dati = {"versione": VERSIONE_CONFERME, "nuvola": "", "conferme": [], "rifiuti": []}
    if percorso.exists():
        with open(percorso, encoding="utf-8") as f:
            letti = json.load(f)
        dati.update({
            "nuvola": letti.get("nuvola", ""),
            "conferme": letti.get("conferme", []),
            "rifiuti": letti.get("rifiuti", []),
        })
    return dati


def salva_conferme(percorso: str | Path, dati: dict) -> None:
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    dati = {"versione": VERSIONE_CONFERME, **dati}
    with open(percorso, "w", encoding="utf-8") as f:
        json.dump(dati, f, ensure_ascii=False, indent=2)


def rimuovi_voci_istanza(dati: dict, firma: dict) -> int:
    """Toglie da conferme e rifiuti ogni voce riferita alla stessa istanza
    fisica (una nuova decisione dell'utente sostituisce la precedente).
    Ritorna quante voci sono state rimosse."""
    prima = len(dati["conferme"]) + len(dati["rifiuti"])
    dati["conferme"] = [
        c for c in dati["conferme"] if not firme_coincidono(c["firma"], firma)
    ]
    dati["rifiuti"] = [
        r for r in dati["rifiuti"] if not firme_coincidono(r["firma"], firma)
    ]
    return prima - len(dati["conferme"]) - len(dati["rifiuti"])


def carica_modelli(percorso: str | Path) -> list[dict]:
    percorso = Path(percorso)
    if not percorso.exists():
        return []
    with open(percorso, encoding="utf-8") as f:
        dati = json.load(f)
    return dati.get("modelli", [])


def salva_modello(percorso: str | Path, modello: dict) -> None:
    """Aggiunge (o sostituisce, a parita' di nome) un modello nel file."""
    percorso = Path(percorso)
    modelli = carica_modelli(percorso)
    modelli = [m for m in modelli if m.get("nome") != modello["nome"]]
    modelli.append(modello)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with open(percorso, "w", encoding="utf-8") as f:
        json.dump(
            {"versione": VERSIONE_FILE, "modelli": modelli},
            f, ensure_ascii=False, indent=2,
        )
