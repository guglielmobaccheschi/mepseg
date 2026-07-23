"""Suggerimento di classe da un modello vision-language LOCALE (Ollama).

Terzo stadio del confronto con la letteratura (Li/Gan/Wang 2025, vedi
`C - Allineamento.md` §3.3): dove ne' la geometria ne' le regole bastano
(pannelli scambiati per luci, armadietti classificati apparecchiatura), un
VLM guarda l'istanza renderizzata in 2-3 viste ortogonali e propone una
classe. Il suggerimento NON sostituisce mai una regola o una conferma:
compare nel pannello "Conferma" della Fase 4 e la decisione resta
all'utente (responsabilita' professionale).

Vincolo architetturale: modello locale via Ollama (server REST su
127.0.0.1:11434) — nessun dato lascia mai la macchina, coerente con il
resto del tool. Ollama e' un PREREQUISITO OPZIONALE: se non e' in
esecuzione, ogni funzione fallisce in modo pulito con
:class:`OllamaNonDisponibile` e il resto della GUI continua a funzionare.

Modelli consigliati (da scaricare con ``ollama pull <nome>``):
  - ``moondream:1.8b``  — minuscolo, per lo smoke-test su CPU;
  - ``qwen2.5vl:3b``    — qualita' migliore, ancora usabile su CPU;
  - ``llava:7b`` / ``minicpm-v`` — con GPU (8GB+ VRAM).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request

import numpy as np

from .classi import ClasseMEP, NOMI_CLASSI

log = logging.getLogger("mepseg")

URL_OLLAMA = "http://127.0.0.1:11434"
MODELLO_DEFAULT = "moondream:1.8b"
TIMEOUT_INFERENZA = 180  # s: su CPU un 3B puo' impiegare piu' di un minuto

# classi proponibili al modello: quelle impiantistiche + l'arredo (il caso
# "armadietto scambiato per apparecchiatura" e' proprio uno dei bersagli).
# Niente classi strutturali: pavimenti/pareti/pilastri li decide la
# geometria a monte, il VLM interviene solo sulle istanze dubbie.
CLASSI_VLM: list[ClasseMEP] = [
    ClasseMEP.CONDOTTA_CIRCOLARE,
    ClasseMEP.CONDOTTA_RETTANGOLARE,
    ClasseMEP.TUBAZIONE,
    ClasseMEP.PASSERELLA_CAVI,
    ClasseMEP.LUCE,
    ClasseMEP.SPRINKLER,
    ClasseMEP.TERMINALE_ARIA,
    ClasseMEP.APPARECCHIATURA,
    ClasseMEP.NON_MEP_ARREDO,
]

# sinonimi che i modelli usano spesso al posto dei nomi della tassonomia
_SINONIMI: dict[str, ClasseMEP] = {
    "armadio": ClasseMEP.NON_MEP_ARREDO,
    "armadietto": ClasseMEP.NON_MEP_ARREDO,
    "scaffale": ClasseMEP.NON_MEP_ARREDO,
    "scrivania": ClasseMEP.NON_MEP_ARREDO,
    "mobile": ClasseMEP.NON_MEP_ARREDO,
    "plafoniera": ClasseMEP.LUCE,
    "lampada": ClasseMEP.LUCE,
    "neon": ClasseMEP.LUCE,
    "tubo": ClasseMEP.TUBAZIONE,
    "canalina": ClasseMEP.PASSERELLA_CAVI,
    "passerella": ClasseMEP.PASSERELLA_CAVI,
    "portacavi": ClasseMEP.PASSERELLA_CAVI,
    "diffusore": ClasseMEP.TERMINALE_ARIA,
    "bocchetta": ClasseMEP.TERMINALE_ARIA,
    "griglia": ClasseMEP.TERMINALE_ARIA,
    "rack": ClasseMEP.APPARECCHIATURA,
    "quadro": ClasseMEP.APPARECCHIATURA,
    "macchina": ClasseMEP.APPARECCHIATURA,
}


class OllamaNonDisponibile(RuntimeError):
    """Ollama non in esecuzione o non raggiungibile: il modulo VLM e'
    opzionale, chi chiama deve degradare senza bloccare la GUI."""


# ---------------------------------------------------------------------------
# Client Ollama (stdlib: nessuna dipendenza nuova)
# ---------------------------------------------------------------------------

def stato_ollama(url: str = URL_OLLAMA, timeout: float = 3.0) -> dict:
    """Stato del server: ``{"disponibile": bool, "modelli": [nomi...]}``.

    Non solleva mai: la GUI lo interroga per abilitare/disabilitare il
    bottone del suggerimento."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            dati = json.loads(r.read().decode("utf-8"))
        modelli = [m.get("name", "") for m in dati.get("models", [])]
        return {"disponibile": True, "modelli": modelli}
    except Exception:  # noqa: BLE001 — qualunque guasto = non disponibile
        return {"disponibile": False, "modelli": []}


def _interroga_ollama(
    immagini_b64: list[str], domanda: str, modello: str,
    url: str = URL_OLLAMA, timeout: float = TIMEOUT_INFERENZA,
) -> str:
    """Una domanda con immagini al modello locale; ritorna il testo."""
    corpo = json.dumps({
        "model": modello,
        "messages": [
            {"role": "user", "content": domanda, "images": immagini_b64}
        ],
        "stream": False,
        "options": {"temperature": 0},  # suggerimenti riproducibili
    }).encode("utf-8")
    richiesta = urllib.request.Request(
        f"{url}/api/chat", data=corpo,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(richiesta, timeout=timeout) as r:
            dati = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        dettaglio = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 404:
            raise OllamaNonDisponibile(
                f"modello '{modello}' non installato in Ollama: "
                f"eseguire `ollama pull {modello}` ({dettaglio})"
            ) from exc
        raise OllamaNonDisponibile(
            f"Ollama ha risposto {exc.code}: {dettaglio}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise OllamaNonDisponibile(
            "Ollama non raggiungibile su "
            f"{url}: installarlo/avviarlo e' un prerequisito opzionale "
            "(https://ollama.com). Il resto del tool funziona senza."
        ) from exc
    return str(dati.get("message", {}).get("content", "")).strip()


# ---------------------------------------------------------------------------
# Rendering delle viste 2D (stesse librerie della pianta: matplotlib)
# ---------------------------------------------------------------------------

def renderizza_viste(
    punti_istanza: np.ndarray,
    punti_contesto: np.ndarray | None = None,
    risoluzione: int = 640,
) -> list[str]:
    """Tre viste ortogonali dell'istanza come PNG base64.

    L'istanza e' colorata per profondita' (dare al modello un indizio di
    forma 3D), il contesto locale e' grigio chiaro. Gli assi restano
    visibili CON le tacche in metri: la scala e' l'informazione che
    un'immagine da sola non avrebbe (una plafoniera e un controsoffitto
    hanno la stessa forma, non la stessa taglia).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # tre proiezioni: dall'alto (X-Y), laterale (X-Z), laterale (Y-Z)
    viste = [
        ((0, 1), 2, "vista dall'alto (X-Y, m)"),
        ((0, 2), 1, "vista laterale (X-Z, m)"),
        ((1, 2), 0, "vista laterale (Y-Z, m)"),
    ]
    immagini: list[str] = []
    for (a, b), profondita, titolo in viste:
        fig, ax = plt.subplots(
            figsize=(risoluzione / 100, risoluzione / 100), dpi=100,
            facecolor="white",
        )
        if punti_contesto is not None and len(punti_contesto):
            ax.scatter(
                punti_contesto[:, a], punti_contesto[:, b],
                s=1.0, c="#c8c8c8", marker=".", linewidths=0,
            )
        ax.scatter(
            punti_istanza[:, a], punti_istanza[:, b],
            s=2.5, c=punti_istanza[:, profondita], cmap="viridis",
            marker=".", linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_title(titolo, fontsize=10)
        ax.tick_params(labelsize=8)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        immagini.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return immagini


# ---------------------------------------------------------------------------
# Domanda strutturata e lettura della risposta
# ---------------------------------------------------------------------------

def _domanda(riga: dict) -> str:
    elenco = "\n".join(
        f"- {c.name.lower()}: {NOMI_CLASSI[c]}" for c in CLASSI_VLM
    )
    return (
        "Le immagini sono tre viste ortogonali (dall'alto, laterale X-Z, "
        "laterale Y-Z) della nuvola di punti di UN oggetto rilevato in un "
        "edificio, durante un rilievo di impianti MEC/MEP. L'oggetto in "
        "esame e' colorato, il contesto circostante e' grigio. Gli assi "
        "sono in metri.\n"
        f"Dimensioni dell'oggetto (m): {riga.get('dim_1')} x "
        f"{riga.get('dim_2')} x {riga.get('dim_3')}; distanza dal "
        f"soffitto {riga.get('dist_soffitto')} m, dal pavimento "
        f"{riga.get('dist_pavimento')} m.\n"
        "A quale di queste classi appartiene con piu' probabilita'?\n"
        f"{elenco}\n"
        "Rispondi nella prima riga SOLO con il codice della classe (es. "
        "'luce'), poi motiva brevemente in italiano."
    )


def classe_da_risposta(testo: str) -> ClasseMEP | None:
    """Estrae la classe suggerita dal testo libero del modello.

    Ordine di ricerca: codici della tassonomia (prima riga e poi tutto il
    testo), nomi per esteso, sinonimi comuni. ``None`` se non si riconosce
    nulla: nessun suggerimento e' meglio di un suggerimento inventato."""
    t = testo.lower()
    prima_riga = t.splitlines()[0] if t.splitlines() else ""
    for zona in (prima_riga, t):
        for c in CLASSI_VLM:
            chiave = c.name.lower()
            if chiave in zona or chiave.replace("_", " ") in zona:
                return c
        for c in CLASSI_VLM:
            if NOMI_CLASSI[c].lower() in zona:
                return c
        for parola, c in _SINONIMI.items():
            if parola in zona:
                return c
    return None


def suggerisci_classe(
    punti_istanza: np.ndarray,
    punti_contesto: np.ndarray | None,
    riga: dict,
    modello: str = MODELLO_DEFAULT,
    url: str = URL_OLLAMA,
) -> dict:
    """Rendering + domanda al VLM locale; ritorna il suggerimento.

    Solleva :class:`OllamaNonDisponibile` se il server non risponde o il
    modello non e' installato — chi chiama degrada senza bloccare.
    Il campo ``modello`` resta nel risultato per la tracciabilita' in
    ``conferme.json`` (capire in futuro quale versione ha suggerito cosa).
    """
    stato = stato_ollama(url)
    if not stato["disponibile"]:
        raise OllamaNonDisponibile(
            f"Ollama non in esecuzione su {url}: avviarlo (o installarlo da "
            "https://ollama.com) per abilitare il suggerimento visivo. "
            "Prerequisito opzionale: il resto del tool funziona senza."
        )

    immagini = renderizza_viste(punti_istanza, punti_contesto)
    risposta = _interroga_ollama(immagini, _domanda(riga), modello, url)
    classe = classe_da_risposta(risposta)
    log.info(
        "VLM (%s) su istanza %s: %s",
        modello, riga.get("id"),
        NOMI_CLASSI[classe] if classe is not None else "nessuna classe riconosciuta",
    )
    return {
        "classe": classe.name.lower() if classe is not None else None,
        "classe_nome": NOMI_CLASSI[classe] if classe is not None else None,
        "risposta": risposta,
        "modello": modello,
        "viste": immagini,
    }
