"""Server web locale per la GUI di mepseg.

Avvio:  mepseg-gui  [--porta 8765]

Espone una single-page app (static/index.html) e una API JSON:
  POST /api/sfoglia      -> apre il selettore file di Windows/OS
  POST /api/anteprima    -> carica/sottocampiona la nuvola e ritorna la pianta
  POST /api/segmenta     -> avvia la segmentazione (in un thread)
  GET  /api/stato        -> fase corrente, log, avanzamento
  GET  /api/risultato    -> report (conteggi, istanze, percorsi output)
  GET  /api/punti3d      -> nuvola segmentata binaria per il viewer three.js

Fase 4 — "Conferma e allinea":
  GET  /api/istanze          -> istanze del run corrente (dal report)
  GET  /api/conferme         -> conferme/rifiuti accumulati (conferme.json)
  POST /api/conferma         -> conferma / correggi / elimina / annulla
  POST /api/istanza_da_punto -> id istanza dal punto cliccato nel viewer
  POST /api/allinea          -> run di allineamento (stessi parametri, dalla
                                cache, con le conferme applicate)
  GET  /api/diff             -> istanze cambiate rispetto al run precedente

Fase 5 — "Addestra" (fine-tuning progressivo della rete):
  GET  /api/dataset_stato    -> nuvole/pesi del dataset cumulativo
  POST /api/esporta_dataset  -> etichette+fiducia del run nel dataset
  POST /api/allena           -> fine-tuning (checkpoint versionato)
  (il run con "usa_rete" fonde l'ultimo checkpoint: la rete riclassifica
   solo il MEP generico, mai strutture o conferme)

Suggerimento visivo (VLM locale via Ollama, prerequisito opzionale):
  GET  /api/vlm_stato        -> Ollama raggiungibile? quali modelli?
  POST /api/chiedi_vlm       -> viste 2D dell'istanza + domanda al modello
                                locale; il suggerimento NON decide mai da
                                solo, resta un consiglio nel pannello
                                Conferma (l'utente ha l'ultima parola)

Tutto gira in locale: nessun dato lascia il computer.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import webbrowser
from pathlib import Path

import numpy as np

try:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse, Response
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "L'interfaccia grafica richiede dipendenze extra: pip install .[gui]"
    ) from exc

CARTELLA_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="mepseg GUI")
log = logging.getLogger("mepseg")

# preset di contesto: classi impossibili per tipo di ambiente
PRESET_AMBIENTI = {
    "terziario": [],
    "data_center": ["sprinkler"],
    "centrale_termica": ["terminale_aria"],
    "industriale": [],
    "personalizzato": [],
}

FASI = [
    ("Epoca", "addestramento della rete"),
    ("Dataset:", "export nel dataset"),
    ("Fusione", "fusione con la rete"),
    ("Caricamento", "lettura della nuvola"),
    ("streaming:", "lettura in streaming"),
    ("cache", "nuvola ridotta dalla cache"),
    ("sottocampionamento", "sottocampionamento"),
    ("Struttura:", "strutture rilevate"),
    ("Cluster MEP", "clustering"),
    ("Contesto:", "regole di contesto"),
    ("Nuvola segmentata", "scrittura output"),
    ("export LAS", "export LAS piena risoluzione"),
    ("Report:", "report generato"),
]


class _StatoJob:
    def __init__(self) -> None:
        self.attivo = False
        self.fase = ""
        self.log: list[str] = []
        self.errore: str | None = None
        self.completato = False
        self.parametri: dict = {}
        self.percorso_ply: Path | None = None
        self.percorso_report: Path | None = None

    def azzera(self, parametri: dict) -> None:
        self.attivo = True
        self.fase = "avvio"
        self.log = []
        self.errore = None
        self.completato = False
        self.parametri = parametri
        self.percorso_ply = None
        self.percorso_report = None


STATO = _StatoJob()
ANTEPRIMA: dict = {}  # punti ridotti per la pianta, estensione
# ultima richiesta di segmentazione andata in porto: il run di allineamento
# la ripete identica (riusa la cache) aggiungendo le conferme
ULTIMA_SEGMENTA: "RichiestaSegmenta | None" = None
# KD-tree dei punti-istanza per la selezione per click nel viewer
_SELEZIONE: dict = {"percorso": None, "albero": None, "ids": None}


class _HandlerLog(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        messaggio = record.getMessage()
        STATO.log.append(messaggio)
        if len(STATO.log) > 400:
            del STATO.log[: len(STATO.log) - 400]
        for chiave, nome_fase in FASI:
            if chiave in messaggio:
                STATO.fase = nome_fase
                break


log.addHandler(_HandlerLog())
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Modelli richieste
# ---------------------------------------------------------------------------

class RichiestaAnteprima(BaseModel):
    percorso: str
    voxel: float = 0.02
    cartella_output: str = ""


class RichiestaSegmenta(BaseModel):
    percorso: str
    voxel: float = 0.02
    ambiente: str = "terziario"
    # elenco esplicito dalle spunte della GUI; se assente si usa il preset
    classi_escluse: list[str] | None = None
    bbox: list[float] | None = None      # [xmin, ymin, zmin, xmax, ymax, zmax]
    margine: float = 1.0
    las: bool = False
    cartella_output: str = ""
    # Fase 5: fusione con l'ultimo checkpoint della rete nel dataset
    usa_rete: bool = False
    cartella_dataset: str = ""


class RichiestaDataset(BaseModel):
    cartella_dataset: str = ""


class RichiestaAllena(BaseModel):
    cartella_dataset: str = ""
    epoche: int = 30
    punti_blocco: int = 8192
    batch: int = 2


class RichiestaConferma(BaseModel):
    id_istanza: int
    azione: str                # conferma | correggi | elimina | annulla
    classe: str | None = None  # chiave classe (solo per "correggi")
    nome: str | None = None
    # suggerimento del modello visivo mostrato all'utente al momento della
    # decisione: {classe, modello, quando} — tracciabilita', non decisione
    suggerimento_vlm: dict | None = None


class RichiestaVLM(BaseModel):
    id_istanza: int
    modello: str = ""          # vuoto = MODELLO_DEFAULT di vlm.py


class RichiestaPunto(BaseModel):
    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def indice():
    return FileResponse(CARTELLA_STATIC / "index.html")


@app.get("/static/{nome}")
def statico(nome: str):
    percorso = (CARTELLA_STATIC / nome).resolve()
    if percorso.parent != CARTELLA_STATIC or not percorso.exists():
        return Response(status_code=404)
    return FileResponse(percorso)


@app.post("/api/sfoglia")
def sfoglia():
    """Apre il selettore file nativo in un PROCESSO separato.

    tkinter dentro un thread del server e' fragile (crash intermittenti su
    Windows): un piccolo processo Python dedicato e' robusto e isolato.
    """
    import subprocess
    import sys

    codice = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "radice = tk.Tk(); radice.withdraw(); radice.attributes('-topmost', True)\n"
        "print(filedialog.askopenfilename(\n"
        "    title='Seleziona la nuvola di punti',\n"
        "    filetypes=[('Nuvole di punti', '*.e57 *.las *.laz *.ply *.pcd *.xyz *.txt'),\n"
        "               ('Tutti i file', '*.*')]))\n"
    )
    try:
        esito = subprocess.run(
            [sys.executable, "-c", codice],
            capture_output=True, text=True, timeout=600,
        )
        return {"percorso": esito.stdout.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"percorso": "", "errore": str(exc)}


def _valida_percorso(percorso_grezzo: str) -> Path:
    """Normalizza e valida il percorso della nuvola con messaggi chiari."""
    from ..io_nuvole import ESTENSIONI_SUPPORTATE

    percorso = Path(percorso_grezzo.strip().strip('"').strip("'"))
    if not percorso.exists():
        raise ValueError(f"File non trovato: {percorso}")
    if percorso.is_dir():
        raise ValueError(
            f"Il percorso e' una CARTELLA: seleziona il file della nuvola al "
            f"suo interno (formati: {', '.join(sorted(ESTENSIONI_SUPPORTATE))})"
        )
    if percorso.suffix.lower() not in ESTENSIONI_SUPPORTATE:
        raise ValueError(
            f"Formato '{percorso.suffix}' non supportato. Formati validi: "
            f"{', '.join(sorted(ESTENSIONI_SUPPORTATE))}"
        )
    return percorso


def _cartella_output(richiesta_percorso: str, cartella: str) -> Path:
    if cartella:
        return Path(cartella)
    origine = Path(richiesta_percorso)
    return origine.parent / f"output_{origine.stem}"


def _cartella_dataset(campo: str) -> Path | None:
    """Cartella del dataset cumulativo: campo esplicito della GUI, oppure
    ``dataset_mep`` accanto alla nuvola dell'ultima segmentazione."""
    if campo.strip():
        return Path(campo.strip().strip('"').strip("'"))
    if ULTIMA_SEGMENTA is not None:
        return Path(ULTIMA_SEGMENTA.percorso).parent / "dataset_mep"
    return None


def _lavoro_anteprima(richiesta: RichiestaAnteprima) -> None:
    try:
        from ..io_nuvole import carica_nuvola
        from ..streaming import conta_punti_e57, sottocampiona_e57_streaming

        percorso = Path(richiesta.percorso)
        cartella = _cartella_output(richiesta.percorso, richiesta.cartella_output)
        voxel = richiesta.voxel

        if percorso.suffix.lower() == ".e57" and conta_punti_e57(percorso) > 20_000_000:
            percorso_cache = cartella / f"{percorso.stem}_ridotta_{voxel * 1000:g}mm.npy"
            if percorso_cache.exists():
                log.info("Riuso la nuvola ridotta in cache: %s", percorso_cache)
                punti = np.load(percorso_cache)
            else:
                log.info("Passata 1: sottocampionamento in streaming...")
                punti, colori = sottocampiona_e57_streaming(percorso, voxel)
                cartella.mkdir(parents=True, exist_ok=True)
                np.save(percorso_cache, punti)
                # cache colore affiancata: il run di segmentazione la riusa
                # per il taglio per colore, l'anteprima usa solo la pianta XYZ
                if colori is not None:
                    np.save(
                        cartella / f"{percorso.stem}_ridotta_{voxel * 1000:g}mm_rgb.npy",
                        colori,
                    )
        else:
            log.info("Caricamento di %s ...", percorso)
            nuvola = carica_nuvola(percorso)
            punti = nuvola.punti

        ANTEPRIMA["punti"] = punti
        ANTEPRIMA["estensione"] = {
            "xmin": float(punti[:, 0].min()), "xmax": float(punti[:, 0].max()),
            "ymin": float(punti[:, 1].min()), "ymax": float(punti[:, 1].max()),
            "zmin": float(punti[:, 2].min()), "zmax": float(punti[:, 2].max()),
        }
        ANTEPRIMA["n_punti"] = int(len(punti))
        log.info("Anteprima pronta: %s punti", f"{len(punti):,}")
        STATO.fase = "anteprima pronta"
    except Exception as exc:  # noqa: BLE001
        STATO.errore = str(exc)
        log.info("Errore: %s", exc)
    finally:
        STATO.attivo = False
        STATO.completato = True


@app.post("/api/anteprima")
def anteprima(richiesta: RichiestaAnteprima):
    if STATO.attivo:
        return JSONResponse({"errore": "elaborazione in corso"}, status_code=409)
    try:
        richiesta.percorso = str(_valida_percorso(richiesta.percorso))
    except ValueError as exc:
        return JSONResponse({"errore": str(exc)}, status_code=400)
    STATO.azzera(richiesta.model_dump())
    threading.Thread(target=_lavoro_anteprima, args=(richiesta,), daemon=True).start()
    return {"avviato": True}


@app.get("/api/pianta")
def pianta():
    """PNG della pianta (vista dall'alto, colorata per quota) + estensione."""
    if "punti" not in ANTEPRIMA:
        return JSONResponse({"errore": "nessuna anteprima caricata"}, status_code=404)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    punti = ANTEPRIMA["punti"]
    if len(punti) > 400_000:
        rng = np.random.default_rng(0)
        punti = punti[rng.choice(len(punti), 400_000, replace=False)]

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#111")
    ax.set_facecolor("#111")
    ax.scatter(
        punti[:, 0], punti[:, 1], s=0.3, c=punti[:, 2], cmap="viridis", marker="."
    )
    ax.set_aspect("equal")
    ax.axis("off")
    e = ANTEPRIMA["estensione"]
    ax.set_xlim(e["xmin"], e["xmax"])
    ax.set_ylim(e["ymin"], e["ymax"])
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=110, bbox_inches="tight", pad_inches=0,
                facecolor="#111")
    plt.close(fig)
    return {
        "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "estensione": e,
        "n_punti": ANTEPRIMA["n_punti"],
    }


@app.get("/api/vista_laterale")
def vista_laterale():
    """PNG dell'elevazione (vista laterale): asse orizzontale = il lato piu'
    lungo in pianta (X o Y), asse verticale = Z. Serve a scegliere la quota
    Z a colpo d'occhio. Colorata per profondita' (l'asse collassato)."""
    if "punti" not in ANTEPRIMA:
        return JSONResponse({"errore": "nessuna anteprima caricata"}, status_code=404)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    punti = ANTEPRIMA["punti"]
    if len(punti) > 400_000:
        rng = np.random.default_rng(0)
        punti = punti[rng.choice(len(punti), 400_000, replace=False)]

    e = ANTEPRIMA["estensione"]
    # asse orizzontale = quello con estensione maggiore in pianta (elevazione
    # piu' informativa); l'asse collassato diventa il colore (profondita')
    asse_h = "x" if (e["xmax"] - e["xmin"]) >= (e["ymax"] - e["ymin"]) else "y"
    ih, ic = (0, 1) if asse_h == "x" else (1, 0)

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#111")
    ax.set_facecolor("#111")
    ax.scatter(
        punti[:, ih], punti[:, 2], s=0.3, c=punti[:, ic], cmap="viridis",
        marker=".",
    )
    ax.axis("off")
    ax.set_xlim(e[f"{asse_h}min"], e[f"{asse_h}max"])
    ax.set_ylim(e["zmin"], e["zmax"])  # aspetto NON equal: Z stirata, leggibile
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=110, bbox_inches="tight", pad_inches=0,
                facecolor="#111")
    plt.close(fig)
    return {
        "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "asse_h": asse_h,
        "estensione": {
            "hmin": e[f"{asse_h}min"], "hmax": e[f"{asse_h}max"],
            "zmin": e["zmin"], "zmax": e["zmax"],
        },
    }


def _lavoro_segmenta(richiesta: RichiestaSegmenta) -> None:
    try:
        from ..cli import principale as cli_principale

        cartella = _cartella_output(richiesta.percorso, richiesta.cartella_output)
        argv = [richiesta.percorso, "-o", str(cartella), "--voxel", str(richiesta.voxel)]

        if richiesta.classi_escluse is not None:
            escluse = [v.strip().lower() for v in richiesta.classi_escluse if v.strip()]
        else:
            escluse = list(PRESET_AMBIENTI.get(richiesta.ambiente, []))
        if escluse:
            argv += ["--escludi", ",".join(escluse)]
        if richiesta.bbox:
            argv.append("--bbox=" + ",".join(f"{v:.3f}" for v in richiesta.bbox))
            if richiesta.margine > 0:
                argv += ["--margine", str(richiesta.margine)]
        if richiesta.las:
            argv.append("--las")

        # le conferme della Fase 4 sopravvivono tra le sessioni accanto
        # all'output: se il file esiste, ogni run le rispetta
        percorso_conferme = cartella / "conferme.json"
        if percorso_conferme.exists():
            argv += ["--conferme", str(percorso_conferme)]

        # Fase 5: fusione con l'ultimo checkpoint del dataset (la rete
        # riclassifica solo il MEP generico, mai strutture o conferme)
        if richiesta.usa_rete:
            from ..dl.esporta import ultimo_checkpoint

            dataset = _cartella_dataset(richiesta.cartella_dataset)
            pesi = ultimo_checkpoint(dataset) if dataset else None
            if pesi is not None:
                argv += ["--pesi", str(pesi)]
                log.info("Fusione con la rete attiva: %s", pesi.name)
            else:
                log.warning(
                    "Nessun checkpoint nel dataset (%s): run senza rete",
                    dataset,
                )

        log.info("Comando equivalente: mepseg %s", " ".join(argv))
        codice = cli_principale(argv)
        if codice != 0:
            raise RuntimeError(f"la segmentazione e' uscita con codice {codice}")

        nome_base = Path(richiesta.percorso).stem
        for suffisso in ("_segmentata_ridotta.ply", "_segmentata.ply"):
            candidato = cartella / f"{nome_base}{suffisso}"
            if candidato.exists():
                STATO.percorso_ply = candidato
                break
        STATO.percorso_report = cartella / f"{nome_base}_report.json"
        STATO.fase = "completato"
    except Exception as exc:  # noqa: BLE001
        STATO.errore = str(exc)
        log.info("Errore: %s", exc)
    finally:
        STATO.attivo = False
        STATO.completato = True


@app.post("/api/segmenta")
def segmenta_api(richiesta: RichiestaSegmenta):
    global ULTIMA_SEGMENTA
    if STATO.attivo:
        return JSONResponse({"errore": "elaborazione in corso"}, status_code=409)
    try:
        richiesta.percorso = str(_valida_percorso(richiesta.percorso))
    except ValueError as exc:
        return JSONResponse({"errore": str(exc)}, status_code=400)
    ULTIMA_SEGMENTA = richiesta
    STATO.azzera(richiesta.model_dump())
    threading.Thread(target=_lavoro_segmenta, args=(richiesta,), daemon=True).start()
    return {"avviato": True}


@app.get("/api/stato")
def stato():
    return {
        "attivo": STATO.attivo,
        "fase": STATO.fase,
        "log": STATO.log[-40:],
        "errore": STATO.errore,
        "completato": STATO.completato,
    }


@app.get("/api/risultato")
def risultato():
    if not STATO.percorso_report or not STATO.percorso_report.exists():
        return JSONResponse({"errore": "nessun risultato disponibile"}, status_code=404)
    report = json.loads(STATO.percorso_report.read_text(encoding="utf-8"))
    return {
        "report": report,
        "percorso_ply": str(STATO.percorso_ply),
        "percorso_report": str(STATO.percorso_report),
    }


@app.get("/api/punti3d")
def punti3d(max_punti: int = 1_500_000):
    """Nuvola segmentata in binario: [uint32 n][xyz f32][rgb u8][classe u8]."""
    if not STATO.percorso_ply or not STATO.percorso_ply.exists():
        return Response(status_code=404)
    from ..io_nuvole import leggi_ply_classificata

    d = leggi_ply_classificata(STATO.percorso_ply)
    if len(d) > max_punti:
        rng = np.random.default_rng(0)
        d = d[rng.choice(len(d), max_punti, replace=False)]
    n = len(d)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype("<f4")
    rgb = np.column_stack([d["red"], d["green"], d["blue"]]).astype("u1")
    classe = d["classe"].astype("u1")
    corpo = (
        np.uint32(n).tobytes() + xyz.tobytes() + rgb.tobytes() + classe.tobytes()
    )
    return Response(content=corpo, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Fase 4 — Conferma e allinea
# ---------------------------------------------------------------------------

def _report_corrente() -> tuple[dict, Path] | None:
    if not STATO.percorso_report or not STATO.percorso_report.exists():
        return None
    report = json.loads(STATO.percorso_report.read_text(encoding="utf-8"))
    return report, STATO.percorso_report


def _nome_base_report(percorso_report: Path) -> str:
    return percorso_report.name.removesuffix("_report.json")


def _percorso_conferme() -> Path:
    return STATO.percorso_report.parent / "conferme.json"


def _conteggio_conferme() -> dict:
    from ..somiglianza import carica_conferme

    dati = carica_conferme(_percorso_conferme())
    return {
        "n_conferme": len(dati["conferme"]),
        "n_rifiuti": len(dati["rifiuti"]),
    }


@app.get("/api/istanze")
def istanze_api():
    """Istanze del run corrente: il report JSON ha gia' tutto."""
    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    report, _ = corrente
    return {"istanze": report["istanze"], **_conteggio_conferme()}


@app.get("/api/conferme")
def conferme_api():
    from ..somiglianza import carica_conferme

    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    dati = carica_conferme(_percorso_conferme())
    voci = [
        {
            "tipo": "conferma", "nome": c.get("nome", ""), "classe": c["classe"],
            "origine": c.get("origine", "conferma"), "quando": c.get("quando", ""),
            "centro": c["firma"]["centro"],
        }
        for c in dati["conferme"]
    ] + [
        {
            "tipo": "rifiuto", "nome": "", "classe": r.get("classe_rifiutata", ""),
            "origine": "rifiuto", "quando": r.get("quando", ""),
            "centro": r["firma"]["centro"],
        }
        for r in dati["rifiuti"]
    ]
    return {
        "n_conferme": len(dati["conferme"]),
        "n_rifiuti": len(dati["rifiuti"]),
        "voci": voci,
    }


@app.post("/api/conferma")
def conferma_api(richiesta: RichiestaConferma):
    from ..classi import ClasseMEP
    from ..somiglianza import (
        carica_conferme,
        crea_conferma,
        crea_rifiuto,
        firma_da_riga,
        rimuovi_voci_istanza,
        salva_conferme,
    )

    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    report, percorso_report = corrente

    riga = next(
        (r for r in report["istanze"] if r["id"] == richiesta.id_istanza), None
    )
    if riga is None:
        return JSONResponse(
            {"errore": f"istanza {richiesta.id_istanza} non trovata"},
            status_code=404,
        )
    firma = firma_da_riga(riga)
    classe_attuale = ClasseMEP(riga["classe_codice"]).name.lower()

    percorso = _percorso_conferme()
    dati = carica_conferme(percorso)
    if not dati["nuvola"]:
        dati["nuvola"] = _nome_base_report(percorso_report)
    # una nuova decisione sulla stessa istanza sostituisce la precedente
    rimuovi_voci_istanza(dati, firma)

    azione = richiesta.azione.strip().lower()
    if azione == "conferma":
        voce = crea_conferma(firma, classe_attuale, nome=richiesta.nome or "")
        # il suggerimento VLM mostrato al momento della decisione resta
        # nella conferma (con il modello che l'ha prodotto): distinguere
        # in futuro cosa ha suggerito quale versione, MAI cosa ha deciso
        # (la classe confermata e' sempre e solo quella dell'utente)
        if richiesta.suggerimento_vlm:
            voce["suggerimento_vlm"] = richiesta.suggerimento_vlm
        dati["conferme"].append(voce)
    elif azione == "correggi":
        if not richiesta.classe:
            return JSONResponse(
                {"errore": "l'azione 'correggi' richiede la classe"},
                status_code=400,
            )
        try:
            ClasseMEP[richiesta.classe.strip().upper()]
        except KeyError:
            return JSONResponse(
                {"errore": f"classe sconosciuta: {richiesta.classe}"},
                status_code=400,
            )
        voce = crea_conferma(
            firma, richiesta.classe, nome=richiesta.nome or "",
            origine="correzione", classe_precedente=classe_attuale,
        )
        if richiesta.suggerimento_vlm:
            voce["suggerimento_vlm"] = richiesta.suggerimento_vlm
        dati["conferme"].append(voce)
    elif azione in ("elimina", "rifiuta"):  # "rifiuta" = nome storico
        dati["rifiuti"].append(crea_rifiuto(firma, classe_attuale))
    elif azione != "annulla":
        return JSONResponse(
            {"errore": f"azione sconosciuta: {richiesta.azione}"}, status_code=400
        )
    salva_conferme(percorso, dati)
    return {
        "n_conferme": len(dati["conferme"]),
        "n_rifiuti": len(dati["rifiuti"]),
        "percorso": str(percorso),
    }


@app.post("/api/istanza_da_punto")
def istanza_da_punto(richiesta: RichiestaPunto):
    """Id dell'istanza piu' vicina al punto cliccato nel viewer.

    Usa il KD-tree dei soli punti che appartengono a un'istanza MEP
    (<nome>_istanze_punti.npz, scritto dal report): cliccare una parete o
    il rumore non seleziona nulla."""
    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    report, percorso_report = corrente

    nome_base = _nome_base_report(percorso_report)
    percorso_npz = percorso_report.parent / f"{nome_base}_istanze_punti.npz"
    if not percorso_npz.exists():
        return JSONResponse(
            {"errore": "punti-istanza non disponibili (rilancia il run)"},
            status_code=404,
        )
    chiave = (str(percorso_npz), percorso_npz.stat().st_mtime)
    if _SELEZIONE["percorso"] != chiave:
        from scipy.spatial import cKDTree

        dati = np.load(percorso_npz)
        _SELEZIONE["albero"] = cKDTree(dati["punti"])
        _SELEZIONE["ids"] = dati["id_istanza"]
        _SELEZIONE["percorso"] = chiave

    dist, indice = _SELEZIONE["albero"].query(
        [richiesta.x, richiesta.y, richiesta.z], distance_upper_bound=0.25
    )
    if not np.isfinite(dist):
        return {"id": 0}
    id_ist = int(_SELEZIONE["ids"][indice])
    riga = next((r for r in report["istanze"] if r["id"] == id_ist), None)
    return {"id": id_ist, "istanza": riga}


@app.post("/api/allinea")
def allinea_api():
    """Run di allineamento: ripete l'ultima segmentazione (riusando la
    cache della nuvola ridotta) con le conferme accumulate; il report
    corrente viene salvato come riferimento per il diff."""
    if STATO.attivo:
        return JSONResponse({"errore": "elaborazione in corso"}, status_code=409)
    if ULTIMA_SEGMENTA is None:
        return JSONResponse(
            {"errore": "esegui prima una segmentazione in questa sessione"},
            status_code=400,
        )
    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    _, percorso_report = corrente
    if not _percorso_conferme().exists():
        return JSONResponse(
            {"errore": "nessuna conferma accumulata: conferma almeno un'istanza"},
            status_code=400,
        )

    nome_base = _nome_base_report(percorso_report)
    percorso_prec = percorso_report.parent / f"{nome_base}_report_prec.json"
    percorso_prec.write_text(
        percorso_report.read_text(encoding="utf-8"), encoding="utf-8"
    )

    STATO.azzera(ULTIMA_SEGMENTA.model_dump())
    threading.Thread(
        target=_lavoro_segmenta, args=(ULTIMA_SEGMENTA,), daemon=True
    ).start()
    return {"avviato": True}


@app.get("/api/diff")
def diff_api():
    """Istanze la cui classe e' cambiata rispetto al run precedente.

    Confronto per firma geometrica (stesso matching del pin: centro +
    lati OBB), non per id: gli id possono slittare tra un run e l'altro."""
    from ..somiglianza import firma_da_riga, firme_coincidono

    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    report, percorso_report = corrente
    nome_base = _nome_base_report(percorso_report)
    percorso_prec = percorso_report.parent / f"{nome_base}_report_prec.json"
    if not percorso_prec.exists():
        return JSONResponse(
            {"errore": "nessun run precedente: esegui prima un allineamento"},
            status_code=404,
        )
    prec = json.loads(percorso_prec.read_text(encoding="utf-8"))

    firme_prec = [(r, firma_da_riga(r)) for r in prec["istanze"]]
    cambiate, accoppiate = [], 0
    for riga in report["istanze"]:
        firma = firma_da_riga(riga)
        migliore, dist_migliore = None, None
        for riga_p, firma_p in firme_prec:
            if not firme_coincidono(firma_p, firma):
                continue
            d = sum(
                (a - b) ** 2
                for a, b in zip(firma["centro"], firma_p["centro"])
            )
            if dist_migliore is None or d < dist_migliore:
                migliore, dist_migliore = riga_p, d
        if migliore is None:
            continue
        accoppiate += 1
        if migliore["classe_codice"] != riga["classe_codice"]:
            cambiate.append({
                "id": riga["id"],
                "classe_prima": migliore["classe"],
                "classe_dopo": riga["classe"],
                "n_punti": riga["n_punti"],
                "centro_x": riga["centro_x"],
                "centro_y": riga["centro_y"],
                "centro_z": riga["centro_z"],
                "dim_1": riga["dim_1"],
                "dim_2": riga["dim_2"],
                "dim_3": riga["dim_3"],
            })
    return {
        "cambiate": cambiate,
        "n_accoppiate": accoppiate,
        "n_prima": len(prec["istanze"]),
        "n_dopo": len(report["istanze"]),
    }


# ---------------------------------------------------------------------------
# Suggerimento visivo — VLM locale via Ollama (prerequisito opzionale)
# ---------------------------------------------------------------------------

# cache dei dati pesanti per il rendering delle viste: i punti-istanza del
# run (npz) e la nuvola ridotta segmentata (ply) si caricano una volta sola
_VLM_CACHE: dict = {"npz_chiave": None, "npz": None, "ply_chiave": None, "ply": None}


@app.get("/api/vlm_stato")
def vlm_stato_api():
    """Ollama e' raggiungibile? Con quali modelli? Non fallisce mai: la GUI
    usa la risposta per abilitare/disabilitare il bottone del suggerimento."""
    from ..vlm import MODELLO_DEFAULT, stato_ollama

    stato = stato_ollama()
    stato["modello_default"] = MODELLO_DEFAULT
    return stato


@app.post("/api/chiedi_vlm")
def chiedi_vlm_api(richiesta: RichiestaVLM):
    """Viste 2D dell'istanza + domanda al modello visivo LOCALE.

    Il risultato e' un SUGGERIMENTO nel pannello Conferma: non tocca mai
    etichette, regole o conferme — decide sempre l'utente. Se Ollama non
    e' in esecuzione risponde 503 con un messaggio chiaro e il resto
    della GUI continua a funzionare (prerequisito opzionale)."""
    from datetime import datetime

    from ..vlm import MODELLO_DEFAULT, OllamaNonDisponibile, suggerisci_classe

    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    report, percorso_report = corrente
    riga = next(
        (r for r in report["istanze"] if r["id"] == richiesta.id_istanza), None
    )
    if riga is None:
        return JSONResponse(
            {"errore": f"istanza {richiesta.id_istanza} non trovata"},
            status_code=404,
        )

    nome_base = _nome_base_report(percorso_report)
    percorso_npz = percorso_report.parent / f"{nome_base}_istanze_punti.npz"
    if not percorso_npz.exists():
        return JSONResponse(
            {"errore": "punti-istanza non disponibili (rilancia il run)"},
            status_code=404,
        )
    chiave = (str(percorso_npz), percorso_npz.stat().st_mtime)
    if _VLM_CACHE["npz_chiave"] != chiave:
        dati = np.load(percorso_npz)
        _VLM_CACHE["npz"] = (dati["punti"], dati["id_istanza"])
        _VLM_CACHE["npz_chiave"] = chiave
    punti_npz, ids_npz = _VLM_CACHE["npz"]
    punti_ist = punti_npz[ids_npz == richiesta.id_istanza]
    if len(punti_ist) == 0:
        return JSONResponse(
            {"errore": "nessun punto per questa istanza"}, status_code=404
        )

    # contesto locale (~1.5 m attorno all'istanza) dalla nuvola ridotta
    # segmentata: aiuta il modello a capire dov'e' l'oggetto (a soffitto,
    # su un rack, a pavimento). Facoltativo: senza PLY si procede comunque
    contesto = None
    if STATO.percorso_ply and STATO.percorso_ply.exists():
        chiave_ply = (str(STATO.percorso_ply), STATO.percorso_ply.stat().st_mtime)
        if _VLM_CACHE["ply_chiave"] != chiave_ply:
            from ..io_nuvole import leggi_ply_classificata

            d = leggi_ply_classificata(STATO.percorso_ply)
            _VLM_CACHE["ply"] = np.column_stack(
                [d["x"], d["y"], d["z"]]
            ).astype(np.float32)
            _VLM_CACHE["ply_chiave"] = chiave_ply
        nuvola = _VLM_CACHE["ply"]
        margine = 1.5
        bmin = punti_ist.min(axis=0) - margine
        bmax = punti_ist.max(axis=0) + margine
        dentro = np.all((nuvola >= bmin) & (nuvola <= bmax), axis=1)
        contesto = nuvola[dentro]
        if len(contesto) > 150_000:  # il rendering non ha bisogno di piu'
            rng = np.random.default_rng(0)
            contesto = contesto[rng.choice(len(contesto), 150_000, replace=False)]

    modello = richiesta.modello.strip() or MODELLO_DEFAULT
    try:
        esito = suggerisci_classe(punti_ist, contesto, riga, modello=modello)
    except OllamaNonDisponibile as exc:
        return JSONResponse(
            {"errore": str(exc), "vlm_assente": True}, status_code=503
        )
    except Exception as exc:  # noqa: BLE001 — mai bloccare la GUI per il VLM
        return JSONResponse({"errore": str(exc)}, status_code=500)
    esito["quando"] = datetime.now().isoformat(timespec="seconds")
    esito["id_istanza"] = richiesta.id_istanza
    return esito


# ---------------------------------------------------------------------------
# Fase 5 — Addestra (fine-tuning della rete, progressivo)
# ---------------------------------------------------------------------------

@app.get("/api/dataset_stato")
def dataset_stato_api(cartella: str = ""):
    from ..dl.esporta import stato_dataset

    dataset = _cartella_dataset(cartella)
    if dataset is None:
        return JSONResponse(
            {"errore": "esegui prima una segmentazione (o indica la cartella)"},
            status_code=400,
        )
    stato = stato_dataset(dataset)
    stato["cartella"] = str(dataset)
    return stato


def _lavoro_esporta_dataset(cartella_run: Path, dataset: Path) -> None:
    try:
        from ..dl.esporta import esporta_run

        stat = esporta_run(cartella_run, dataset)
        STATO.fase = (
            f"dataset aggiornato: {stat['nuvole_nel_dataset']} nuvole, "
            f"{stat['punti_confermati']:,} punti confermati in questa"
        )
    except Exception as exc:  # noqa: BLE001
        STATO.errore = str(exc)
        log.info("Errore: %s", exc)
    finally:
        STATO.attivo = False
        STATO.completato = True


@app.post("/api/esporta_dataset")
def esporta_dataset_api(richiesta: RichiestaDataset):
    if STATO.attivo:
        return JSONResponse({"errore": "elaborazione in corso"}, status_code=409)
    corrente = _report_corrente()
    if corrente is None:
        return JSONResponse({"errore": "nessun run disponibile"}, status_code=404)
    dataset = _cartella_dataset(richiesta.cartella_dataset)
    if dataset is None:
        return JSONResponse({"errore": "cartella dataset non determinabile"},
                            status_code=400)
    _, percorso_report = corrente
    STATO.azzera({"esporta_dataset": str(dataset)})
    threading.Thread(
        target=_lavoro_esporta_dataset,
        args=(percorso_report.parent, dataset), daemon=True,
    ).start()
    return {"avviato": True, "cartella": str(dataset)}


def _lavoro_allena(dataset: Path, richiesta: RichiestaAllena) -> None:
    try:
        from ..dl.allenamento import principale as allena_principale
        from ..dl.esporta import prossimo_checkpoint

        destinazione = prossimo_checkpoint(dataset)
        argv = [
            str(dataset), "-o", str(destinazione),
            "--epoche", str(richiesta.epoche),
            "--punti-blocco", str(richiesta.punti_blocco),
            "--batch", str(richiesta.batch),
        ]
        log.info("Comando equivalente: python -m mepseg.dl.allenamento %s",
                 " ".join(argv))
        codice = allena_principale(argv)
        if codice != 0:
            raise RuntimeError(f"l'addestramento e' uscito con codice {codice}")
        STATO.fase = f"addestramento completato: {destinazione.name}"
    except Exception as exc:  # noqa: BLE001
        STATO.errore = str(exc)
        log.info("Errore: %s", exc)
    finally:
        STATO.attivo = False
        STATO.completato = True


@app.post("/api/allena")
def allena_api(richiesta: RichiestaAllena):
    if STATO.attivo:
        return JSONResponse({"errore": "elaborazione in corso"}, status_code=409)
    dataset = _cartella_dataset(richiesta.cartella_dataset)
    if dataset is None:
        return JSONResponse({"errore": "cartella dataset non determinabile"},
                            status_code=400)
    if not (dataset / "train").exists() or not (dataset / "val").exists():
        return JSONResponse(
            {"errore": "dataset vuoto: esporta prima le etichette di un run"},
            status_code=400,
        )
    STATO.azzera({"allena": str(dataset), "epoche": richiesta.epoche})
    threading.Thread(
        target=_lavoro_allena, args=(dataset, richiesta), daemon=True
    ).start()
    return {"avviato": True, "cartella": str(dataset)}


@app.get("/api/classi")
def classi_api():
    from ..classi import COLORI_CLASSI, NOMI_CLASSI, ClasseMEP

    # struttura (in tutte le sue forme), generico, arredo e waste non sono
    # "cercabili": sono rispettivamente la base, la ricaduta, il
    # declassamento e lo scarto — non ha senso escluderli
    non_escludibili = {
        ClasseMEP.STRUTTURA, ClasseMEP.MEP_GENERICO, ClasseMEP.NON_MEP_ARREDO,
        ClasseMEP.PAVIMENTO, ClasseMEP.SOFFITTO, ClasseMEP.PARETE,
        ClasseMEP.PILASTRO, ClasseMEP.TRAVE, ClasseMEP.WASTE,
    }
    return {
        str(int(c)): {
            "nome": NOMI_CLASSI[c],
            "colore": COLORI_CLASSI[c],
            "chiave": c.name.lower(),
            "escludibile": c not in non_escludibili,
        }
        for c in ClasseMEP
    }


@app.get("/api/ambienti")
def ambienti_api():
    return PRESET_AMBIENTI


def principale() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Interfaccia web locale di mepseg")
    parser.add_argument("--porta", type=int, default=8765)
    parser.add_argument("--niente-browser", action="store_true")
    argomenti = parser.parse_args()

    if not argomenti.niente_browser:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{argomenti.porta}")
        ).start()
    uvicorn.run(app, host="127.0.0.1", port=argomenti.porta, log_level="warning")


if __name__ == "__main__":
    principale()
