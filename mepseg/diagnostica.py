"""Autodiagnosi dell'ambiente per mepseg.

Elenca cosa serve, cosa e' presente e cosa manca — con il comando esatto
per installare cio' che manca. Non fallisce mai: e' pensato per il primo
avvio di un utente qualsiasi ("perche' non parte?").

Eseguibile con ``mepseg --controlla`` oppure ``python -m mepseg.diagnostica``.
"""
from __future__ import annotations

import importlib
import sys

PYTHON_MIN = (3, 9)

# dipendenze base, obbligatorie: (modulo da importare, pacchetto pip, uso)
BASE = [
    ("numpy", "numpy", "array e algebra"),
    ("scipy", "scipy", "KD-tree, matrici sparse, ndimage"),
    ("sklearn", "scikit-learn", "DBSCAN di raffinamento"),
    ("open3d", "open3d", "normali, RANSAC, voxel"),
    ("laspy", "laspy[lazrs]", "lettura/scrittura LAS/LAZ"),
    ("yaml", "pyyaml", "file di configurazione"),
]

# gruppi opzionali: nome -> (extra pip, [(modulo, pacchetto, uso), ...])
OPZIONALI = {
    "E57 (lettura file .e57)": ("e57", [
        ("pye57", "pye57", "lettura file E57"),
    ]),
    "Deep learning (rete opzionale)": ("dl", [
        ("torch", "torch", "rete di segmentazione (CPU o GPU CUDA)"),
    ]),
    "GUI (interfaccia web locale)": ("gui", [
        ("fastapi", "fastapi", "server web"),
        ("uvicorn", "uvicorn", "server ASGI"),
        ("matplotlib", "matplotlib", "pianta e viste 2D"),
    ]),
}


def _importa(modulo: str) -> tuple[bool, str | None]:
    try:
        m = importlib.import_module(modulo)
        ver = getattr(m, "__version__", None)
        # alcuni pacchetti (es. pye57) espongono __version__ come sotto-modulo
        return True, ver if isinstance(ver, str) else "?"
    except Exception:
        return False, None


def controlla_ambiente() -> dict:
    """Report strutturato dell'ambiente (non stampa nulla)."""

    def blocco(elenco):
        voci = []
        for modulo, pip, uso in elenco:
            ok, ver = _importa(modulo)
            voci.append({"modulo": modulo, "pip": pip, "uso": uso,
                         "ok": ok, "versione": ver})
        return voci

    base = blocco(BASE)
    opzionali = {}
    for nome, (extra, elenco) in OPZIONALI.items():
        voci = blocco(elenco)
        opzionali[nome] = {"extra": extra, "voci": voci,
                           "completo": all(v["ok"] for v in voci)}

    # Ollama: strumento esterno (non pip) per il suggerimento visivo VLM
    try:
        from .vlm import stato_ollama
        ollama = stato_ollama()
    except Exception:
        ollama = {"disponibile": False, "modelli": []}

    return {
        "python": {
            "versione": ".".join(map(str, sys.version_info[:3])),
            "ok": sys.version_info[:2] >= PYTHON_MIN,
            "minimo": ".".join(map(str, PYTHON_MIN)),
        },
        "base": base,
        "base_ok": all(v["ok"] for v in base),
        "opzionali": opzionali,
        "ollama": ollama,
    }


def _riga(v: dict) -> str:
    stato = " ok  " if v["ok"] else "MANCA"
    ver = f" {v['versione']}" if v["ok"] and v["versione"] else ""
    return f"    [{stato}] {v['modulo']}{ver} - {v['uso']}"


def stampa_report(rep: dict | None = None) -> bool:
    """Stampa il report leggibile. Ritorna True se la base e' a posto."""
    rep = rep or controlla_ambiente()
    print("mepseg - controllo ambiente\n")

    p = rep["python"]
    esito = "ok" if p["ok"] else f"TROPPO VECCHIO (serve >= {p['minimo']})"
    print(f"Python {p['versione']}  [{esito}]\n")

    print("Dipendenze base (obbligatorie):")
    for v in rep["base"]:
        print(_riga(v))
    if not rep["base_ok"]:
        mancanti = " ".join(v["pip"] for v in rep["base"] if not v["ok"])
        print(f"    -> installa la base:  pip install -e .   (mancano: {mancanti})")

    print("\nGruppi opzionali (installali solo se ti servono):")
    for nome, g in rep["opzionali"].items():
        esito = "ok" if g["completo"] else "incompleto"
        print(f"\n  {nome}  [{esito}]")
        for v in g["voci"]:
            print(_riga(v))
        if not g["completo"]:
            print(f"    -> pip install -e .[{g['extra']}]")

    o = rep["ollama"]
    print("\nStrumento esterno opzionale - Ollama (suggerimento visivo, "
          "solo GUI):")
    if o["disponibile"]:
        modelli = ", ".join(o["modelli"]) or "nessuno (fai: ollama pull moondream:1.8b)"
        print(f"    [ ok  ] Ollama in ascolto su 127.0.0.1:11434 - modelli: {modelli}")
    else:
        print("    [ off ] Ollama non in esecuzione (opzionale). Installalo da")
        print("            https://ollama.com, poi:  ollama pull moondream:1.8b")

    print()
    if p["ok"] and rep["base_ok"]:
        print("Base a posto: mepseg puo' girare. I gruppi opzionali sopra "
              "abilitano E57, deep learning e interfaccia grafica.")
    else:
        print("Manca qualcosa di obbligatorio: vedi le righe MANCA qui sopra.")
    return p["ok"] and rep["base_ok"]


def principale(argv: list[str] | None = None) -> int:
    return 0 if stampa_report() else 1


if __name__ == "__main__":
    raise SystemExit(principale())
