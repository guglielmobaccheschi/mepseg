"""Generazione dei report JSON e CSV per istanza."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .classi import ClasseMEP, NOMI_CLASSI
from .pipeline import RisultatoSegmentazione


def _riga_istanza(istanza) -> dict:
    f = istanza.feature
    riga = {
        "id": istanza.id,
        "classe_codice": int(istanza.classe),
        "classe": NOMI_CLASSI[istanza.classe],
        "n_punti": f.n_punti,
        "centro_x": round(float(f.centro[0]), 3),
        "centro_y": round(float(f.centro[1]), 3),
        "centro_z": round(float(f.centro[2]), 3),
        "dim_1": round(float(f.dimensioni[0]), 3),
        "dim_2": round(float(f.dimensioni[1]), 3),
        "dim_3": round(float(f.dimensioni[2]), 3),
        "quota_top": round(f.quota_top, 3),
        "quota_bottom": round(f.quota_bottom, 3),
        # firma per la ricerca per somiglianza (e per scegliere i modelli)
        "estensione_z": round(float(f.estensione_z_robusta), 3),
        "dist_soffitto": round(float(f.dist_da_soffitto), 3),
        "dist_pavimento": round(float(f.dist_da_pavimento), 3),
        "verticalita": round(float(f.verticalita_asse), 3),
    }
    # colore (solo se la nuvola ha RGB): visibile per istanza nel report,
    # utile all'occhio, al suggerimento VLM e al fine-tuning
    if getattr(f, "colore_dominante", None) is not None:
        riga["colore_rgb"] = [int(v) for v in f.colore_dominante]
        riga["uniformita_colore"] = round(float(f.uniformita_colore), 3)
    if f.cilindro.valido and istanza.classe in (
        ClasseMEP.TUBAZIONE,
        ClasseMEP.CONDOTTA_CIRCOLARE,
    ):
        riga.update(
            {
                "raggio_stimato": round(f.cilindro.raggio, 4),
                "diametro_stimato": round(2.0 * f.cilindro.raggio, 4),
                "lunghezza_stimata": round(f.cilindro.lunghezza, 3),
                "asse_x": round(float(f.cilindro.asse[0]), 4),
                "asse_y": round(float(f.cilindro.asse[1]), 4),
                "asse_z": round(float(f.cilindro.asse[2]), 4),
            }
        )
    return riga


def genera_report(
    risultato: RisultatoSegmentazione, cartella: str | Path, nome_base: str
) -> tuple[Path, Path]:
    """Scrive ``<nome_base>_report.json`` e ``<nome_base>_istanze.csv``."""
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)

    etichette = risultato.etichette
    conteggi = {
        NOMI_CLASSI[classe]: int((etichette == int(classe)).sum())
        for classe in ClasseMEP
    }
    righe = [_riga_istanza(ist) for ist in risultato.istanze]

    report = {
        "punti_totali": int(len(etichette)),
        "quota_soffitto": round(risultato.quota_soffitto, 3),
        "quota_pavimento": round(risultato.quota_pavimento, 3),
        "punti_per_classe": conteggi,
        "piani_strutturali": risultato.piani_struttura,
        "n_istanze_mep": len(righe),
        "istanze": righe,
    }
    percorso_json = cartella / f"{nome_base}_report.json"
    percorso_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # punti (nuvola ridotta) con l'id dell'istanza di appartenenza: serve
    # alla selezione per click nel viewer della Fase 4 (il server risale
    # dall'istanza al punto piu' vicino via KD-tree). Solo i punti che
    # appartengono a un'istanza MEP: struttura e rumore non sono cliccabili.
    if risultato.punti_ridotti is not None and risultato.istanze:
        id_istanza = np.zeros(len(risultato.punti_ridotti), dtype=np.uint32)
        for ist in risultato.istanze:
            if ist.indici is not None:
                id_istanza[ist.indici] = ist.id
        con_istanza = id_istanza > 0
        np.savez(
            cartella / f"{nome_base}_istanze_punti.npz",
            punti=risultato.punti_ridotti[con_istanza].astype(np.float32),
            id_istanza=id_istanza[con_istanza],
        )

    percorso_csv = cartella / f"{nome_base}_istanze.csv"
    campi = [
        "id", "classe_codice", "classe", "n_punti",
        "centro_x", "centro_y", "centro_z",
        "dim_1", "dim_2", "dim_3", "quota_top", "quota_bottom",
        "estensione_z", "dist_soffitto", "dist_pavimento", "verticalita",
        "raggio_stimato", "diametro_stimato", "lunghezza_stimata",
        "asse_x", "asse_y", "asse_z", "uniformita_colore",
    ]
    with open(percorso_csv, "w", newline="", encoding="utf-8") as f:
        # extrasaction: colore_rgb (lista) resta solo nel JSON, non nel CSV
        writer = csv.DictWriter(
            f, fieldnames=campi, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(righe)

    return percorso_json, percorso_csv
