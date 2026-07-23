"""Confronto PER POSIZIONE tra due run (lezione di metodo: mai per totali).

Ogni istanza del run A viene cercata nel run B per firma geometrica
(centro ±0.3 m + lati OBB ±20%, lo stesso matching del pin di Fase 4).
Il report dice, classe per classe, quante istanze sono STABILI (stessa
classe), CAMBIATE (in cosa), PERSE (footprint sparito/rimescolato) e
quante NUOVE compaiono in B. E' il metro con cui giudicare ogni modifica:
i totali per classe possono restare uguali mentre sotto si rimescola
tutto, o esplodere mentre gli oggetti veri sono stabili.

Uso:
  python scripts/confronta_run.py output_locale1_v10 output_locale1_v12
  python scripts/confronta_run.py A B --classe luce   (dettaglio una classe)
  python scripts/confronta_run.py A B --min-punti 300 (ignora le piccole)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mepseg.somiglianza import firma_da_riga, firme_coincidono  # noqa: E402


def carica_istanze(cartella: str) -> list[dict]:
    percorso = next(Path(cartella).glob("*_report.json"))
    report = json.loads(percorso.read_text(encoding="utf-8"))
    return report["istanze"]


def principale() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", help="cartella del run di riferimento")
    parser.add_argument("run_b", help="cartella del run da giudicare")
    parser.add_argument("--min-punti", type=int, default=100,
                        help="ignora le istanze piu' piccole (default 100)")
    parser.add_argument("--classe", help="dettaglio istanza per istanza di una classe (chiave, es. luce)")
    argomenti = parser.parse_args()

    ist_a = [i for i in carica_istanze(argomenti.run_a) if i["n_punti"] >= argomenti.min_punti]
    ist_b = [i for i in carica_istanze(argomenti.run_b) if i["n_punti"] >= argomenti.min_punti]
    firme_b = [(i, firma_da_riga(i)) for i in ist_b]

    esiti = defaultdict(Counter)   # classe A -> esito
    cambi = defaultdict(Counter)   # classe A -> classe B dei cambi
    dettagli = []
    abbinate_b = set()
    for a in ist_a:
        fa = firma_da_riga(a)
        trovata = None
        for b, fb in firme_b:
            if id(b) not in abbinate_b and firme_coincidono(fb, fa):
                trovata = b
                break
        if trovata is None:
            esiti[a["classe"]]["persa"] += 1
            dettagli.append((a, None))
            continue
        abbinate_b.add(id(trovata))
        if trovata["classe"] == a["classe"]:
            esiti[a["classe"]]["stabile"] += 1
        else:
            esiti[a["classe"]]["cambiata"] += 1
            cambi[a["classe"]][trovata["classe"]] += 1
            dettagli.append((a, trovata))

    nuove = Counter(
        b["classe"] for b, _ in firme_b if id(b) not in abbinate_b
    )

    n_a, n_b = len(ist_a), len(ist_b)
    n_stabili = sum(c["stabile"] for c in esiti.values())
    print(f"\nA = {argomenti.run_a} ({n_a} istanze >= {argomenti.min_punti} pti)")
    print(f"B = {argomenti.run_b} ({n_b} istanze)")
    print(f"STABILI: {n_stabili}/{n_a} ({100 * n_stabili / max(n_a, 1):.0f}%)\n")
    print(f"{'classe (in A)':24s} {'stabili':>8s} {'cambiate':>9s} {'perse':>6s}   verso")
    for classe in sorted(esiti, key=lambda c: -sum(esiti[c].values())):
        e = esiti[classe]
        verso = ", ".join(
            f"{n}x {c}" for c, n in cambi[classe].most_common(3)
        ) if cambi[classe] else ""
        print(f"{classe:24s} {e['stabile']:8d} {e['cambiata']:9d} {e['persa']:6d}   {verso}")
    if nuove:
        print("\nNUOVE in B (footprint senza corrispondente in A):")
        for classe, n in nuove.most_common():
            print(f"  {classe:24s} {n}")

    if argomenti.classe:
        chiave = argomenti.classe.strip().lower()
        print(f"\n--- dettaglio '{chiave}' (cambiate/perse) ---")
        for a, b in dettagli:
            if chiave not in a["classe"].lower().replace(" ", "_"):
                continue
            dove = f"({a['centro_x']:.1f}, {a['centro_y']:.1f}, {a['centro_z']:.1f})"
            cosa = b["classe"] if b else "PERSA"
            print(f"  id {a['id']:5d} {a['n_punti']:8,} pti {dove:24s} -> {cosa}")
    return 0


if __name__ == "__main__":
    sys.exit(principale())
