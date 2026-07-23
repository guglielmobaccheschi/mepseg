"""Motore a regole per la classificazione dei cluster MEP.

Le regole sono valutate in ordine di specificita'. Tutte le soglie sono
configurabili da ``config/default.yaml`` (sezione ``regole``): i valori di
default riflettono dimensioni tipiche di impianti in edifici civili/terziario
e locali tecnici.
"""
from __future__ import annotations

import logging

import numpy as np

from .classi import ClasseMEP
from .primitive import FeatureCluster

log = logging.getLogger("mepseg")


def classi_escluse(cfg: dict) -> set[ClasseMEP]:
    """Legge da ``regole.classi_escluse`` le classi impossibili nel contesto.

    Esempio: in un data center non esistono sprinkler (l'acqua distruggerebbe
    le macchine); escludendo la classe, i piccoli elementi a soffitto che ne
    ricordano la forma scivolano alle regole successive invece di generare
    falsi positivi.
    """
    escluse = set()
    for nome in cfg.get("classi_escluse") or []:
        try:
            escluse.add(ClasseMEP[str(nome).strip().upper()])
        except KeyError:
            valide = ", ".join(c.name.lower() for c in ClasseMEP)
            raise ValueError(
                f"Classe da escludere sconosciuta: '{nome}'. Valide: {valide}"
            ) from None
    return escluse


def classifica_cluster(f: FeatureCluster, cfg: dict) -> ClasseMEP:
    r = cfg  # alias breve
    escluse = classi_escluse(cfg)

    # --- 0. Sbarramento di evidenza: nessuna etichetta impiantistica senza
    #     prova che il cluster sia superficie reale. Un cluster rarefatto
    #     rispetto al proprio ingombro (nuvolette di rumore, riflessi,
    #     code di scansione) puo' solo essere waste o generico: le regole
    #     sotto non vengono nemmeno interpellate, cosi' non "trovano"
    #     sprinkler o luci dove c'e' solo rumore della taglia giusta.
    if f.densita_superficie < float(r.get("min_densita_mep", 0.05)):
        volume_obb = float(f.dimensioni[0] * f.dimensioni[1] * f.dimensioni[2])
        if volume_obb >= float(r.get("volume_min_waste", 0.20)):
            return ClasseMEP.WASTE
        return ClasseMEP.MEP_GENERICO

    # rumore SOTTO il pavimento locale (riflessi attraverso il pavimento
    # flottante, code di scansione oltre la soletta): mai un impianto
    quota_top_su_pav = f.dist_da_pavimento + (f.quota_top - f.quota_bottom)
    if quota_top_su_pav < -0.05:
        return ClasseMEP.WASTE

    def basta_evidenza(nome_classe: str, minimo: int) -> bool:
        return f.n_punti >= int(r.get(f"min_punti_{nome_classe}", minimo))

    # --- 1. Sprinkler: piccolo, compatto, appeso vicino al soffitto o
    #        staccato da una tubazione (calate e teste sprinkler).
    if (
        ClasseMEP.SPRINKLER not in escluse
        and basta_evidenza("sprinkler", 20)
        and f.dimensioni[0] <= float(r.get("dim_max_sprinkler", 0.35))
        and (
            f.sotto_tubazione
            or f.dist_da_soffitto <= float(r.get("dist_soffitto_sprinkler", 1.0))
        )
    ):
        return ClasseMEP.SPRINKLER

    cil = f.cilindro

    # --- 1bis. Pilastro strutturale: verticale, attraversa il locale dal
    #     pavimento al soffitto, sezione compatta. Valutato PRIMA dei
    #     cilindri: un pilastro tondo ha un fit cilindrico perfetto e
    #     diventerebbe "condotta"; un cilindro sottile (montante) invece
    #     resta una tubazione.
    if (
        f.verticalita_asse >= 0.7
        and f.dist_da_soffitto <= float(r.get("dist_soffitto_pilastro", 0.4))
        and f.dist_da_pavimento <= float(r.get("dist_pavimento_pilastro", 0.4))
        and f.estensione_z_robusta >= float(r.get("altezza_min_pilastro", 2.0))
        and float(r.get("lato_min_pilastro", 0.2))
        <= f.dimensioni[1]
        <= float(r.get("lato_max_pilastro", 1.2))
        and not (
            cil.valido
            and cil.raggio <= float(r.get("raggio_max_tubazione", 0.10))
        )
    ):
        return ClasseMEP.PILASTRO

    cilindrico = (
        cil.valido
        and cil.inlier_ratio >= float(r.get("min_inlier_cilindro", 0.55))
        and cil.coerenza_radiale >= float(r.get("min_coerenza_radiale", 0.60))
        and cil.sezione_su_raggio >= float(r.get("min_sezione_su_raggio", 0.75))
        and cil.concentrazione_normali
        <= float(r.get("max_concentrazione_normali", 0.55))
        and f.linearita >= float(r.get("min_linearita_cilindro", 0.5))
        and cil.lunghezza >= float(r.get("lunghezza_min_cilindro", 0.4))
    )

    # --- 2. Elementi cilindrici estesi: tubazioni e condotte circolari.
    if cilindrico:
        if (
            ClasseMEP.TUBAZIONE not in escluse
            and basta_evidenza("tubazione", 30)
            and cil.raggio <= float(r.get("raggio_max_tubazione", 0.10))
        ):
            return ClasseMEP.TUBAZIONE
        if (
            ClasseMEP.CONDOTTA_CIRCOLARE not in escluse
            and basta_evidenza("condotta", 80)
            and cil.raggio <= float(r.get("raggio_max_condotta", 0.60))
            # una condotta grande appoggiata al pavimento non esiste:
            # e' un armadio/guscio curvo (le condotte corrono in quota)
            and not (
                cil.raggio >= float(r.get("raggio_condotta_sospetta", 0.35))
                and f.dist_da_pavimento
                <= float(r.get("dist_pavimento_min_condotta", 0.2))
            )
        ):
            return ClasseMEP.CONDOTTA_CIRCOLARE

    orizzontale = f.verticalita_asse <= float(r.get("max_verticalita_orizzontale", 0.35))
    elongato = f.dimensioni[0] >= float(r.get("elongazione_min", 2.0)) * max(
        f.dimensioni[1], 1e-6
    )
    vicino_soffitto = f.dist_da_soffitto <= float(r.get("dist_soffitto_impianti", 1.5))

    # --- 3. Passerella cavi: elemento lungo, orizzontale, a sezione bassa
    #        e stretta, sospeso vicino al soffitto.
    if (
        ClasseMEP.PASSERELLA_CAVI not in escluse
        and basta_evidenza("passerella", 80)
        and elongato
        and orizzontale
        and vicino_soffitto
        and f.dimensioni[0] >= float(r.get("lunghezza_min_passerella", 2.0))
        and f.dimensioni[2] <= float(r.get("altezza_max_passerella", 0.15))
        and float(r.get("larghezza_min_passerella", 0.10))
        <= f.dimensioni[1]
        <= float(r.get("larghezza_max_passerella", 0.70))
    ):
        return ClasseMEP.PASSERELLA_CAVI

    # --- 3bis. Passerella CARICA di cavi: i cavi che debordano alzano la
    #     sezione oltre il limite "nudo" di 0.15 e la facevano cadere nella
    #     condotta rettangolare. La banda alzata (0.15-0.30) pero' e' anche
    #     quella delle batterie di neon: per non rubare LUCI (ne' condotte
    #     vere, che sono piu' larghe/alte) qui il nastro deve essere
    #     inequivocabile — piu' lungo e piu' allungato della via normale.
    if (
        ClasseMEP.PASSERELLA_CAVI not in escluse
        and basta_evidenza("passerella", 80)
        and orizzontale
        and vicino_soffitto
        and f.dimensioni[0]
        >= float(r.get("elongazione_min_passerella_carica", 3.0))
        * max(f.dimensioni[1], 1e-6)
        and f.dimensioni[0] >= float(r.get("lunghezza_min_passerella_carica", 2.5))
        and f.dimensioni[2] <= float(r.get("altezza_max_passerella_carica", 0.30))
        and float(r.get("larghezza_min_passerella_carica", 0.20))
        <= f.dimensioni[1]
        <= float(r.get("larghezza_max_passerella", 0.70))
    ):
        return ClasseMEP.PASSERELLA_CAVI

    # --- 4. Condotta rettangolare: lunga, orizzontale, sezione rettangolare
    #        di dimensioni impiantistiche tipiche.
    if (
        ClasseMEP.CONDOTTA_RETTANGOLARE not in escluse
        and basta_evidenza("condotta", 80)
        and elongato
        and orizzontale
        and vicino_soffitto
        and float(r.get("larghezza_min_condotta_rett", 0.20))
        <= f.dimensioni[1]
        <= float(r.get("larghezza_max_condotta_rett", 1.50))
        and float(r.get("altezza_min_condotta_rett", 0.12))
        <= f.dimensioni[2]
        <= float(r.get("altezza_max_condotta_rett", 1.20))
    ):
        return ClasseMEP.CONDOTTA_RETTANGOLARE

    # --- 5. Elementi piatti a soffitto: luci e terminali aria.
    # "piatto" = spessore piccolo in assoluto e rispetto alla larghezza
    # (la planarita' PCA non funziona per oggetti piatti ma allungati).
    piatto = f.dimensioni[2] <= float(
        r.get("spessore_max_piatto", 0.25)
    ) and f.dimensioni[2] <= 0.5 * f.dimensioni[1]
    # ...e ORIZZONTALE: una lastra inclinata che tocca il soffitto con un
    # bordo (rampa di cavi, pannello) e' piatta per l'OBB ma si sviluppa in
    # verticale. L'estensione robusta ignora pendini e cavi di sospensione.
    orizzontale_piatto = f.estensione_z_robusta <= float(
        r.get("estensione_z_max_piatto", 0.4)
    ) or f.frazione_z_alta >= float(r.get("frazione_alta_min_piatto", 0.5))
    if (
        piatto
        and orizzontale_piatto
        and f.dist_da_soffitto <= float(r.get("dist_soffitto_piatti", 0.6))
        and f.dimensioni[0] <= float(r.get("dim_max_piatto", 2.0))
        and f.dimensioni[0] >= float(r.get("dim_min_piatto", 0.15))
        # anti-riflesso: il bagliore e' rado, una plafoniera e' piena
        and f.densita_superficie >= float(r.get("min_densita_luce", 0.08))
    ):
        rapporto = f.dimensioni[0] / max(f.dimensioni[1], 1e-6)
        # i terminali aria (diffusori/griglie) sono tipicamente quasi quadrati,
        # i corpi illuminanti lineari sono allungati.
        if (
            ClasseMEP.TERMINALE_ARIA not in escluse
            and basta_evidenza("terminale", 25)
            and rapporto <= float(r.get("rapporto_max_terminale", 1.6))
            and f.dimensioni[0] <= float(r.get("dim_max_terminale", 0.9))
        ):
            return ClasseMEP.TERMINALE_ARIA
        if ClasseMEP.LUCE not in escluse and basta_evidenza("luce", 30):
            return ClasseMEP.LUCE

    # --- 5bis. Lastra verticale strutturale: fascia di muro sopra i rack,
    #      tramezzo occluso, serramento — grande, sottile e densa. La fase
    #      strutturale la boccia quando la base e' occlusa (non "incontra"
    #      il pavimento), ma un muro resta un muro: senza questa regola
    #      finirebbe in Apparecchiatura per pura taglia.
    if (
        f.verticalita_asse <= 0.35  # asse lungo orizzontale (fascia)
        or f.verticalita_asse >= 0.7  # oppure sviluppo verticale
    ) and (
        f.dimensioni[2] <= float(r.get("spessore_max_lastra_parete", 0.15))
        and f.dimensioni[0] >= float(r.get("lato_min_lastra_parete", 1.5))
        and f.dimensioni[1] >= 0.5
        and f.estensione_z_robusta >= float(r.get("altezza_min_lastra_parete", 0.8))
        and f.planarita + f.linearita >= 0.7
        and f.densita_superficie >= 0.10
        and f.dimensioni[0] * f.dimensioni[1]
        >= float(r.get("area_min_lastra_parete", 2.0))
    ):
        return ClasseMEP.PARETE

    # --- 6. Apparecchiatura: volume compatto e consistente (pompe, UTA,
    #        quadri, unita' a blocco), tipicamente a pavimento o a parete.
    #        L'ingombro da solo non basta: serve EVIDENZA che la scatola sia
    #        piena di superficie reale (densita' rispetto al guscio OBB),
    #        altrimenti qualunque sbuffo di rumore esteso diventerebbe
    #        un'apparecchiatura.
    voluminoso = (
        f.dimensioni[2] >= float(r.get("dim_min_apparecchiatura", 0.35))
        and f.dimensioni[0] >= float(r.get("dim_max_lato_apparecchiatura", 0.60))
    )
    volume_obb = float(f.dimensioni[0] * f.dimensioni[1] * f.dimensioni[2])
    consistente = f.densita_superficie >= float(
        r.get("min_densita_apparecchiatura", 0.10)
    )
    # una macchina sta a terra (o su basamento): i grumi sospesi a
    # mezz'aria sono fasci di cavi o pezzi di fascia soffitto, non
    # apparecchiature — restano generico per il finder/DL
    appoggiata = f.dist_da_pavimento <= float(
        r.get("appoggio_max_apparecchiatura", 0.5)
    )
    if (
        ClasseMEP.APPARECCHIATURA not in escluse
        and basta_evidenza("apparecchiatura", 200)
        and consistente
        and appoggiata
        and (
            voluminoso
            or volume_obb >= float(r.get("volume_min_apparecchiatura", 0.40))
        )
    ):
        return ClasseMEP.APPARECCHIATURA

    # --- 7. Nessuna regola soddisfatta: elemento MEP generico.
    return ClasseMEP.MEP_GENERICO


def applica_guida_colore(istanze: list, etichette_r, guida: list[dict], cfg: dict) -> int:
    """Spareggio per COLORE (opt-in): riclassifica le istanze MEP_GENERICO il
    cui colore dominante combacia con una voce della guida colore dell'utente.

    E' deliberatamente SECONDARIO e RISTRETTO:
      - agisce solo su MEP_GENERICO (il bucket onesto), mai sovrascrive una
        classe decisa dalla geometria — il colore non deve inventare condotte;
      - richiede colore UNIFORME (un cluster cromaticamente misto non e' un
        indizio affidabile e viene saltato);
      - la guida codifica la convenzione cromatica dell'IMPIANTO SPECIFICO
        (es. canaline dati gialle, blindosbarre blu), quindi va fornita solo
        per nuvole che seguono quel codice — vuota di default, il tool resta
        agnostico. Ogni voce: ``{classe, rgb:[r,g,b], tolleranza?}``.
    Ritorna il numero di istanze riclassificate.
    """
    if not guida:
        return 0
    tol_default = float(cfg.get("tolleranza_guida_colore", 45.0))
    min_unif = float(cfg.get("min_uniformita_guida", 0.6))
    n = 0
    for ist in istanze:
        if ist.classe != ClasseMEP.MEP_GENERICO:
            continue
        f = ist.feature
        if f.colore_dominante is None or f.uniformita_colore is None:
            continue
        if f.uniformita_colore < min_unif:
            continue  # colore troppo misto: indizio non affidabile
        dom = np.asarray(f.colore_dominante, dtype=np.float32)
        for voce in guida:
            tol = float(voce.get("tolleranza", tol_default))
            if np.linalg.norm(dom - np.asarray(voce["rgb"], dtype=np.float32)) <= tol:
                classe = ClasseMEP[str(voce["classe"]).strip().upper()]
                ist.classe = classe
                if ist.indici is not None:
                    etichette_r[ist.indici] = int(classe)
                n += 1
                break
    if n:
        log.info("Guida colore: %d istanze generiche riclassificate per colore", n)
    return n
