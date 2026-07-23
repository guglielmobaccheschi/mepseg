# mepseg — Segmentazione di nuvole di punti per impianti MEP

Pipeline **scan-to-BIM** per il riconoscimento automatico degli impianti
meccanici, elettrici e idraulici (MEP) in nuvole di punti da laser scanner.
Il tool etichetta e colora ogni punto e produce un
report per istanza (diametri, assi, dimensioni) come base per la successiva
modellazione BIM.

L'approccio è **ibrido**:

1. **Pipeline geometrica** (sempre attiva, nessun training richiesto):
   rimozione degli elementi strutturali con RANSAC, clustering per continuità
   di superficie (con **taglio per colore** quando la nuvola ha RGB, per
   separare oggetti attaccati dello stesso materiale), fitting di primitive
   (cilindri, box orientati) e classificazione a regole con soglie
   dimensionali tipiche dell'impiantistica.
2. **Deep learning opzionale**: rete neurale per punti (implementazione
   PyTorch propria; architettura dalla letteratura, si veda la sezione
   Riferimenti) per raffinare le zone che la geometria non riesce a
   classificare. Funziona su GPU NVIDIA (CUDA) o CPU; è pensata per il
   fine-tuning su scansioni proprie annotate.

## Classi e colori

| Codice | Classe | Colore RGB |
|-------:|--------|------------|
| 0 | Struttura / non-MEP (generica) | grigio (128,128,128) |
| 1 | Condotta circolare | azzurro (0,150,255) |
| 2 | Condotta rettangolare | blu (0,60,220) |
| 3 | Tubazione | verde (0,200,80) |
| 4 | Passerella cavi | arancione (255,140,0) |
| 5 | Luce | giallo (255,220,0) |
| 6 | Sprinkler | rosso (255,0,60) |
| 7 | Terminale aria | ciano (0,230,230) |
| 8 | Apparecchiatura | viola (170,0,255) |
| 9 | Elemento MEP generico | rosa (255,105,180) |
| 10 | Arredo (oggetti reali non-MEP) | marrone (139,87,42) |
| 11 | Pavimento | grigio scuro (105,105,105) |
| 12 | Soffitto | grigio chiaro (210,210,210) |
| 13 | Parete | grigio medio (160,160,160) |
| 14 | Pilastro | grigio scurissimo (80,80,80) |
| 15 | Trave | grigio-blu (110,110,140) |
| 16 | Waste (rumore di scansione) | nero morbido (40,40,45) |

## Installazione

Requisiti: **Python 3.9–3.12** (consigliato 3.11). Testato su Windows;
funziona anche su Linux e macOS.

```bash
git clone https://github.com/guglielmobaccheschi/mepseg.git
cd mepseg
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

Dipendenze installate automaticamente: `numpy`, `scipy`, `scikit-learn`,
[`open3d`](https://www.open3d.org/) (normali, RANSAC, DBSCAN),
[`laspy`](https://laspy.readthedocs.io/) (LAS/LAZ), `pyyaml`.

Moduli opzionali:

```bash
pip install -e .[e57]    # lettura file E57 (pye57)
pip install -e .[dl]     # deep learning (PyTorch)
pip install -e .[gui]    # interfaccia web locale (mepseg-gui)
pip install -e .[tutto]  # tutto
```

Per verificare in qualsiasi momento **cosa è installato e cosa manca**
(Python, dipendenze base e opzionali, Ollama) con i comandi per installare
ciò che serve:

```bash
mepseg --controlla
```

### Interfaccia grafica — il modo consigliato di usare mepseg

**mepseg è pensato per essere usato dalla sua interfaccia grafica**: tutto il
flusso di lavoro (preparazione, elaborazione, verifica, conferma e
addestramento) vive lì. Su Windows il modo normale di avviarlo è un
**doppio click su `avvia_gui.bat`** nella cartella del progetto: fa partire il
server locale e apre il browser sull'interfaccia. Non serve la riga di comando.

```bash
# equivalente da terminale (Linux/macOS, o se preferisci la shell):
mepseg-gui              # avvia il server locale e apre il browser
```

> ⚠️ Non aprire `mepseg/gui/static/index.html` direttamente dal disco:
> l'interfaccia ha bisogno del suo server locale (la pagina stessa, se
> aperta cosi', mostra le istruzioni di avvio corrette).

Flusso in cinque fasi, tutto in locale (nessun dato lascia il computer):

1. **Prepara** — scegli il file (selettore nativo), il livello di dettaglio
   e il tipo di ambiente (i preset escludono le classi impossibili, es.
   niente sprinkler in un data center). L'anteprima genera la **pianta** con
   una griglia nei metri reali dello scanner e una **vista laterale**
   (elevazione) per scegliere la quota Z; disegni sopra il rettangolo della
   zona di lavoro (bbox + margine di contesto). All'avvio dell'elaborazione
   la preparazione si blocca sui parametri del run.
2. **Elabora** — avanzamento per fase e registro in tempo reale.
3. **Verifica** — viewer 3D colorato per classe con legenda filtrabile e
   tabella delle istanze: un click su una riga evidenzia e inquadra
   l'oggetto nella vista.
4. **Conferma e allinea** — conferma, correggi o elimina le istanze per
   click; un run di allineamento propaga le conferme a tutta la nuvola.
   Suggerimento visivo opzionale da un modello locale (vedi sotto).
5. **Addestra** — esporta le etichette nel dataset cumulativo e fai il
   fine-tuning della rete, che nei run successivi affina il MEP generico.

> **GPU NVIDIA**: per usare CUDA installare PyTorch seguendo
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
> prima di `pip install -e .[dl]`. Senza GPU tutto funziona comunque
> (la parte DL gira su CPU, la pipeline geometrica non usa PyTorch).

### Suggerimento visivo (opzionale, richiede Ollama)

Nel pannello di conferma la GUI puo' chiedere un parere a un modello
vision-language **locale**: l'istanza selezionata viene renderizzata in tre
viste ortogonali e il modello propone una classe con una breve motivazione.
E' solo un suggerimento in piu' — la decisione resta sempre all'utente e
nessuna etichetta viene mai cambiata dal modello.

Prerequisito **opzionale**: [Ollama](https://ollama.com) in esecuzione
sulla stessa macchina (server locale su `127.0.0.1:11434`, nessun dato
lascia il computer). Non e' un pacchetto pip: va installato a parte, poi:

```bash
ollama pull moondream:1.8b   # modello minuscolo, va bene anche su CPU
# alternativa di qualita' migliore: ollama pull qwen2.5vl:3b
```

Se Ollama non e' installato o non e' in esecuzione, il bottone del
suggerimento resta disabilitato con una nota e tutto il resto della GUI
funziona normalmente. Su CPU l'inferenza richiede da qualche secondo a
un minuto per istanza (i pesi occupano 1.5-3+ GB su disco).

## Uso da riga di comando (avanzato / automazione)

Il modo normale di lavorare è la **GUI** (vedi sopra: `avvia_gui.bat`). La riga
di comando è lo stesso motore che la GUI richiama, ed è utile per automazione,
script ed elaborazioni batch.

```bash
# prova immediata su una scena sintetica inclusa
python esempi/genera_scena_demo.py
mepseg esempi/scena_demo.ply -o output_demo

# su una scansione reale
mepseg rilievo.e57 -o output --las
mepseg rilievo.las --voxel 0.02
mepseg rilievo.ply -c mia_config.yaml --pesi modello_mep.pth

# contesto di scena: classi impossibili nell'ambiente rilevato
# (es. data center: gli sprinkler non esistono, meglio non cercarli)
mepseg sala_dati.e57 --escludi sprinkler,terminale_aria

# ritaglio: scarta i punti residui di scansioni adiacenti fuori dal locale
mepseg locale.e57 --bbox 0.5,17.5,-2.0,4.1,27.0,3.1

# ritaglio con contesto: analizza 1 m oltre il box (tubi passanti,
# ancoraggi alle pareti) ma esporta solo l'interno del box
mepseg locale.e57 --bbox 0.5,17.5,-2.0,4.1,27.0,3.1 --margine 1.0

# "trova simili": salva la firma geometrica di un'istanza confermata
# (l'ID si legge nel report/CSV di un run precedente) e riusala per
# riclassificare le istanze dubbie che le somigliano
mepseg rilievo.e57 --modelli modelli.json --salva-modello 90:plafoniera:luce
mepseg rilievo.e57 --modelli modelli.json
```

### Nuvole enormi (streaming)

I file E57 sopra i 20 milioni di punti (soglia `streaming.soglia_punti` in
configurazione) vengono elaborati **a blocchi senza caricare tutto in RAM**:
prima passata di lettura + sottocampionamento voxel incrementale su griglia
globale, segmentazione sulla nuvola ridotta, e (con `--las`) seconda passata
di rilettura che propaga le etichette e scrive il LAS classificato a blocchi.
Testato su nuvole da centinaia di milioni di punti su macchine con 16 GB di
RAM. In modalita' streaming il PLY di output e' la nuvola ridotta
(`*_segmentata_ridotta.ply`); il LAS resta a piena risoluzione. Se l'E57 ha
colore, viene mediato per voxel e portato fino al clustering (cache colore
affiancata `*_ridotta_<mm>mm_rgb.npy`): le cache generate da versioni
precedenti sono senza colore e vanno rigenerate per abilitare il taglio
per colore.

Formati di ingresso: `.e57`, `.las`, `.laz`, `.ply`, `.pcd`, `.xyz`/`.txt`.

### Output

| File | Contenuto |
|------|-----------|
| `<nome>_segmentata.ply` | nuvola a piena risoluzione, colorata per classe, con scalar field `classe` (leggibile in CloudCompare) |
| `<nome>_segmentata.las` | (con `--las`) LAS 1.4 con RGB e campo `classification` |
| `<nome>_report.json` | conteggi per classe, piani strutturali, istanze con dimensioni, assi e diametri stimati |
| `<nome>_istanze.csv` | le stesse istanze in formato tabellare |

## Come funziona

```
nuvola (E57/LAS/PLY/...)
  │ 1. sottocampionamento voxel + rimozione outlier + stima normali
  │ 2. RANSAC iterativo → PAVIMENTO / SOFFITTO / PARETE, ciascuno con
  │    la propria etichetta (criteri dimensionali evitano di scambiare
  │    le condotte per soffitti)
  │ 3. clustering per CONTINUITA' DI SUPERFICIE (edge detection 3D: bordi
  │    di salto e di piega con criterio di convessita' alla LCCP; DBSCAN
  │    resta disponibile) → cluster candidati MEP. Con nuvole a COLORE,
  │    una forte discontinuita' RGB separa oggetti attaccati dello stesso
  │    materiale (es. rack grigio / passerella gialla / blindosbarra blu) che
  │    la geometria fonderebbe; i cluster fusi con ASSI DIVERSI si
  │    dividono per orientamento (RANSAC sulle normali)
  │ 4. per ogni cluster: PCA, box orientata, fitting cilindro (asse dalle
  │    normali + cerchio ai minimi quadrati) e — se c'e' RGB — colore
  │    dominante e uniformita'
  │ 5. classificazione a regole (diametri, sezioni, quota dal soffitto, ecc.)
  │    + raffinamento delle appendici (calate sprinkler, valvole)
  │    + regole di contesto (gli impianti sono una rete): file di neon
  │      sotto il soffitto = luci; tubi/condotte con le estremita' nel
  │      vuoto = arredo/non-MEP; ecc.
  │    + [opt-in] spareggio per colore sui casi rimasti "generico"
  │ 6. [opzionale] fusione con la rete (DL locale) dove la geometria dice "generico"
  │ 7. propagazione delle etichette alla nuvola a piena risoluzione
  ▼
nuvola etichettata + colorata + report JSON/CSV
```

Tutte le soglie (diametri massimi, distanze dal soffitto, dimensioni
tipiche di passerelle e condotte…) sono in
[`config/default.yaml`](config/default.yaml): copiare il file, adattarlo al
proprio caso e passarlo con `-c`.

## Fine-tuning su scansioni proprie

Il modulo DL non richiede dataset pubblici: si parte dalla pipeline
geometrica e si migliora con i propri dati.

1. Segmentare le scansioni con `mepseg` (solo geometria) e confermare o
   correggere le istanze nella GUI (fase "Conferma e allinea").
2. Esportare le etichette nel dataset cumulativo (dalla GUI, fase
   "Addestra", oppure `python -m mepseg.dl.esporta cartella_run -o dataset_mep`).
3. Addestrare:

   ```bash
   python -m mepseg.dl.allenamento dataset_mep -o pesi/rete_mep_v1.pth --epoche 50
   ```

4. Usare il modello: `mepseg rilievo.e57 --pesi pesi/rete_mep_v1.pth`
   (o la spunta "usa la rete" nella GUI).

## Limiti noti e consigli

- La pipeline geometrica assume scansioni **registrate e in metri**, con
  l'asse Z verticale.
- Elementi MEP **a contatto tra loro** possono finire nello stesso cluster;
  aiutano il clustering per continuità di superficie e, su nuvole a colore,
  il taglio per discontinuità RGB; ridurre `preprocess.voxel` su scansioni
  dense migliora ulteriormente la separazione.
- Piccoli terminali richiedono nuvole sufficientemente dense
  (passo ≲ 2 cm in zona soffitto); con `voxel: 0.03` gli oggetti sotto
  ~5 cm tendono a sparire — usare `--voxel 0.015` in ambienti ricchi di
  dettagli.
- Il riconoscimento è **semantico, non parametrico**: il report fornisce
  diametri e assi dei cilindri come base, ma la ricostruzione BIM vera e
  propria (IFC o BIM Authoring proprietario) è un passo successivo.

## Riferimenti

- L'architettura della rete di segmentazione (modulo `dl`, implementazione
  propria in PyTorch) e' descritta in: Q. Hu, B. Yang, L. Xie, S. Rosa,
  Y. Guo, Z. Wang, N. Trigoni, A. Markham, *[RandLA-Net: Efficient Semantic
  Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net#randla-net-efficient-semantic-segmentation-of-large-scale-point-clouds-cvpr-2020)*, CVPR 2020.
- Elaborazione geometrica: [Open3D](https://www.open3d.org/).

Licenza: [MIT](LICENSE).
