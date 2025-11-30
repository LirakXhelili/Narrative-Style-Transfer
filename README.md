## Detektimi Automatik i Ndryshimeve të Stilit në Tekstet Narrative

## University of Prishtina
<img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="110" align="right"/>

**Fakulteti i Inxhinierise Elektrike dhe Kompjuterike (FIEK)**  
**Program:** Computer and Software Engineering - Master  
**Course:** Procesimi i gjuhëve natyrale 

## Course Professor
**Prof. Dr. Mërgim Hoti**

##  Përshkrimi i Projektit

Qëllimi i këtij projekti është ndërtimi i një sistemi AI që automatikisht **detekton ndryshimet e stilit të shkrimit** në tekstet narrative. Sistemi identifikon katër lloje të ndryshimeve stilistike:

| Lloji i Ndryshimit | Përshkrimi |
|-------------------|------------|
| **NARRATOR_SHIFT** | Ndryshimi i perspektivës së tregimtarit (vetë e parë ↔ vetë e tretë) |
| **TENSE_SHIFT** | Ndryshimi i kohës gramatikore (e shkuar ↔ e tashme ↔ e ardhme) |
| **REGISTER_SHIFT** | Ndryshimi i regjistrit të gjuhës (formale ↔ joformale) |
| **EMOTION_SHIFT** | Ndryshimi i tonit emocional (pozitiv ↔ negativ) |

---

## Dataset-i

### Burimi i të Dhënave

Dataset-i është ndërtuar nga **10 vepra letrare klasike** të marra nga [Project Gutenberg](https://www.gutenberg.org/), një librari digjitale me libra falas në domenin publik.

### Veprat e Përdorura

| Nr. | Titulli | Autori | Madhësia |
|-----|---------|--------|----------|
| 1 | Alice's Adventures in Wonderland | Lewis Carroll | 174 KB |
| 2 | The Call of the Wild | Jack London | 200 KB |
| 3 | David Copperfield | Charles Dickens | 2.03 MB |
| 4 | Dracula | Bram Stoker | 887 KB |
| 5 | Great Expectations | Charles Dickens | 1.06 MB |
| 6 | Heart of Darkness | Joseph Conrad | 234 KB |
| 7 | The Picture of Dorian Gray | Oscar Wilde | 466 KB |
| 8 | The Secret Garden | Frances Hodgson Burnett | 462 KB |
| 9 | The Turn of the Screw | Henry James | 259 KB |
| 10 | White Fang | Jack London | 429 KB |

### Statistikat e Dataset-it

| Metrikë | Vlera |
|---------|-------|
| **Numri total i segmenteve** | 19,693 |
| **Madhësia totale e të dhënave të papërpunuara** | ~6.2 MB |
| **Numri i atributeve (features)** | 8 karakteristika numerike |
| **Etiketat (labels)** | 4 lloje ndryshimesh + 1 binary (has_transfer) |
| **Formati i ruajtjes** | JSONL (JSON Lines) |

### Atributet e Nxjerra (Features)

Për secilin segment teksti, nxirren këto karakteristika:

| Nr. | Feature | Përshkrimi |
|-----|---------|------------|
| 1 | `fp_ratio` | Raporti i përemrave vetë e parë (I, we, me, us, my, our) |
| 2 | `tp_ratio` | Raporti i përemrave vetë e tretë (he, she, they, him, her, them) |
| 3 | `verb_ratio` | Raporti i foljeve në segment |
| 4 | `neg_ratio` | Raporti i fjalëve me konotacion negativ emocional |
| 5 | `formal_ratio` | Raporti i fjalëve formale (therefore, however, moreover) |
| 6 | `informal_ratio` | Raporti i fjalëve joformale (gonna, wanna, kinda) |
| 7 | `avg_token_len` | Gjatësia mesatare e token-ave |
| 8 | `length` | Numri total i token-ave në segment |

---
## Arkitektura e Sistemit

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE E PËRPUNIMIT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Tekstet     │───▶│  Segmentimi  │───▶│  Auto-Label  │      │
│  │  Raw (.txt)  │    │  (spaCy)     │    │  (Heuristikë)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Modeli     │◀───│   Training   │◀───│  Preprocess  │      │
│  │   Final      │    │              │    │   (spaCy)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Struktura e Skedarëve

```
Narrative-Style-Transfer/
├── data/
│   ├── raw/                      # Tekstet origjinale (.txt)
│   ├── annotations/              # Dataset-i (JSONL)
│   │   ├── candidate_segments.jsonl
│   │   ├── candidate_segments_labeled.jsonl
│   │   └── narrative_cues.jsonl
│   └── processed/                # Të dhëna të përpunuara (.pkl)
│       └── narrative_cues.pkl
├── models/                       # Modelet e trajnuara
│   └── traditional_logreg.joblib
├── src/
│   ├── config.py                 # Konfigurimet dhe konstantet
│   ├── dataset.py                # Ngarkimi i dataset-it
│   ├── features.py               # Nxjerrja e features
│   ├── make_segments.py          # Segmentimi i teksteve
│   ├── auto_label.py             # Etiketimi automatik
│   ├── preprocess.py             # Përpunimi me spaCy
│   ├── train_traditional.py      # Trajnimi i Logistic Regression
│   ├── train_transformer.py      # Trajnimi i DistilBERT
│   └── inspect_traditional.py    # Analiza e peshave të modelit
├── requirements.txt
└── README.md
```
## Metodologjia

### 1. Përgatitja e të Dhënave

#### 1.1 Segmentimi i Teksteve

Tekstet e papërpunuara ndahen në segmente duke përdorur spaCy për sentence tokenization. Çdo segment përmban **3 fjali të njëpasnjëshme**, duke krijuar kontekst të mjaftueshëm për detektimin e ndryshimeve stilistike.

```python
# Nga make_segments.py
chunk_size = 3
for i in range(0, len(sentences), chunk_size):
    chunk_sents = sentences[i:i + chunk_size]
    segment_text = " ".join(chunk_sents)
```

**Arsyetimi:** Zgjedhja e 3 fjalive si madhësi segmenti balancon nevojën për kontekst të mjaftueshëm (për të detektuar ndryshime stili) dhe shmangien e segmenteve tepër të gjata që mund të përmbajnë shumë ndryshime të ndryshme.

#### 1.2 Etiketimi Automatik (Auto-Labeling)

Për të krijuar një dataset fillestar, është përdorur një sistem etiketimi automatik me rregulla heuristike:

| Lloji | Rregulla Heuristike |
|-------|---------------------|
| **NARRATOR_SHIFT** | Prania e përemrave vetë e parë DHE vetë e tretë në të njëjtin segment |
| **TENSE_SHIFT** | Prania e foljeve në kohë të ndryshme (e shkuar + e tashme ose e shkuar + e ardhme) |
| **REGISTER_SHIFT** | Prania e fjalëve formale DHE joformale në të njëjtin segment |
| **EMOTION_SHIFT** | Prania e fjalëve me emocione pozitive DHE negative, ose fjala negative + "but" |

**Arsyetimi:** Etiketimi automatik me heuristikë lejon krijimin e shpejtë të një dataset-i trajnimi. Megjithëse jo perfekt, këto etiketa sigurojnë një pikënisje solide që mund të përmirësohet me etiketim manual.

#### 1.3 Përpunimi Gjuhësor (NLP Preprocessing)

Për çdo segment, nxirren informacione gjuhësore duke përdorur spaCy:

- **Tokenization**: Ndarja në fjalë/shenja individuale
- **Lemmatization**: Reduktimi i fjalëve në formën bazë
- **POS Tagging**: Identifikimi i pjesëve të ligjëratës

```python
# Nga preprocess.py
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
for doc in nlp.pipe(texts, batch_size=32):
    tokens = [t.text for t in doc]
    lemmas = [t.lemma_ for t in doc]
    pos = [t.pos_ for t in doc]
```

---

### 2. Nxjerrja e Karakteristikave (Feature Engineering)

Për secilin segment, llogariten 8 karakteristika numerike që kapin aspekte të ndryshme të stilit:

```python
# Nga features.py
return numpy.array([
    fp_ratio,        # Raporti vetë e parë
    tp_ratio,        # Raporti vetë e tretë
    verb_ratio,      # Raporti i foljeve
    neg_ratio,       # Raporti i fjalëve negative
    formal_ratio,    # Raporti i fjalëve formale
    informal_ratio,  # Raporti i fjalëve joformale
    avg_token_len,   # Gjatësia mesatare e fjalëve
    length,          # Numri i fjalëve
], dtype=float)
```

**Arsyetimi i Zgjedhjes së Features:**

| Feature | Arsyetimi |
|---------|-----------|
| `fp_ratio` / `tp_ratio` | Detektojnë ndryshimin e perspektivës së tregimtarit |
| `verb_ratio` | Tregon densitetin e veprimeve në tekst |
| `neg_ratio` | Kap tonin emocional të tekstit |
| `formal_ratio` / `informal_ratio` | Identifikojnë ndryshimin e regjistrit |
| `avg_token_len` | Fjalët më të gjata shpesh tregojnë stil më formal |
| `length` | Kontrollon për efektin e gjatësisë së segmentit |

---

### 3. Trajnimi i Modeleve

#### 3.1 Model Tradicional: Logistic Regression

**Arsyetimi i Zgjedhjes:** Logistic Regression është zgjedhur si baseline për disa arsye:
- Interpretueshmëri e lartë (mund të shohim peshat e secilit feature)
- Trajnim i shpejtë
- Performancë e mirë për probleme binare
- Rezistencë ndaj overfitting me dataset të vogël

```python
# Nga train_traditional.py
clf = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
```

**Parametrat:**
- `max_iter=1000`: Numri maksimal i iteracioneve për konvergjencë
- `test_size=0.3`: 70% trajnim, 30% testim
- `stratify=y`: Ruajtja e proporcionit të klasave në ndarje

#### 3.2 Model Transformer: DistilBERT

**Arsyetimi i Zgjedhjes:** DistilBERT ofron:
- Kuptim të thellë kontekstual të tekstit
- Performancë të ngjashme me BERT por 40% më të shpejtë
- Aftësi për të kapur nuanca stilistike që feature engineering nuk i identifikon

```python
# Nga train_transformer.py
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
)
```

**Parametrat dhe Arsyetimi:**
| Parametër | Vlera | Arsyetimi |
|-----------|-------|-----------|
| `num_train_epochs` | 3 | Balancon trajnimin e mjaftueshëm pa overfitting |
| `batch_size` | 8 | Madhësi e arsyeshme për memorie GPU |
| `learning_rate` | 5e-5 | Vlera standarde për fine-tuning të transformerëve |
| `weight_decay` | 0.01 | Regularizim për të parandaluar overfitting |
| `max_length` | 256 | Mjaftueshëm për 3 fjali, eficient në memorie |

---
