## Detektimi Automatik i Ndryshimeve të Stilit në Tekstet Narrative



**Lënda**: Procesimi i gjuhës natyrale

**Profesori i lëndës**: Mërgim Hoti

**Studimet**: Master - Semestri III

**Universititeti** : Universiteti i Prishtinës - " Hasan Prishtina "

<img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="110" align="right"/>

**Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK

**Drejtimi** : Inxhinieri Kompjuterike dhe Softuerike - IKS

## Anëtarët e grupit
**Blerta Krasniqi**
**Lirak Xhelili**
**Zana Guda**
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

### Atributet e Nxjerra (Features)

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

```python
# Nga preprocess.py
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
for doc in nlp.pipe(texts, batch_size=32):
    tokens = [t.text for t in doc]
    lemmas = [t.lemma_ for t in doc]
    pos = [t.pos_ for t in doc]
```

---

### 2. Trajnimi i Modeleve

#### 2.1 Model Tradicional: Logistic Regression

```python
clf = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
```

#### 2.2 Model Transformer: DistilBERT

```python
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

| Parametër | Vlera | Arsyetimi |
|-----------|-------|-----------|
| `num_train_epochs` | 3 | Balancon trajnimin e mjaftueshëm pa overfitting |
| `batch_size` | 8 | Madhësi e arsyeshme për memorie GPU |
| `learning_rate` | 5e-5 | Vlera standarde për fine-tuning të transformerëve |
| `weight_decay` | 0.01 | Regularizim për të parandaluar overfitting |
| `max_length` | 256 | Mjaftueshëm për 3 fjali, eficient në memorie |

---

##  Rezultatet

### Performanca e Modelit DistilBERT

#### Raporti i Klasifikimit

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **NO_TRANSFER (0)** | 0.960 | 0.954 | 0.957 | 2,255 |
| **HAS_TRANSFER (1)** | 0.972 | 0.976 | 0.974 | 3,653 |

#### Metrikat e Përgjithshme

| Metrikë | Vlera |
|---------|-------|
| **Accuracy** | **96.7%** |
| **Macro Avg Precision** | 0.966 |
| **Macro Avg Recall** | 0.965 |
| **Macro Avg F1-Score** | 0.965 |
| **Weighted Avg F1-Score** | 0.967 |

### Statistikat e Trajnimit

| Metrikë | Vlera |
|---------|-------|
| **Koha e trajnimit** | 20 minuta 39 sekonda |
| **Samples/sekondë** | 33.36 |
| **Hapa totale** | 5,172 |
| **Loss fillestar** | 0.593 |
| **Loss përfundimtar** | 0.047 |
| **Train Loss mesatar** | 0.163 |

### Grafiku i Loss gjatë Trajnimit

```
Loss
0.60 ┤╮
0.50 ┤╰╮
0.40 ┤ ╰╮
0.30 ┤  ╰╮
0.20 ┤   ╰──╮
0.10 ┤      ╰────────────────────
0.05 ┤                          ╰──
     └────────────────────────────────
     0.0   0.5   1.0   1.5   2.0   2.5   3.0  Epoch
```

### Analiza e Rezultateve

####  Pikat e Forta:
- **Accuracy 96.7%** - Performancë e shkëlqyer në detektimin e ndryshimeve stilistike
- **F1-Score 0.974** për klasën HAS_TRANSFER - Model i besueshëm për identifikimin e ndryshimeve
- **Recall i lartë (97.6%)** - Modeli identifikon shumicën e rasteve me ndryshim stili
- **Loss i ulët (0.047)** - Konvergjencë e suksesshme pa overfitting

####  Kufizimet:
- Etiketimi automatik mund të ketë gabime
- Dataset-i përmban vetëm tekste nga epoka Viktoriane
- Disa lloje ndryshimesh janë më të vështira për t'u detektuar

####  Punë e Ardhshme:
- Zgjerimi i dataset-it me tekste moderne
- Etiketimi manual i një nën-bashkësie për validim
- Eksperimentimi me modele të tjera (RoBERTa, BERT-large)
- Klasifikimi multi-label për llojin specifik të ndryshimit

---

##  Instalimi dhe Përdorimi

### Kërkesat

```bash
# Krijoni dhe aktivizoni mjedisin virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ose
venv\Scripts\activate     # Windows

# Instaloni varësitë
pip install -r requirements.txt

# Shkarkoni modelin spaCy
python -m spacy download en_core_web_sm
```

### Ekzekutimi i Pipeline-it

```bash
# 1. Segmentimi i teksteve
python -m src.make_segments

# 2. Etiketimi automatik
python -m src.auto_label

# 3. Përpunimi me spaCy
python -m src.preprocess

# 4. Trajnimi i modelit tradicional
python -m src.train_traditional

# 5. Trajnimi i modelit transformer
python -m src.train_transformer

# 6. Inspektimi i modelit tradicional
python -m src.inspect_traditional
```

---

##  Varësitë (Dependencies)

| Librari | Përdorimi |
|---------|-----------|
| pandas | Manipulimi i të dhënave |
| numpy | Operacione numerike |
| scikit-learn | Modeli tradicional ML |
| spacy | Përpunimi gjuhësor |
| transformers | Modeli DistilBERT |
| torch | Backend për transformers |
| joblib | Ruajtja e modeleve |
| tqdm | Progress bars |

---

##  Referencat

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
2. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter."
3. Project Gutenberg. https://www.gutenberg.org/
4. spaCy Documentation. https://spacy.io/

---

##  Licenca

Ky projekt është zhvilluar për qëllime akademike në kuadër të lëndës NLP.

---
