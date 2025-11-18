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
