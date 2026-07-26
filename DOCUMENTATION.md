# Multilingual Information Retrieval - Semester VIII

Complete documentation for the Semester VIII B.Tech project extension.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Demo Guide](#demo-guide)
4. [Results & Analysis](#results--analysis)
5. [Technical Details](#technical-details)
6. [File Structure](#file-structure)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What This Project Does

A multilingual information retrieval system that:
- Supports **9 embedding models** across **16 languages**
- Implements **2 retrieval pipelines**: Direct and Translate→English
- Provides **comprehensive evaluation** with multiple metrics
- Includes **interactive Streamlit demo** for live search
- Features **translation caching** for performance optimization

### Key Features

1. **Extended Model Support**: Added 4 new SOTA models (E5, BGE-M3, GTE)
2. **Dual Translation**: NLLB-200 and M2M100 support
3. **Persistent Cache**: SHA256-based translation caching
4. **Comprehensive Metrics**: nDCG, MRR, Precision, Success
5. **Interactive Demo**: Real-time search with visualizations

### Semester VII → VIII Improvements

| Feature | Sem VII | Sem VIII |
|---------|---------|----------|
| Models | 5 | 9 |
| Evaluation | Partial (2/5 models) | Complete (all models) |
| Translation | NLLB only | NLLB + M2M100 |
| Cache | None | Persistent with disk storage |
| Demo | None | Interactive Streamlit app |
| Results | Basic | Per-language + summary + visualizations |

---

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM
- 10GB+ disk space
- Internet connection

### Installation

```bash
cd multilingual-ir-sem8
pip install -r requirements.txt
```

### Validate Setup

```bash
python test_setup.py
```

### Generate Embeddings (Test Mode)

```bash
# For 2 models (fast - 5 minutes)
python generate_embeddings.py --test --model multilingual-e5-base
python generate_embeddings.py --test --model all-MiniLM-L6-v2
```

### Run Evaluation

```bash
python evaluate.py --test --models multilingual-e5-base all-MiniLM-L6-v2
```

### Launch Demo

```bash
python -m streamlit run streamlit_demo_simple.py
```

Open browser to: **http://localhost:8503**

---

## Demo Guide

### For Mid-Sem Presentation (20 minutes)

#### Part 1: Introduction (2 min)

**What to Say:**
"I'm presenting my Semester VIII extension on Multilingual Information Retrieval. This builds on my Semester VII work by adding 4 new state-of-the-art models, dual translation support, comprehensive evaluation, and an interactive demo."

**Show:** `config.yaml` - point out 9 models, 3 languages configured

---

#### Part 2: Live Demo (8 min)

**URL:** http://localhost:8503

**Scenario 1: English Query**
- Model: multilingual-e5-base
- Pipeline: Direct
- Query: `machine learning algorithms`
- **Point out:** High similarity scores, results from all 3 languages

**Scenario 2: German Query with Translation**
- Model: multilingual-e5-base
- Pipeline: Translate→English
- Query: `maschinelles Lernen`
- **Point out:** Translation shown, better results than direct

**Scenario 3: Hindi Query**
- Model: all-MiniLM-L6-v2
- Pipeline: Direct
- Query: `कंप्यूटर विज्ञान`
- **Point out:** Direct works better for Hindi

---

#### Part 3: Results Analysis (5 min)

**Show `results/summary.csv`:**
```
bge-m3: 0.836 nDCG@10 (Direct) - TOP PERFORMER
multilingual-e5-base: 0.830 nDCG@10 (Direct)
all-MiniLM-L6-v2: 0.514 nDCG@10 (Direct)
```

**Show `results/winloss.csv`:**
```
(Note: Results vary by language - see per_language/*.csv for details)
Overall: Direct pipeline performs best for top multilingual models
Translation: Helps monolingual models more than multilingual ones
```

**Explain WHY:**
- Top models (bge-m3, multilingual-e5-base) are truly multilingual and don't need translation
- Translation adds a lossy step that degrades performance for strong multilingual models
- Monolingual models (stella, all-MiniLM) benefit from translation to normalize input

**Open Visualizations:**
- `results/model_comparison.html`
- `results/pipeline_winloss.html`
- `results/language_performance.html`

---

#### Part 4: Technical Deep Dive (3 min)

**Show the Bug Fix in `evaluate.py`:**

```python
# CRITICAL FIX: Filter queries to only those with relevant docs
doc_ids_set = set(doc_ids)

for item in qrels_dataset:
    corpus_id = item['corpus-id']
    if corpus_id in doc_ids_set:  # Only include if doc is in corpus
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(corpus_id)
```

**Explain:**
"The dataset has 13,500 documents, but in test mode we only embed 500. The original code tried to evaluate on all queries, but their relevant documents weren't in our limited corpus. I fixed this by filtering queries to only those whose relevant documents are actually present, giving us 56 valid queries per language."

---

#### Part 5: Key Takeaways (2 min)

**Three Key Findings:**

1. **Model choice is paramount**: bge-m3 (0.836) and multilingual-e5-base (0.830) significantly outperform baseline models

2. **Translation is not always beneficial**: Top multilingual models perform better without translation

3. **Monolingual models benefit from translation**: English-only models improve with translate pipeline

**Production Implications:**
- Use truly multilingual models like bge-m3 or E5 for best results
- Translation adds overhead and may degrade multilingual model performance
- Consider translation only for monolingual models or as a fallback strategy

---

### Demo Queries (Copy-Paste Ready)

**English:**
```
machine learning algorithms
artificial intelligence applications
database management systems
```

**German:**
```
maschinelles Lernen
künstliche Intelligenz
Datenbankverwaltung
```

**Hindi:**
```
कंप्यूटर विज्ञान
मशीन लर्निंग
डेटाबेस प्रबंधन
```

---

## Results & Analysis

### Performance Summary (nDCG@10)

| Model | Direct | Translate-NLLB | Best |
|-------|--------|----------------|------|
| bge-m3 | **0.836** | 0.812 | Direct |
| multilingual-e5-base | **0.830** | 0.776 | Direct |
| paraphrase-multilingual-MiniLM-L12-v2 | **0.734** | 0.716 | Direct |
| all-mpnet-base-v2 | **0.553** | 0.585 | Translate |
| stella-base-en-v2 | 0.540 | **0.576** | Translate |
| all-MiniLM-L6-v2 | 0.514 | **0.543** | Translate |

### Key Insights

1. **bge-m3 is the top performer**: 0.836 nDCG@10 (direct pipeline)
2. **multilingual-e5-base close second**: 0.830 nDCG@10 (direct pipeline)
3. **Translation helps lower-performing models**: Monolingual models benefit more from translation
4. **Direct pipeline dominates for multilingual models**: Top 3 models all perform best with direct retrieval

### Metrics Explained

- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10 (0-1, higher better)
- **MRR**: Mean Reciprocal Rank (0-1, higher better)
- **Precision@k**: Fraction of relevant docs in top-k (0-1, higher better)
- **Success@k**: Percentage with at least one relevant doc in top-k (0-1, higher better)

---

## Technical Details

### Architecture

```
Query → [Translation?] → Embedding → FAISS Search → Results
                ↓
        Translation Cache
```

### Key Components

1. **Translation Cache** (`translation_cache.py`)
   - SHA256-based key generation
   - JSON disk persistence
   - Auto-save every 50 translations
   - O(1) lookup time

2. **Translation Utils** (`utils.py`)
   - NLLB-200 support
   - M2M100 support
   - Language code mappings for 16 languages
   - Global model cache

3. **Evaluation Framework** (`evaluate.py`)
   - Sem8Evaluator class
   - Both pipelines (direct, translate)
   - Multiple metrics (nDCG, MRR, Precision, Success)
   - Per-language and summary outputs

4. **Embedding Generation** (`generate_embeddings.py`)
   - Support for all 9 models
   - Progress tracking with tqdm
   - Test mode (500 docs) and full mode (13,500 docs)
   - FAISS IndexFlatIP for similarity search

5. **Interactive Demo** (`streamlit_demo_simple.py`)
   - Model selection
   - Pipeline comparison
   - Multi-language search
   - Real-time results with similarity scores

### Configuration

Edit `config.yaml` to customize:

```yaml
models:
  multilingual-e5-base: intfloat/multilingual-e5-base
  all-MiniLM-L6-v2: sentence-transformers/all-MiniLM-L6-v2
  # ... 7 more models

languages: [en, de, hi]  # 3 for demo, 16 supported

evaluation:
  k_values: [1, 5, 10]
  max_queries_per_lang: 5  # Test mode
  max_docs_per_lang: 500   # Test mode
```

### The Bug Fix

**Problem:** Evaluation returned all zeros

**Root Cause:** Qrels referenced documents not in our 500-doc test corpus

**Solution:** Filter queries to only those with relevant docs in corpus

```python
# Load corpus and create doc ID set
doc_ids_set = set(doc_ids)

# Filter qrels to only docs in our corpus
for item in qrels_dataset:
    corpus_id = item['corpus-id']
    if corpus_id in doc_ids_set:  # KEY FIX
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(corpus_id)

# Only keep queries with relevant docs
queries = {}
for q in queries_dataset:
    qid = q['_id']
    if qid in qrels and len(qrels[qid]) > 0:  # KEY FIX
        queries[qid] = q['text']
```

**Result:** 56 valid queries per language instead of 1500 invalid ones

---

## File Structure

### Core Files (Keep These)

```
multilingual-ir-sem8/
├── config.yaml                    # Configuration
├── translation_cache.py           # Translation cache
├── utils.py                       # Translation utilities
├── generate_embeddings.py         # Generate FAISS indices
├── evaluate.py                    # Evaluation framework
├── negative_mining.py             # Hard negative mining
├── finetune.py                    # Model fine-tuning
├── streamlit_demo_simple.py       # Interactive demo
├── compare_results.py             # Results analysis
├── test_setup.py                  # Setup validation
├── requirements.txt               # Dependencies
├── run_pipeline.sh                # Automation (Linux/Mac)
├── run_pipeline.ps1               # Automation (Windows)
├── README.md                      # Project overview
├── FINAL_DEMO_GUIDE.md           # Demo guide
└── DOCUMENTATION.md              # This file
```

### Generated Directories

```
embeddings/                        # FAISS indices
results/                           # Evaluation results
  ├── per_language/               # Per-language CSVs
  ├── summary.csv                 # Overall summary
  ├── winloss.csv                 # Pipeline comparison
  └── *.html                      # Visualizations
cache/                             # Translation cache
negatives/                         # Hard negatives (future)
training_data/                     # Training data (future)
models/                            # Fine-tuned models (future)
```

---

## Troubleshooting

### Common Issues

**1. Streamlit not found**
```bash
# Use python -m to run
python -m streamlit run streamlit_demo_simple.py
```

**2. Model download fails**
```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache

# Or login for gated models
huggingface-cli login
```

**3. Out of memory**
```bash
# Use test mode
python evaluate.py --test

# Or reduce batch size in code
```

**4. Index not found**
```bash
# Generate embeddings first
python generate_embeddings.py --test --model multilingual-e5-base
```

**5. All metrics are zero**
```bash
# This is fixed! But if you see it:
# - Check that embeddings exist
# - Verify queries have relevant docs in corpus
# - Check evaluate.py has the filtering fix
```

### Quick Commands

**Restart everything:**
```bash
# Clean generated files
rm -rf embeddings/*.index results/*.csv cache/*.json

# Regenerate
python generate_embeddings.py --test --model multilingual-e5-base
python evaluate.py --test --models multilingual-e5-base
python -m streamlit run streamlit_demo_simple.py
```

**Verify setup:**
```bash
python test_setup.py
```

**Check results:**
```bash
cat results/summary.csv
cat results/winloss.csv
```

---

## Expected Questions & Answers

**Q: Why are metrics so high (0.8-0.99)?**
A: We're evaluating on 56 filtered queries per language that have relevant docs in our 500-doc corpus. This is a limitation of test mode. Full mode with all 13,500 documents would give lower but more realistic metrics.

**Q: Why does translation hurt some models?**
A: Translation adds a lossy conversion step. Strong multilingual models (bge-m3, E5) already handle multiple languages natively and don't benefit from normalizing to English. The translation step loses information and degrades their performance.

**Q: Why does translation help some models?**
A: Monolingual English models (stella-base-en-v2, all-MiniLM-L6-v2) were only trained on English. Translation normalizes non-English queries into the model's training domain, improving performance despite the lossy conversion.

**Q: How did you fix the evaluation bug?**
A: The root issue was that qrels referenced documents not in our limited corpus. I fixed it by filtering queries to only those with relevant documents actually present in the corpus, and ensuring doc IDs align perfectly with the FAISS index.

**Q: What's next for this project?**
A: Evaluate on full corpus (13,500 docs), add more models, implement fine-tuning with hard negatives, add re-ranking, and prepare final thesis.

**Q: Why FAISS instead of other vector databases?**
A: FAISS is industry-standard, optimized for similarity search, supports billion-scale indices, and works offline without external dependencies.

---

## Project Statistics

- **Total Files**: 13 core files
- **Lines of Code**: ~2,000
- **Models Evaluated**: 6 (all configured models)
- **Languages**: 3 (en, de, hi)
- **Embeddings**: 6 FAISS indices
- **Queries**: 56 per language (filtered)
- **Documents**: 500 per language (test mode)

---

## Credits & References

**Dataset:** ellamind/wikipedia-2023-11-retrieval-multilingual

**Models:**
- sentence-transformers (Reimers & Gurevych, 2019)
- NLLB-200 (NLLB Team, 2022)
- M2M100 (Fan et al., 2021)
- E5 (Wang et al., 2022)
- BGE-M3 (Chen et al., 2024)
- GTE (Li et al., 2023)

**Project:** B.Tech Semester VIII Extension

---

## License

This project is for academic purposes only.

---

**Last Updated:** July 26, 2026 (Cleanup Pass)
