# Multilingual Retrieval: Direct vs. Translate-then-Retrieve

A comparison of two approaches to cross-lingual information retrieval — embedding documents directly in their native language versus machine-translating them to English first — across six sentence-embedding models, for German and Hindi queries against an English-pivot corpus.

The core question: **for a given embedding model, does translating everything to English before retrieval help or hurt?** The answer turns out to depend heavily on whether the model was trained to be multilingual in the first place.

Evaluation uses the full corpus and query set (13,500 documents and 1,500 queries per language) with graded relevance judgments (nDCG computed from actual relevance scores, not just binary presence/absence).

## Setup

```bash
pip install -r requirements.txt
```

**If you have an NVIDIA GPU**, install a CUDA-enabled build of PyTorch explicitly — `pip install torch` alone often resolves to a CPU-only wheel even when a GPU is present, which silently makes translation and embedding dramatically slower with no error message. Check your driver's supported CUDA version with `nvidia-smi`, then install a matching build, e.g.:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify everything is set up correctly:

```bash
python test_setup.py
```

This checks required packages, config validity, directory structure, GPU availability, and the translation cache.

## Running the pipeline

The pipeline runs in this order. Each step is resumable — if interrupted, rerunning the same command picks up where it left off rather than starting over, via checkpointing and skip-if-complete logic.

```bash
# 1. Translate the non-English corpus to English (NLLB)
python translate_corpus.py --language de --translation-model nllb
python translate_corpus.py --language hi --translation-model nllb

# 2. Translate the non-English queries to English (NLLB)
python translate_queries.py --language de --translation-model nllb
python translate_queries.py --language hi --translation-model nllb

# 3. Generate embeddings for native-language corpora (en, de, hi)
python generate_embeddings.py

# 4. Generate embeddings for the translated corpora (de, hi)
python generate_embeddings_translated.py

# 5. Run the evaluation (all 6 models, both pipelines, full query set)
python evaluate.py

# 6. Build result visualizations
python compare_results.py --no-plots
```

Each script also accepts `--model`/`--models` to target a single model, and `translate_corpus.py`/`translate_queries.py` accept `--language`. `evaluate.py` accepts `--test` to run a quick, reduced-query sanity check against the full indices without touching real results.

## Results

Average nDCG@10 across all queries (1,500/language), per model:

| Model | Direct | Translate |
|---|---|---|
| bge-m3 | **0.7304** | 0.6228 |
| multilingual-e5-base | **0.7212** | 0.6056 |
| paraphrase-multilingual-MiniLM-L12-v2 | **0.4978** | 0.4538 |
| stella-base-en-v2 | 0.4384 | **0.5524** |
| all-mpnet-base-v2 | 0.4393 | **0.5479** |
| all-MiniLM-L6-v2 | 0.4098 | **0.5401** |

**The finding**: it splits cleanly along one line — whether the model was trained on multiple languages.

- **bge-m3** and **multilingual-e5-base** (large multilingual models) do *better* embedding documents in their native language directly. Both lose several points of nDCG@10 once the corpus is translated first — translation strips out signal these models already know how to use natively (bge-m3: 0.732→0.664 on German, 0.675→0.582 on Hindi; multilingual-e5-base: 0.734→0.643 on German, 0.651→0.568 on Hindi).
- **all-MiniLM-L6-v2**, **all-mpnet-base-v2**, and **stella-base-en-v2** (English-only models) are barely functional on native Hindi (nDCG@10 as low as 0.033 for all-MiniLM-L6-v2) but recover almost entirely once the corpus is translated to English first (up to 0.533 on the same language) — unsurprising, since they were never trained on Hindi text at all.
- **paraphrase-multilingual-MiniLM-L12-v2**, a smaller multilingual model, sits in between and even flips by language — direct wins on German, translation wins on Hindi — plausibly because its multilingual training data covers German more thoroughly than Hindi.

Full per-language, per-model breakdowns are in `results/winloss.csv`; per-language raw metrics (precision, success, MRR, nDCG at k=1/5/10) are in `results/per_language/*.csv`; aggregated summaries are in `results/summary.csv`. Interactive charts: `results/model_comparison.html` and `results/model_language_winloss.html`.

## Repo structure

```
.
├── config.yaml                          # models, languages, dataset names, k-values
├── requirements.txt
├── test_setup.py                        # environment/setup sanity check
├── verify_embeddings.py                 # confirms all embedding indices are present and correctly sized
├── utils.py                             # translation functions (single + batched)
├── translation_cache.py                 # SHA256-keyed translation cache with disk persistence
├── translate_corpus.py                  # corpus → English (batched, checkpointed, logged)
├── translate_queries.py                 # queries → English (same pattern)
├── generate_embeddings.py               # native-language embeddings → FAISS indices
├── generate_embeddings_translated.py    # translated-corpus embeddings → FAISS indices
├── evaluate.py                          # runs both pipelines, computes graded metrics
├── compare_results.py                   # builds result visualizations
├── embeddings/                          # FAISS indices (.index files)
├── translated_corpus/                   # translated documents (JSON)
├── translated_queries/                  # translated queries (JSON)
├── cache/                               # translation cache + backups
├── results/                             # per-model/language CSVs, summary, winloss, charts
├── logs/                                # timestamped logs for every pipeline run
└── audit/                               # scripts used to verify dataset integrity and benchmark hardware
```

## Future work

Hard-negative mining and fine-tuning were part of an earlier version of this project but are out of scope here — they answer a different question (can retrieval be improved through training) than the one this project focuses on (how do existing models compare, as-is, across retrieval strategies). Similarly, an interactive search demo was cut in favor of keeping the project centered on the evaluation itself; a demo built directly on the FAISS indices and result data produced here would be a natural extension.
