# Multilingual Retrieval: Direct vs. Translate-then-Retrieve

A comparison of two approaches to cross-lingual information retrieval — embedding documents directly in their native language versus machine-translating them to English first — across six sentence-embedding models, for German and Hindi.

The core question: **for a given embedding model, does translating everything to English before retrieval help or hurt?** The answer depends almost entirely on whether the model was trained to be multilingual.

Evaluation uses the full corpus and query set (13,500 passages and 1,500 queries per language) with graded relevance judgments: per query there is on average one fully relevant (1.0) and eight partially relevant (0.5) passages, and nDCG is computed from those grades rather than from binary relevance.

## Results

Mean nDCG@10 over the two non-English languages (German and Hindi), 1,500 queries each. Both pipelines are averaged over the same languages. The English column is a reference only: English queries and passages differ from the German and Hindi ones, so it is not directly comparable.

| Model | Direct | Translate | Translate − Direct | English (direct, reference) |
|---|---|---|---|---|
| bge-m3 | **0.7035** | 0.6512 | −0.052 | 0.7842 |
| multilingual-e5-base | **0.6926** | 0.6404 | −0.052 | 0.7783 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.4229 | **0.5299** | +0.107 | 0.6477 |
| stella-base-en-v2 | 0.2741 | **0.6262** | +0.352 | 0.7670 |
| all-mpnet-base-v2 | 0.2730 | **0.6185** | +0.345 | 0.7719 |
| all-MiniLM-L6-v2 | 0.2393 | **0.6068** | +0.367 | 0.7506 |

![Direct vs translate by model](results/model_comparison.png)

### Findings

1. **Strong multilingual embedders do better on native text.** bge-m3 and multilingual-e5-base lose about 0.05 nDCG@10 when the corpus is translated first (bge-m3: 0.732→0.679 on German, 0.675→0.623 on Hindi; multilingual-e5-base: 0.734→0.669 and 0.651→0.612). Translation discards signal these models already know how to use.
2. **English-only embedders need translation.** On native Hindi, stella-base-en-v2, all-mpnet-base-v2 and all-MiniLM-L6-v2 score 0.03–0.07, close to unusable; translation lifts them to 0.57–0.59. On German they go from 0.45–0.51 to 0.64–0.66.
3. **Translation largely equalizes the embedders, but does not overtake the best multilingual ones.** After translation, five of the six models land between 0.607 and 0.651, while their direct scores range from 0.24 to 0.70. The best translate-pipeline result (bge-m3, 0.651) is still below the best direct result (0.704), and the best English-only model on translated text (stella, 0.626) stays below both strong multilingual models used directly.
4. **paraphrase-multilingual-MiniLM-L12-v2, a weaker multilingual model, prefers translation in both languages** (+0.058 on German, +0.156 on Hindi): being multilingual is not enough; the model has to be strong enough that native retrieval beats an English pipeline.

**Statistics.** All 12 direct-vs-translate differences (6 models × 2 languages) are significant: paired randomization test on per-query nDCG@10, Holm-corrected, adjusted p ≈ 0.0012 (the smallest value 10,000 permutations can give across 12 tests), and every 95% bootstrap confidence interval excludes zero (half-widths about 0.007–0.014). See `results/significance.csv`. This quantifies noise from sampling queries only; it says nothing about other datasets or translators.

**Effect of the translator.** An earlier version of this comparison used a translation step with two defects: the source-language tag was never passed to NLLB, and long passages were only partly translated (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md), entry 13). The numbers above come from the corrected step. Recomputing raised translate-pipeline nDCG@10 for every (model, language) pair, by 0.015 to 0.113, with all 95% intervals above zero (`results/v1_vs_v2.csv`). The split between English-only and multilingual embedders held in 11 of 12 comparisons; the margin for bge-m3 and multilingual-e5-base shrank from about 0.08–0.09 to 0.05, and the weaker multilingual model switched from preferring direct retrieval on German to preferring translation.

Per-language detail is in `results/winloss.csv` and `results/per_language/*.csv` (precision, success, MRR and nDCG at k = 1/5/10).

![Direct vs translate by model and language](results/model_language_winloss.png)

## Method

- **Direct pipeline:** the query and the corpus stay in their native language; the embedding model has to handle the language itself.
- **Translate pipeline:** both the corpus and the query are translated to English with NLLB-200-distilled-600M, then embedded and searched in an all-English index.
- **Translation:** passages are split into sentences with a boundary-aware splitter (so dates such as "24. Februar" and abbreviations are not cut apart) and translated sentence by sentence in length-sorted GPU batches. The source-language tag is built explicitly, output length is capped relative to the input, and degenerate repetition triggers a retranslation with beam search. Translating all 13,500 passages takes about 15 minutes for German and 24 minutes for Hindi on an RTX 4050 laptop GPU.
- **Metrics:** graded nDCG@k plus precision, success and MRR at k = 1, 5, 10. Embedder-specific prefixes (e5's `query:` / `passage:`) are applied.
- **Reproducibility:** recomputing the evaluation from the saved indices reproduced the earlier per-language result files exactly.

## Setup

```bash
pip install -r requirements.txt
```

**If you have an NVIDIA GPU**, install a CUDA-enabled build of PyTorch explicitly — `pip install torch` alone often resolves to a CPU-only wheel even when a GPU is present, which silently makes translation and embedding dramatically slower with no error message. Check your driver's supported CUDA version with `nvidia-smi`, then install a matching build, e.g.:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Static chart export (`compare_results.py`) uses `kaleido`, which needs a Chrome installation (`plotly_get_chrome -y` installs one if missing).

Verify the setup:

```bash
python test_setup.py
```

## Running the pipeline

Each step is resumable: if interrupted, rerunning the same command continues from the last checkpoint instead of starting over.

```bash
# 1. Translate the non-English corpus to English (NLLB)
python translate_corpus.py --language de --translation-model nllb-v2
python translate_corpus.py --language hi --translation-model nllb-v2

# 2. Translate the non-English queries to English
python translate_queries.py --language de --translation-model nllb-v2
python translate_queries.py --language hi --translation-model nllb-v2

# 3. Embeddings: native corpora, then translated corpora
python generate_embeddings.py
python generate_embeddings_translated.py
python verify_embeddings.py

# 4. Evaluate all models, both pipelines, full query set
python evaluate.py

# 5. Significance tests on direct vs translate
python significance_analysis.py

# 6. Charts
python compare_results.py --no-plots
```

Useful flags: `--model`/`--models` select specific models; `evaluate.py --force` recomputes even if results exist; `evaluate.py --summary-only` rebuilds `summary.csv` and `winloss.csv` from the existing per-language CSVs without recomputing anything; `evaluate.py --test` runs a reduced-query sanity check. `python audit/audit_translation_health.py --tag nllb-v2` checks the translated text (length ratios by passage length, repetition, leftover source words).

## Repo structure

```
.
├── config.yaml                          # models, languages, dataset names, k-values
├── requirements.txt
├── test_setup.py                        # environment/setup sanity check
├── verify_embeddings.py                 # confirms all embedding indices are present and correctly sized
├── utils.py                             # sentence-level NLLB translation (explicit language tag, loop guard)
├── translation_cache.py                 # SHA256-keyed per-language translation cache, atomic saves
├── translate_corpus.py                  # corpus → English (batched, checkpointed, logged)
├── translate_queries.py                 # queries → English (same pattern)
├── generate_embeddings.py               # native-language embeddings → FAISS indices
├── generate_embeddings_translated.py    # translated-corpus embeddings → FAISS indices
├── evaluate.py                          # both pipelines, graded metrics, per-query output
├── significance_analysis.py             # paired randomization tests + bootstrap CIs
├── compare_results.py                   # result charts
├── embeddings/                          # FAISS indices (gitignored, regenerable)
├── translated_corpus/                   # translated passages (gitignored, regenerable)
├── translated_queries/                  # translated queries (gitignored, regenerable)
├── cache/                               # translation caches (gitignored)
├── results/                             # CSVs, significance table, v1-vs-v2 comparison, PNG charts
├── logs/                                # timestamped run logs (gitignored)
└── audit/                               # dataset-integrity, translation-quality and hardware checks
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for the bugs found and fixed along the way.

## Relation to prior work

This project was prompted by the TIFIN India system for SemEval-2025 Task 7 (Devadiga et al., 2025, *Harnessing Translation to Overcome Multilingual IR Challenges in Fact-Checked Claim Retrieval*), which translates posts to English with an LLM and retrieves with a fine-tuned English embedding model, and recommends translation plus a fine-tuned embedder over heavier post-processing. Their ablation measures the benefit of translation for one English-only embedder (Stella 400M). The comparison here asks the same question across six embedders and finds the benefit depends on the embedder: it holds for English-only models and reverses for strong multilingual ones. The two are not directly comparable — different task (fact-checked claim retrieval vs. Wikipedia retrieval), different metric (success@10 vs. graded nDCG@10), and they use LLM translation and fine-tuning, which are not used here.

## Limitations

- **Translator quality.** All translation uses NLLB-200-distilled-600M, a small model. A stronger (e.g. LLM-based) translator would likely raise the translate pipeline's scores, so the conclusion that direct beats translate for the strongest multilingual embedders is conditional on this translator; the fix described above already moved those margins by 0.015–0.04.
- **Residual translation defects (measured, small).** In the translated corpora, 96 German passages (0.7%) and 12 Hindi passages (0.09%) have more than 2.5 times as many words as their source; the highest cases are hallucinated boilerplate for bibliography lines and bare URLs. One of 1,500 German queries was affected the same way. Eight of 27,000 passages still contain degenerate repetition after the loop guard.
- **Two non-English languages and one dataset.** Findings about "language dependence" rest on German and Hindi only.
- **Small collection.** 13,500 passages per language; absolute scores would be lower on larger collections.
- **No fine-tuning, reranking or hybrid search.** All embedders are used as published.
- **Significance covers query sampling only**, not translator, dataset or corpus choice.

## Future work

Extending to the other languages available in the dataset, a stronger translator as an ablation, and fine-tuning with hard negatives (an earlier version of this project scaffolded it; it was cut to keep the focus on the retrieval comparison). An interactive search demo built on the saved indices is another natural extension.
