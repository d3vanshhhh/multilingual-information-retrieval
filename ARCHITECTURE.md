# Architecture

## Pipeline overview

```
                     ┌─────────────────────┐
                     │  corpus_dataset      │  (HF: en/de/hi, 13,500 docs each)
                     │  query_dataset       │  (HF: en/de/hi, 1,500 queries each)
                     │  qrels_dataset       │  (graded relevance: 0.5 / 1.0)
                     └──────────┬───────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │                                     │
     translate_corpus.py                   translate_queries.py
     (de, hi → en, via NLLB,                (de, hi → en, via NLLB,
      batched fp16, checkpointed)            batched fp16, checkpointed)
              │                                     │
              ▼                                     ▼
   translated_corpus/*.json              translated_queries/*.json
              │
              ▼
generate_embeddings.py          generate_embeddings_translated.py
(native corpus, all 3 langs)     (translated corpus, de/hi only)
              │                                     │
              ▼                                     ▼
      embeddings/*.index                  embeddings/*_translated_en_nllb.index
              │                                     │
              └──────────────────┬──────────────────┘
                                  ▼
                            evaluate.py
                (direct pipeline + translate pipeline,
                 graded nDCG, per-model/language CSVs)
                                  │
                                  ▼
                          results/*.csv
                                  │
                                  ▼
                        compare_results.py
                                  │
                                  ▼
                          results/*.html
```

## File naming conventions

- `corpus_{lang}_translated_en_nllb.json` — translated corpus documents for a non-English language, keyed by original `_id`.
- `queries_{lang}_translated_en_nllb.json` — translated queries, same convention.
- `{model}_{lang}.index` — FAISS index of native-language embeddings for a given model/language.
- `{model}_{lang}_translated_en_nllb.index` — FAISS index built from the *translated* corpus for that language.
- The `_nllb` suffix identifies which translation engine produced the underlying text (this project settled on NLLB as the sole translation engine — see README for why).

## Document ID alignment

FAISS indices are plain vector stores — they have no concept of document IDs, only positions. Every script that builds or queries an index must therefore load documents in *exactly* the same order the index's vectors were written in, and map search results back to IDs by position.

- **Native embeddings**: built by iterating the Hugging Face corpus dataset directly (`load_dataset(..., split='test')`); document order is confirmed deterministic across repeated loads (verified independently before relying on it). `evaluate.py`'s direct pipeline loads the same corpus the same way to reconstruct the ID list.
- **Translated embeddings**: built by iterating the `translated_corpus/*.json` file, which itself preserves the original corpus's iteration order (the translation script writes it out in that same order after translating). `evaluate.py`'s translate pipeline reads document IDs from that same JSON file, not from re-loading the HF dataset, so the order is guaranteed to match what the embedding script used.

## Evaluation pipelines

- **`direct`**: query and corpus both stay in their native language; the embedding model must handle the language pair natively (or, for English-only models evaluated on German/Hindi, largely fails to).
- **`translate`**: both the corpus and the query are translated to English before embedding and search — every document is retrieved from an all-English index using an all-English query.

Both pipelines are evaluated for every configured model, for German and Hindi (English has no translate pipeline, since it's already the target language).

## Metrics

For each query, at each `k` in `config.yaml`'s `k_values` (`[1, 5, 10]`):

- **Precision@k**, **Success@k**: binary — a document counts as relevant if its qrels score is `> 0`.
- **MRR**: reciprocal rank of the first binary-relevant document.
- **nDCG@k**: graded — uses the actual qrels score as the relevance grade, via the standard formula:

  ```
  DCG@k  = Σ (2^score_i − 1) / log2(rank_i + 1)   for i = 1..k
  IDCG@k = same formula applied to the ideal (score-sorted-descending) ranking
  nDCG@k = DCG@k / IDCG@k
  ```

  This rewards a fully-relevant (score 1.0) document more than a partially-relevant one (score 0.5) at the same rank, which flat binary DCG cannot express.

All per-query metrics are averaged per (model, pipeline, language). `results/summary.csv` then averages each (model, pipeline) over the **non-English** languages only, so that the direct and translate pipelines are always averaged over the same language set (English has no translate pipeline and would otherwise inflate the direct average). English direct performance is reported separately in the `en_ndcg@10` column.

## Statistical testing

`evaluate.py` writes per-query nDCG@10 to `results/per_query/{model}_{pipeline}_{language}.csv` (gitignored; regenerated by every full run). `significance_analysis.py` pairs the direct and translate scores per query for each (model, non-English language) and computes:

- the mean paired difference (translate − direct), checked against the margin in `winloss.csv`;
- a 95% confidence interval from a paired bootstrap (10,000 resamples, percentile method);
- a two-sided paired randomization (sign-flip) test, 10,000 permutations, following Smucker, Allan & Carterette (CIKM 2007);
- Holm–Bonferroni adjustment across all comparisons.

With 10,000 permutations the smallest attainable p-value is 1/10,001, so the table reports "p < 1e-4" for the strongest effects rather than a precise value. Results land in `results/significance.csv`. The tests cover sampling noise from the query set only.

## config.yaml structure

- `models`: name → Hugging Face model ID, for every embedding model evaluated.
- `translation_models`: name → Hugging Face model ID, for the translation engine (currently NLLB only).
- `query_dataset` / `corpus_dataset` / `qrels_dataset`: Hugging Face dataset identifiers.
- `languages`: which languages are evaluated (`en`, `de`, `hi`).
- `evaluation.k_values`: the cutoffs metrics are computed at.
- `evaluation.max_queries_per_lang` / `max_docs_per_lang`: used only when a script is run with `--test`, for fast sanity checks against the real full-scale indices — never used in a full run.

## Directory purposes

| Directory | Contents |
|---|---|
| `embeddings/` | FAISS `.index` files — native and translated, per model/language |
| `translated_corpus/` | Translated corpus documents, JSON, per language |
| `translated_queries/` | Translated queries, JSON, per language |
| `cache/` | Persistent translation cache (SHA256-keyed) plus dated backups taken before destructive changes |
| `results/` | Per-(model,pipeline) metric CSVs, aggregated summary/winloss CSVs, significance table, PNG charts; `per_query/` (gitignored) holds per-query scores |
| `logs/` | Timestamped logs from every pipeline script run — start/end time, counts, errors |
| `audit/` | Scripts used to verify dataset integrity (no missing/duplicate references, deterministic load order, graded qrels) and benchmark hardware (CPU vs GPU, batching) before committing to the full-scale run |
