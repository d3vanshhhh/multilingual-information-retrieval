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
     (de, hi → en, NLLB nllb-v2,                (de, hi → en, NLLB nllb-v2,
      sentence-level, checkpointed)            sentence-level, checkpointed)
              │                                     │
              ▼                                     ▼
   translated_corpus/*.json              translated_queries/*.json
              │
              ▼
generate_embeddings.py          generate_embeddings_translated.py
(native corpus, all 3 langs)     (translated corpus, de/hi only)
              │                                     │
              ▼                                     ▼
      embeddings/*.index                  embeddings/*_translated_en_nllb-v2.index
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

- `corpus_{lang}_translated_en_nllb-v2.json` — translated corpus documents for a non-English language, keyed by original `_id`.
- `queries_{lang}_translated_en_nllb-v2.json` — translated queries, same convention.
- `{model}_{lang}.index` — FAISS index of native-language embeddings for a given model/language.
- `{model}_{lang}_translated_en_nllb-v2.index` — FAISS index built from the *translated* corpus for that language.
- The `_nllb-v2` suffix is the translation tag (`TRANSLATION_TAG` in `utils.py`). It identifies which version of the translation step produced the underlying text and is part of every cache, output and index name, so results from different translation methods can never be mixed. The earlier, defective version (`nllb`) is described in TROUBLESHOOTING.md, entry 13.

## Document ID alignment

FAISS indices are plain vector stores — they have no concept of document IDs, only positions. Every script that builds or queries an index must therefore load documents in *exactly* the same order the index's vectors were written in, and map search results back to IDs by position.

- **Native embeddings**: built by iterating the Hugging Face corpus dataset directly (`load_dataset(..., split='test')`); document order is confirmed deterministic across repeated loads (verified independently before relying on it). `evaluate.py`'s direct pipeline loads the same corpus the same way to reconstruct the ID list.
- **Translated embeddings**: built by iterating the `translated_corpus/*.json` file, which itself preserves the original corpus's iteration order (the translation script writes it out in that same order after translating). `evaluate.py`'s translate pipeline reads document IDs from that same JSON file, not from re-loading the HF dataset, so the order is guaranteed to match what the embedding script used.

## Translation method

`utils.translate_batch_to_english` translates with NLLB-200-distilled-600M:

- **Explicit language tag.** Input ids are built manually as `[source-language token] text </s>` using `NllbTokenizer`, and the code raises if the language token is `<unk>` or missing. (With transformers 5.x, setting `src_lang` on an `AutoTokenizer` silently does nothing.)
- **Sentence-level.** Each text is split into sentences by a boundary-aware splitter: a split after a terminator is undone when the previous piece ends in a 1–3 digit number, a single letter, or a known abbreviation (German ordinals such as "24. Februar", initials, "z. B."), and pieces under three words are merged into the previous one. The splitter is a heuristic; a false merge only makes a translation unit slightly longer.
- **Batching.** Sentences from all texts in a chunk are pooled, sorted by token length, and batched up to 64 sentences or 4,096 padded tokens; results are re-joined per text through recorded index ranges, with assertions that nothing is missing.
- **Decoding guards.** Greedy decoding with an output cap of twice the input length plus ten tokens. If an output repeats a word 4-gram eight or more times, that sentence is retranslated alone with beam search and a repeated-trigram ban. Errors are never swallowed: failures raise, and per-chunk statistics (sentences, loop fallbacks, empty outputs, seconds) are logged.
- **Caching.** One cache file per tag and language (`cache/translation_cache_{tag}_{lang}.json`), saved only at checkpoints and written atomically; a corrupt cache raises instead of silently starting empty. Batch composition can change individual outputs slightly (fp16), so a resumed run is not guaranteed to be bit-identical to an uninterrupted one.

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
- `translation_models`: name → Hugging Face model ID, for the translation engine (currently `nllb-v2`, NLLB-200-distilled-600M).
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
| `cache/` | Per-language translation caches (`translation_cache_{tag}_{lang}.json`, SHA256-keyed), written atomically at each checkpoint; gitignored because translation is fast enough to regenerate |
| `results/` | Per-(model,pipeline) metric CSVs, aggregated summary/winloss CSVs, significance table, PNG charts; `per_query/` (gitignored) holds per-query scores |
| `logs/` | Timestamped logs from every pipeline script run — start/end time, counts, errors |
| `audit/` | Scripts used to verify dataset integrity (no missing/duplicate references, deterministic load order, graded qrels) and benchmark hardware (CPU vs GPU, batching) before committing to the full-scale run |
