# Troubleshooting

Real issues hit while building this project, with root cause and the actual fix applied — not generic advice.

## 1. `pip install torch` silently installed a CPU-only build

**Symptom**: `torch.cuda.is_available()` returned `False` on a machine with a working NVIDIA GPU. No error, no warning — translation and embedding just ran (very) slowly on CPU.

**Root cause**: `pip install torch` can resolve to a CPU-only wheel depending on the environment, even when `requirements.txt` doesn't request one explicitly. `pip show torch` confirmed the installed version had the `+cpu` suffix.

**Fix**:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
(`cu126` chosen to stay at or below the installed driver's max supported CUDA version, checked via `nvidia-smi`.) Verified afterward with `torch.cuda.get_device_name(0)`.

## 2. Hugging Face cache hang during M2M100 loading

**Symptom**: a batched GPU translation script reproducibly hung right after "Loading m2m100 model..." — no error, no progress, indefinitely.

**Root cause**: two problems stacked. First, Windows Developer Mode (needed for filesystem symlinks) was off, so Hugging Face's cache fell back to full file duplication instead of symlinks — every model pull used far more disk than necessary. Second, free disk space had dropped to ~4.8GB, and near-full-disk multi-gigabyte file operations on Windows can stall rather than fail cleanly. A stale, weights-incomplete snapshot revision (from an earlier interrupted pull) was also sitting in the local cache, which caused a separate, faster failure once offline mode was forced for diagnosis (`OSError: ... does not appear to have a file named model.safetensors`).

**Fix**: freed disk space (to ~43GB), enabled Developer Mode, deleted the stale cache folder entirely (`~/.cache/huggingface/hub/models--facebook--m2m100_418M`), and re-downloaded cleanly in online mode.

## 3. The "translate" pipeline was searching the wrong index

**Symptom**: results for the translate-then-retrieve pipeline looked plausible but weren't actually testing what they claimed to.

**Root cause**: the evaluation code translated the *query* to English but then searched the *native-language* FAISS index — the translated-corpus embedding infrastructure existed in the codebase but was never wired in. This meant "translate" results were really "translated query vs. native-language documents," a different and non-standard setup.

**Fix**: rewrote the translate pipeline to load `embeddings/{model}_{lang}_translated_en_nllb.index` and pull document IDs from `translated_corpus/corpus_{lang}_translated_en_nllb.json`, so both query and corpus are genuinely in the same (translated) language space.

## 4. nDCG was binary, but relevance judgments are graded

**Symptom**: nDCG@10 gave identical credit to a "partially relevant" document and a "fully relevant" one.

**Root cause**: the qrels dataset assigns graded scores (0.5 = partial, 1.0 = full relevance), but the nDCG computation only checked whether a document was *present* in qrels at all, discarding the actual score — effectively binary DCG, not standard nDCG.

**Fix**: implemented the standard graded-relevance formula, `DCG = Σ (2^score − 1) / log2(rank + 1)`, with IDCG computed from the ideal (score-sorted) ranking. Precision/Success/MRR remain binary (`score > 0`), which is standard for those metrics.

## 5. `multilingual-e5-base` needs explicit query/passage prefixes

**Symptom**: no visible error — this was an assumption that all six models could be encoded identically, which turned out to be wrong for one of them.

**Root cause**: E5-family models require input text prefixed with `"query: "` (for queries) or `"passage: "` (for documents) to perform correctly; without it, retrieval quality degrades silently — no error is raised either way. Verified against the model's own documentation before treating this as a real issue.

**Fix**: added conditional prefixing (only for `model_name == 'multilingual-e5-base'`) in the embedding-generation scripts (documents) and the evaluation script (queries), and regenerated all five of that model's embedding indices, since the existing ones had been built without the prefix.

## 6. Resumability logic trusted stale results

**Symptom**: a full evaluation run silently skipped the "direct" pipeline for one model, because a pre-existing results file for that model happened to already have rows for all expected languages — except those rows were from an old, much smaller test run (56 queries instead of 1,500).

**Root cause**: the initial resumability check only verified language coverage, not whether the data was actually current or complete.

**Fix**: added checks that a results file has the expected `ndcg@10` column, does not have a legacy schema marker column, and — critically — that `num_queries` matches the true full query-set size for that language before treating it as "done." Runs started with `--test` now never satisfy resumability, so a quick verification run can never be mistaken for real progress.

## 7. Selectively purging one translation model's cache entries

**Context**: after deciding to drop one translation backend from the project, its ~27,000 cached translations needed removing from a cache shared with the kept backend, without touching those entries.

**Approach**: the cache's keys are `SHA256(f"{model}::{lang}::{text}")` — deterministic and reconstructable. A one-off script recomputed that exact hash for every corpus document under the model being removed and deleted matching keys, verified by checking the before/after entry count against the known corpus size (accounting for a handful of legitimately duplicate source texts).

## 8. Recovering from an interrupted overnight run

**Symptom**: a long (~13 hour) translation job spanning four language/model combinations stopped partway through the last one overnight; exact cause unconfirmed.

**Recovery**: because the translation script checkpoints progress every 500 documents and skips any combination whose output file already matches the full expected document count, simply rerunning the exact same command skipped the three already-complete jobs instantly and resumed the interrupted one from its last checkpoint rather than from zero.

**Takeaway**: any job expected to run unattended for hours needs checkpointing and skip-if-complete logic from the start — not added defensively after the fact.

## 9. Visualization script broke after a results schema change

**Symptom**: after reworking how win/loss comparisons were computed (per model-and-language instead of averaged across all models into one number per language), the plotting script referenced columns that no longer existed.

**Fix**: replaced the old aggregated pie/bar charts with a single faceted bar chart (one facet per language, bars grouped by model and colored by pipeline) — which is also a strictly more informative view, since averaging across models had been hiding the fact that different model types disagree about which pipeline wins.

## 10. The summary table averaged different language sets for the two pipelines

**Symptom**: `summary.csv` and the first README table showed English-only models gaining about +0.11 from translation, when the per-language results showed gains of +0.27 to +0.30.

**Root cause**: the direct pipeline is evaluated on English, German and Hindi, but the translate pipeline only on German and Hindi (English needs no translation). Averaging each pipeline over its own languages therefore compared a three-language mean (including the easy English queries) with a two-language mean. For stella-base-en-v2 the reported direct average of 0.4384 was exactly (0.7670 + 0.4740 + 0.0741) / 3. The per-language numbers were correct; only the aggregate was misleading.

**Fix**: `summary.csv` now averages both pipelines over the non-English languages only and reports English direct performance in a separate `en_ndcg@10` column. `evaluate.py --summary-only` rebuilds the summary and win/loss tables from existing per-language CSVs without recomputing anything.

## 11. Resuming a partially finished evaluation

**Symptom**: a full `evaluate.py --force` run was killed by a tool timeout after finishing three of six models. A plain rerun would have skipped the unfinished models entirely, and `--force` alone would have redone the finished ones.

**Root cause**: resumability is decided per (model, pipeline) from the existing per-language CSV. The unfinished models still had complete CSVs from an earlier run, so they looked "done" even though the new per-query files had not been written for them.

**Fix**: resume with `--force` restricted to the remaining models, e.g. `python evaluate.py --force --models <unfinished models>`. Long runs are now started manually in a terminal rather than through a tool with a fixed time limit.

## 12. Intermittent GPU slowdowns

**Symptom**: in one evaluation run, encoding 1,500 queries took anywhere from about one second to over seven minutes depending on the step, a roughly 100x swing. A rerun of the same code on the same data and machine ran at normal speed.

**Status**: the cause was not isolated. Likely suspects are power or thermal throttling on a laptop GPU, contention from other GPU-using applications, or VRAM paging. The same kind of variance showed up as uneven checkpoint intervals in the long translation runs.

**Mitigation**: before long runs, plug in the charger, set the power mode to Best performance, close GPU-heavy applications, and watch `nvidia-smi -l 5` in a second terminal. Every long stage checkpoints its progress, so a slow or killed run loses little.

## 13. NLLB ran without the source-language tag, and long documents were only partly translated

**Symptom**: German questions starting with "Was" ("what") came back with "Was" untranslated, e.g. "Was an untrue empire?". A full audit found this in 280 of 502 such questions (56%), and 293 of 1,500 German queries (19.5%) kept their first word unchanged; the figure for Hindi was 0 of 1,500. The same audit of the translated corpora found two more problems:
- **Dropped content in long documents.** The median translated/source word ratio falls with source length, from 0.84 to 0.41 for Hindi and from 1.02 to 0.58 for German. 1,057 Hindi documents (7.8%) and 151 German documents (1.1%) kept less than half their words; among Hindi documents over 300 words it is 68%. Some translations start mid-document.
- **Decoding loops.** 111 German and 83 Hindi corpus documents (0.8% and 0.6%) contain a word 4-gram repeated 20 or more times (up to 451), running to the 512-token output cap.

**Root cause (language tag)**: with transformers 5.3.0, `AutoTokenizer.from_pretrained` returns a generic tokenizer for NLLB, and neither `tokenizer.src_lang = ...` nor `src_lang=` passed to `AutoTokenizer` has any effect. Every input ended in `<unk>` where the source-language token belongs (`[..., </s>, <unk>]`), identically for German, Hindi and English. `NllbTokenizer.from_pretrained(model_id, src_lang=code)` produces the documented `[lang] X </s>` layout, and with it all test sentences translated correctly ("Was ist eine unechte Empore?" becomes "What is a false empore?"). Without a tag the model must guess the source language: Latin-script German, which shares many words with English, suffered most, while Hindi's script gave it an obvious cue.

**Root cause (dropped content)**: NLLB is a sentence-level model, and fed whole multi-sentence passages it tends to skip content. Truncation at 512 tokens is not the main cause: at most 0.76% of documents in any language exceed 512 NLLB tokens.

**Fix**: translation was rewritten and tagged `nllb-v2`. The language token is built explicitly (`NllbTokenizer`, manual `[lang] X </s>` layout, an exception if it is ever `<unk>`), documents are translated sentence by sentence in length-sorted batches, output length is capped, and degenerate repetition triggers a retranslation with beam search and a repeated-trigram ban. New outputs, caches and indices carry the tag, and the old files were left untouched, so nothing could silently reuse the defective translations (the old cache would otherwise have returned them with zero misses).

**Result**: re-translating all 13,500 passages took 15 minutes (German) and 24 minutes (Hindi), against about two hours each before. Audit of the full corpora, old to new:
- median word ratio by source length (short to over 300 words): German 1.02 to 0.58 became 1.10 to 1.04; Hindi 0.84 to 0.41 became 0.91 to 0.92;
- passages keeping under half their words: German 151 to 3, Hindi 1,057 to 41;
- severe repetition: German 111 to 2, Hindi 83 to 6 (the loop guard fired 8 times for German and 107 times for Hindi);
- German queries starting "Was": 502 of 502 now translate to "What" (280 did not before); no query is identical to its source.

Effect on results (`results/v1_vs_v2.csv`): translate-pipeline nDCG@10 rose by 0.015 to 0.113 per (model, language), every 95% interval above zero. Hindi gained about 0.04 for every model. German gained about 0.10 for the English-only models but only 0.015 to 0.025 for bge-m3 and multilingual-e5-base, which fits (but was not tested) untranslated German words having matched between queries and passages for the multilingual embedders. The English-only versus multilingual split held in 11 of 12 comparisons; the one change was paraphrase-multilingual-MiniLM on German, which now prefers translation.

**Remaining defects (measured)**: 96 German passages (0.7%) and 12 Hindi passages (0.09%) have a translated/source word ratio above 2.5. The highest cases are hallucinated boilerplate ("The Commission has decided…") for bibliography lines and bare URLs, and one German query ("Warum hat Ryanair einen Anteil an Laudamotion übernommen?") was affected the same way. Two German and six Hindi passages still contain degenerate repetition.

**Lesson**: a translation step can fail silently while producing fluent output. The checks that found this were cheap: look at the token ids, count how often outputs keep the source's first word, and compare translated and source lengths by passage length.

## 14. The sentence splitter cut German dates and abbreviations in half

**Symptom**: after the language-tag fix, a consistency check showed fragments such as "The 24th . In February 2014…" and "…since the early 16th century. Century (first recorded in 1518…".

**Root cause**: the first splitter broke after every period followed by whitespace, but German ordinals ("am 24. Februar"), initials ("J. R. R.") and abbreviations ("z. B.", "Hrsg.") contain sentence-internal periods. An audit of the two corpora found at least one false boundary in 3,198 German passages (23.7%, 3,819 boundaries) and 274 Hindi passages (2.0%, 329 boundaries); among queries, 27 German (1.8%) and none in Hindi.

**Fix**: a split is undone when the previous piece ends in a 1–3 digit number with a period, a single letter with a period, or a known abbreviation. Checked on ten strings in five languages (German, Czech, Danish, Bulgarian, Finnish) plus Hindi. This is a heuristic: a wrong merge only makes a unit slightly longer. The abbreviation list covers German, Czech, Danish, Dutch, Finnish and Bulgarian forms and has not been tuned for every language.

**Lesson**: the pilot examples contained no ordinals, so the flaw surfaced only in an unrelated consistency check. Test splitters on date-heavy text.
