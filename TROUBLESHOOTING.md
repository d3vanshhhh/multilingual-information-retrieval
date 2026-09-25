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
