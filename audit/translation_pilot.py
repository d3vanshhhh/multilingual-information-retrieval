"""
Translation pilot script comparing tokenizer fixes and document segmentation strategies.

Tests different translation approaches:
- V1: Existing translations from translated_queries/corpus files
- Fixed: Using NllbTokenizer with proper language ID prefix
- W: Whole document as single unit
- S: Sentence-by-sentence translation
- P: Packed sentences (max 120 NLLB tokens)
"""
import argparse
import json
import yaml
import re
import random
import time
import torch
import numpy as np
import faiss
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import transformers.utils.logging
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, '.')
from utils import NLLB_LANG_CODES
from evaluate import load_qrels, compute_metrics, apply_query_prefix

# Silence repeated max_new_tokens warnings
transformers.utils.logging.set_verbosity_error()


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def split_sentences(text):
    """Split text into sentences and merge short ones"""
    # Split on common sentence terminators
    pieces = re.split(r'(?<=[.!?।॥؟۔…])\s+', text.strip())
    pieces = [p for p in pieces if p]  # Drop empty
    
    if not pieces:
        return []
    
    # Merge pieces with fewer than 3 words into previous
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if len(piece.split()) < 3:
            merged[-1] = merged[-1] + ' ' + piece
        else:
            merged.append(piece)
    
    return merged


def count_word_ngrams(text, n=4):
    """Count word n-grams in text"""
    words = text.split()
    if len(words) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def has_severe_repetition(text, min_words=40, threshold=8):
    """Check if text has severe repetition (most frequent 4-gram >= threshold)"""
    words = text.split()
    if len(words) < min_words:
        return False
    
    ngrams = count_word_ngrams(text, n=4)
    if not ngrams:
        return False
    
    max_freq = max(ngrams.values())
    return max_freq >= threshold


def compute_german_leakage(text, leak_words_set):
    """Compute German word leakage for a translation"""
    words = text.split()
    if not words:
        return 0.0, 0
    
    leak_count = 0
    for word in words:
        # Strip punctuation and lowercase
        clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word, flags=re.UNICODE).lower()
        if clean_word in leak_words_set:
            leak_count += 1
    
    leak_ratio = leak_count / len(words)
    return leak_ratio, leak_count


def has_devanagari(text):
    """Check if text contains any Devanagari character"""
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            return True
    return False


def pack_sentences(sentences, tokenizer, max_tokens=120):
    """Pack sentences into units not exceeding max_tokens"""
    units = []
    current_unit = []
    current_tokens = 0
    
    for sentence in sentences:
        # Count tokens for this sentence
        tokens = tokenizer(sentence, add_special_tokens=False, truncation=False)
        sent_tokens = len(tokens['input_ids'])
        
        # If adding this sentence would exceed max_tokens, start new unit
        if current_unit and current_tokens + sent_tokens > max_tokens:
            units.append(' '.join(current_unit))
            current_unit = [sentence]
            current_tokens = sent_tokens
        else:
            current_unit.append(sentence)
            current_tokens += sent_tokens
    
    # Add remaining unit
    if current_unit:
        units.append(' '.join(current_unit))
    
    return units


def translate_units(units, lang_code, lang_id, eos_id, eng_id, tokenizer, model, device, batch_size=32):
    """
    Translate a list of text units with proper language ID prefix.
    
    Returns:
        translations: List of translated texts in original order
        fallback_used: Count of units that required beam search fallback
    """
    if not units:
        return [], 0
    
    # Create items with original index and text
    unit_items = [(i, text) for i, text in enumerate(units)]
    
    # Tokenize to get lengths
    lengths = []
    for i, text in unit_items:
        tokens = tokenizer(text, add_special_tokens=False, truncation=True, max_length=510)
        lengths.append(len(tokens['input_ids']))
    
    # Sort by length descending, keeping (original_index, text) together
    sorted_items = sorted(zip(unit_items, lengths), key=lambda x: x[1], reverse=True)
    
    translations = [None] * len(units)
    fallback_used = 0
    
    # Process in batches
    for batch_start in range(0, len(sorted_items), batch_size):
        batch_slice = sorted_items[batch_start:batch_start + batch_size]
        batch_indices = [item[0][0] for item in batch_slice]  # Original indices
        batch_texts = [item[0][1] for item in batch_slice]   # Texts
        
        # Build input_ids manually with language ID prefix
        batch_input_ids = []
        batch_lengths = []
        
        for text in batch_texts:
            tokens = tokenizer(text, add_special_tokens=False, truncation=True, max_length=510)
            ids = [lang_id] + tokens['input_ids'] + [eos_id]
            
            # Assert language ID is present
            assert ids[0] == lang_id, f"Language ID {lang_id} not at start of sequence"
            assert lang_id != tokenizer.unk_token_id, f"Language ID is unk_token_id"
            
            batch_input_ids.append(ids)
            batch_lengths.append(len(ids))
        
        # Pad to batch maximum
        max_len = max(batch_lengths)
        padded_ids = []
        attention_masks = []
        
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded = ids + [tokenizer.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
        
        # Calculate max_new_tokens based on batch max input length
        batch_max_input = max(batch_lengths)
        max_new_tokens = min(512, int(2.0 * batch_max_input) + 10)
        
        # Translate with greedy decoding
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=eng_id,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False
            )
        
        batch_translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Check for severe repetition and retranslate if needed
        for i, (orig_idx, translation) in enumerate(zip(batch_indices, batch_translations)):
            if has_severe_repetition(translation, min_words=40, threshold=8):
                # Retranslate this one with beam search and no_repeat_ngram_size
                single_input_ids = input_ids[i:i+1]
                single_attention_mask = attention_mask[i:i+1]
                
                with torch.no_grad():
                    generated_fallback = model.generate(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        forced_bos_token_id=eng_id,
                        max_new_tokens=max_new_tokens,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        do_sample=False
                    )
                
                translation = tokenizer.batch_decode(generated_fallback, skip_special_tokens=True)[0]
                fallback_used += 1
            
            translations[orig_idx] = translation
    
    return translations, fallback_used


def bootstrap_ci(differences, n_resamples=2000, confidence=0.95):
    """Compute bootstrap confidence interval for mean difference"""
    rng = np.random.RandomState(42)
    bootstrap_means = []
    
    for _ in range(n_resamples):
        resample = rng.choice(differences, size=len(differences), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper


def evaluate_queries(language, v1_queries, fixed_queries, model_name, model_id, config, device):
    """Evaluate v1 vs fixed queries for one model"""
    # Load existing translated corpus index
    index_path = Path('embeddings') / f"{model_name}_{language}_translated_en_nllb.index"
    if not index_path.exists():
        print(f"  ERROR: Index not found: {index_path}")
        return None
    
    index = faiss.read_index(str(index_path))
    
    # Load translated corpus to get doc_ids
    corpus_path = Path('translated_corpus') / f"corpus_{language}_translated_en_nllb.json"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        translated_corpus = json.load(f)
    doc_ids = [doc['_id'] for doc in translated_corpus]
    
    # Load qrels
    qrels = load_qrels(config, language)
    
    # Build query lists with IDs
    query_items = []
    for v1_q, fixed_q in zip(v1_queries, fixed_queries):
        query_id = v1_q['_id']
        if query_id in qrels:
            query_items.append({
                'id': query_id,
                'v1_text': v1_q['text'],
                'fixed_text': fixed_q['text']
            })
    
    if not query_items:
        print(f"  ERROR: No queries with qrels")
        return None
    
    # Load embedding model
    model = SentenceTransformer(model_id)
    model = model.to(device)
    
    # Encode v1 and fixed queries
    v1_texts = [apply_query_prefix(q['v1_text'], model_name) for q in query_items]
    fixed_texts = [apply_query_prefix(q['fixed_text'], model_name) for q in query_items]
    
    v1_embeddings = model.encode(v1_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    fixed_embeddings = model.encode(fixed_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    
    # Normalize
    v1_embeddings = v1_embeddings.astype(np.float32)
    fixed_embeddings = fixed_embeddings.astype(np.float32)
    faiss.normalize_L2(v1_embeddings)
    faiss.normalize_L2(fixed_embeddings)
    
    # Search
    _, v1_indices = index.search(v1_embeddings, 10)
    _, fixed_indices = index.search(fixed_embeddings, 10)
    
    # Compute metrics
    v1_ndcg = []
    v1_success = []
    fixed_ndcg = []
    fixed_success = []
    
    for i, query_item in enumerate(query_items):
        query_id = query_item['id']
        qrels_for_query = qrels[query_id]
        
        # V1 metrics
        v1_retrieved = [doc_ids[idx] for idx in v1_indices[i]]
        v1_metrics = compute_metrics(v1_retrieved, qrels_for_query, [10])
        v1_ndcg.append(v1_metrics['ndcg@10'])
        v1_success.append(v1_metrics['success@10'])
        
        # Fixed metrics
        fixed_retrieved = [doc_ids[idx] for idx in fixed_indices[i]]
        fixed_metrics = compute_metrics(fixed_retrieved, qrels_for_query, [10])
        fixed_ndcg.append(fixed_metrics['ndcg@10'])
        fixed_success.append(fixed_metrics['success@10'])
    
    # Compute differences and bootstrap CI
    ndcg_diff = [f - v for f, v in zip(fixed_ndcg, v1_ndcg)]
    ci_lower, ci_upper = bootstrap_ci(ndcg_diff, n_resamples=2000)
    
    result = {
        'model': model_name,
        'n_queries': len(query_items),
        'v1_ndcg': np.mean(v1_ndcg),
        'v1_success': np.mean(v1_success),
        'fixed_ndcg': np.mean(fixed_ndcg),
        'fixed_success': np.mean(fixed_success),
        'ndcg_diff': np.mean(ndcg_diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Translation pilot with tokenizer fixes")
    parser.add_argument('--smoke', action='store_true', help='Smoke test: 40 queries, 12 docs per language')
    parser.add_argument('--languages', nargs='+', default=['de', 'hi'], help='Languages to process')
    args = parser.parse_args()
    
    config = load_config()
    
    # Setup output directory
    output_dir = Path('logs') / 'pilot'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NLLB model
    print("Loading NLLB model...")
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = NllbTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
    
    if torch.cuda.is_available():
        model = model.half().cuda()
        device = 'cuda'
        print(f"Model loaded on GPU with fp16")
    else:
        device = 'cpu'
        print(f"Model loaded on CPU with fp32")
    
    model.eval()
    
    # Get token IDs
    eng_id = tokenizer.convert_tokens_to_ids('eng_Latn')
    eos_id = tokenizer.eos_token_id
    
    # ========== PART 1: QUERIES ==========
    print("\n" + "="*80)
    print("PART 1: QUERY TRANSLATION")
    print("="*80)
    
    query_results = {}
    
    for lang in args.languages:
        print(f"\nProcessing {lang} queries...")
        
        lang_code = NLLB_LANG_CODES[lang]
        lang_id = tokenizer.convert_tokens_to_ids(lang_code)
        
        assert lang_id != tokenizer.unk_token_id, f"Language ID for {lang} is unk_token_id"
        
        # Load source queries
        queries_dataset = load_dataset(config['query_dataset'], lang, split='test')
        queries_to_translate = list(queries_dataset)
        
        if args.smoke:
            queries_to_translate = queries_to_translate[:40]
        
        print(f"  Translating {len(queries_to_translate)} queries...")
        
        # Translate queries
        query_texts = [q['text'] for q in queries_to_translate]
        start_time = time.time()
        
        with torch.no_grad():
            translations, fallback_count = translate_units(
                query_texts, lang_code, lang_id, eos_id, eng_id,
                tokenizer, model, device, batch_size=32
            )
        
        elapsed = time.time() - start_time
        print(f"  Translated in {elapsed:.1f}s (fallback: {fallback_count})")
        
        # Save fixed queries
        fixed_queries = [
            {'_id': q['_id'], 'text': translation}
            for q, translation in zip(queries_to_translate, translations)
        ]
        
        output_file = output_dir / f"queries_{lang}_fixed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_queries, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_file}")
        
        # Load v1 queries for comparison
        v1_path = Path('translated_queries') / f'queries_{lang}_translated_en_nllb.json'
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1_queries = json.load(f)
        
        # Ensure same order and IDs
        v1_dict = {q['_id']: q for q in v1_queries}
        v1_queries_ordered = [v1_dict[q['_id']] for q in fixed_queries]
        
        query_results[lang] = {
            'v1': v1_queries_ordered,
            'fixed': fixed_queries
        }
        
        # For German: count Was queries
        if lang == 'de':
            was_source_count = sum(1 for q in queries_to_translate if q['text'].startswith('Was '))
            was_fixed_what = sum(1 for fq in fixed_queries if fq['text'].startswith('What'))
            was_fixed_was = sum(1 for fq in fixed_queries if fq['text'].startswith('Was '))
            
            query_results[lang]['was_analysis'] = {
                'source_was_count': was_source_count,
                'fixed_what_count': was_fixed_what,
                'fixed_was_count': was_fixed_was
            }
    
    # Evaluate queries
    eval_results = {}
    
    # In smoke mode, evaluate with all-mpnet-base-v2 only
    models_to_eval = ['all-mpnet-base-v2'] if args.smoke else ['all-mpnet-base-v2', 'bge-m3']
    
    print("\n" + "="*80)
    print("EVALUATING QUERIES")
    print("="*80)
    
    for model_name in models_to_eval:
        print(f"\nModel: {model_name}")
        model_id = config['models'][model_name]
        
        for lang in args.languages:
            print(f"  Language: {lang}")
            result = evaluate_queries(
                lang,
                query_results[lang]['v1'],
                query_results[lang]['fixed'],
                model_name,
                model_id,
                config,
                device
            )
            
            if result:
                if lang not in eval_results:
                    eval_results[lang] = {}
                eval_results[lang][model_name] = result
    
    # ========== PART 2: DOCUMENTS ==========
    print("\n" + "="*80)
    print("PART 2: DOCUMENT TRANSLATION")
    print("="*80)
    
    doc_results = {}
    
    for lang in args.languages:
        print(f"\nProcessing {lang} documents...")
        
        lang_code = NLLB_LANG_CODES[lang]
        lang_id = tokenizer.convert_tokens_to_ids(lang_code)
        
        # Load FULL source corpus for total word count
        corpus_dataset = load_dataset(config['corpus_dataset'], lang, split='test')
        full_corpus_list = list(corpus_dataset)
        
        # Compute total source words for the full corpus
        total_source_words = sum(len(doc['text'].split()) for doc in full_corpus_list)
        print(f"  Full corpus: {len(full_corpus_list)} docs, {total_source_words} total source words")
        
        # Filter to docs with >= 100 source words
        eligible_docs = []
        for doc in full_corpus_list:
            word_count = len(doc['text'].split())
            if word_count >= 100:
                eligible_docs.append(doc)
        
        # Sample documents
        sample_size = 12 if args.smoke else 150
        rng = random.Random(42)
        sampled_docs = rng.sample(eligible_docs, min(sample_size, len(eligible_docs)))
        
        print(f"  Sampled {len(sampled_docs)} documents (>= 100 words)")
        
        # Load v1 translations
        v1_path = Path('translated_corpus') / f'corpus_{lang}_translated_en_nllb.json'
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1_corpus = json.load(f)
        v1_dict = {doc['_id']: doc['text'] for doc in v1_corpus}
        
        # Prepare variants
        variants = {
            'v1': [],
            'W': [],
            'S': [],
            'P': []
        }
        
        variant_stats = {
            'v1': {'fallback_used': 0, 'time': 0},
            'W': {'fallback_used': 0, 'time': 0},
            'S': {'fallback_used': 0, 'time': 0},
            'P': {'fallback_used': 0, 'time': 0}
        }
        
        # V1: just collect from existing translations
        for doc in sampled_docs:
            variants['v1'].append({
                '_id': doc['_id'],
                'source': doc['text'],
                'translation': v1_dict[doc['_id']]
            })
        
        # W: whole document
        print(f"  Translating variant W (whole)...")
        start_time = time.time()
        with torch.no_grad():
            w_texts = [doc['text'] for doc in sampled_docs]
            w_translations, w_fallback = translate_units(
                w_texts, lang_code, lang_id, eos_id, eng_id,
                tokenizer, model, device, batch_size=8
            )
        variant_stats['W']['time'] = time.time() - start_time
        variant_stats['W']['fallback_used'] = w_fallback
        
        for doc, translation in zip(sampled_docs, w_translations):
            variants['W'].append({
                '_id': doc['_id'],
                'source': doc['text'],
                'translation': translation
            })
        
        # S: sentence by sentence
        print(f"  Translating variant S (sentences)...")
        start_time = time.time()
        s_fallback_total = 0
        
        for doc in sampled_docs:
            sentences = split_sentences(doc['text'])
            with torch.no_grad():
                s_translations, s_fallback = translate_units(
                    sentences, lang_code, lang_id, eos_id, eng_id,
                    tokenizer, model, device, batch_size=32
                )
            s_fallback_total += s_fallback
            
            combined = ' '.join(s_translations)
            variants['S'].append({
                '_id': doc['_id'],
                'source': doc['text'],
                'translation': combined
            })
        
        variant_stats['S']['time'] = time.time() - start_time
        variant_stats['S']['fallback_used'] = s_fallback_total
        
        # P: packed sentences
        print(f"  Translating variant P (packed)...")
        start_time = time.time()
        p_fallback_total = 0
        
        for doc in sampled_docs:
            sentences = split_sentences(doc['text'])
            units = pack_sentences(sentences, tokenizer, max_tokens=120)
            with torch.no_grad():
                p_translations, p_fallback = translate_units(
                    units, lang_code, lang_id, eos_id, eng_id,
                    tokenizer, model, device, batch_size=16
                )
            p_fallback_total += p_fallback
            
            combined = ' '.join(p_translations)
            variants['P'].append({
                '_id': doc['_id'],
                'source': doc['text'],
                'translation': combined
            })
        
        variant_stats['P']['time'] = time.time() - start_time
        variant_stats['P']['fallback_used'] = p_fallback_total
        
        # Save all variants
        for variant_name, variant_docs in variants.items():
            output_file = output_dir / f"docs_{lang}_{variant_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(variant_docs, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved all variants to {output_dir}")
        
        # Compute sample source words
        sample_source_words = sum(len(doc['text'].split()) for doc in sampled_docs)
        
        # Compute statistics per variant
        doc_results[lang] = {
            'n': len(sampled_docs),
            'total_source_words': total_source_words,
            'sample_source_words': sample_source_words,
            'variants': {},
            'docs': {}  # Store actual variant docs for examples
        }
        
        # Store the variant docs for later access
        for variant_name, variant_docs in variants.items():
            doc_results[lang]['docs'][variant_name] = variant_docs
        
        # German leak words
        german_leak_words = {
            'und', 'ist', 'nicht', 'mit', 'das', 'dem', 'den', 'eine', 'einen',
            'für', 'auf', 'im', 'des', 'sich', 'auch', 'wird', 'werden', 'bei',
            'nach', 'dass', 'zum', 'zur', 'sind', 'oder', 'aber', 'wurde', 'wurden'
        }
        
        for variant_name, variant_docs in variants.items():
            # Compute word ratios
            ratios = []
            ratio_by_bucket = {'101-200': [], '201-300': [], '>300': []}
            under_0_5 = 0
            severe_rep = 0
            
            # Leakage metrics
            leak_ratios = []
            leak_3plus_count = 0
            devanagari_count = 0
            
            for vdoc in variant_docs:
                source_words = len(vdoc['source'].split())
                trans_words = len(vdoc['translation'].split())
                ratio = trans_words / max(1, source_words)
                ratios.append((vdoc['_id'], ratio))
                
                if ratio < 0.5:
                    under_0_5 += 1
                
                # Bucket by source length
                if 101 <= source_words <= 200:
                    ratio_by_bucket['101-200'].append(ratio)
                elif 201 <= source_words <= 300:
                    ratio_by_bucket['201-300'].append(ratio)
                elif source_words > 300:
                    ratio_by_bucket['>300'].append(ratio)
                
                # Severe repetition check
                ngrams = count_word_ngrams(vdoc['translation'], n=4)
                if ngrams and max(ngrams.values()) >= 20:
                    severe_rep += 1
                
                # Leakage metrics
                if lang == 'de':
                    leak_ratio, leak_count = compute_german_leakage(vdoc['translation'], german_leak_words)
                    leak_ratios.append(leak_ratio)
                    if leak_count >= 3:
                        leak_3plus_count += 1
                elif lang == 'hi':
                    if has_devanagari(vdoc['translation']):
                        devanagari_count += 1
            
            # Sort by ratio to find lowest
            ratios_sorted = sorted(ratios, key=lambda x: x[1])
            
            # Compute projection based on words
            variant_total_seconds = variant_stats[variant_name]['time']
            if sample_source_words > 0:
                projected_hours = (variant_total_seconds / sample_source_words) * total_source_words / 3600
            else:
                projected_hours = 0
            
            doc_results[lang]['variants'][variant_name] = {
                'median_ratio': np.median([r for _, r in ratios]),
                'ratio_by_bucket': {
                    bucket: {
                        'median': np.median(vals) if vals else None,
                        'n': len(vals)
                    }
                    for bucket, vals in ratio_by_bucket.items()
                },
                'under_0_5_count': under_0_5,
                'under_0_5_share': 100.0 * under_0_5 / len(variant_docs),
                'severe_rep_count': severe_rep,
                'lowest_ratio_ids': [doc_id for doc_id, _ in ratios_sorted[:3]],
                'lowest_ratio_values': [ratio for _, ratio in ratios_sorted[:3]],
                'fallback_used': variant_stats[variant_name]['fallback_used'],
                'total_seconds': variant_stats[variant_name]['time'],
                'seconds_per_doc': variant_stats[variant_name]['time'] / len(variant_docs) if variant_docs else 0,
                'projected_hours': projected_hours
            }
            
            # Add language-specific leakage metrics
            if lang == 'de':
                doc_results[lang]['variants'][variant_name]['mean_leak'] = np.mean(leak_ratios) if leak_ratios else 0
                doc_results[lang]['variants'][variant_name]['leak_3plus_count'] = leak_3plus_count
            elif lang == 'hi':
                doc_results[lang]['variants'][variant_name]['devanagari_count'] = devanagari_count
        
        # Compute full corpus leakage for v1
        if lang == 'de':
            print(f"  Computing full corpus German leakage...")
            v1_path = Path('translated_corpus') / f'corpus_{lang}_translated_en_nllb.json'
            with open(v1_path, 'r', encoding='utf-8') as f:
                v1_full_corpus = json.load(f)
            
            full_leak_ratios = []
            full_leak_3plus_count = 0
            
            for doc in v1_full_corpus:
                leak_ratio, leak_count = compute_german_leakage(doc['text'], german_leak_words)
                full_leak_ratios.append(leak_ratio)
                if leak_count >= 3:
                    full_leak_3plus_count += 1
            
            doc_results[lang]['v1_full_corpus'] = {
                'mean_leak': np.mean(full_leak_ratios),
                'leak_3plus_count': full_leak_3plus_count,
                'n': len(v1_full_corpus)
            }
        
        elif lang == 'hi':
            print(f"  Computing full corpus Hindi Devanagari leakage...")
            v1_path = Path('translated_corpus') / f'corpus_{lang}_translated_en_nllb.json'
            with open(v1_path, 'r', encoding='utf-8') as f:
                v1_full_corpus = json.load(f)
            
            full_devanagari_count = 0
            for doc in v1_full_corpus:
                if has_devanagari(doc['text']):
                    full_devanagari_count += 1
            
            doc_results[lang]['v1_full_corpus'] = {
                'devanagari_count': full_devanagari_count,
                'n': len(v1_full_corpus)
            }
    
    # ========== REPORT ==========
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("="*120)
    
    # PART 1: Query results
    print("\nPART 1: QUERY TRANSLATION RESULTS")
    print("="*120)
    
    if eval_results:
        print("\nQuery Evaluation (v1 vs fixed):")
        print(f"{'Lang':<6} {'Model':<25} {'V1_nDCG':<10} {'V1_Succ':<10} {'Fixed_nDCG':<12} {'Fixed_Succ':<12} {'Diff':<10} {'95% CI':<20}")
        
        for lang in args.languages:
            if lang in eval_results:
                for model_name in sorted(eval_results[lang].keys()):
                    r = eval_results[lang][model_name]
                    ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                    print(f"{lang:<6} {model_name:<25} {r['v1_ndcg']:<10.4f} {r['v1_success']:<10.4f} "
                          f"{r['fixed_ndcg']:<12.4f} {r['fixed_success']:<12.4f} {r['ndcg_diff']:<10.4f} {ci_str:<20}")
    
    # German Was analysis
    print("\nGerman 'Was' Query Analysis:")
    if 'de' in query_results and 'was_analysis' in query_results['de']:
        wa = query_results['de']['was_analysis']
        print(f"  Source queries starting with 'Was ': {wa['source_was_count']}")
        print(f"  Fixed translations starting with 'What': {wa['fixed_what_count']}")
        print(f"  Fixed translations still starting with 'Was ': {wa['fixed_was_count']}")
    
    # PART 2: Document results
    print("\n" + "="*120)
    print("PART 2: DOCUMENT TRANSLATION RESULTS")
    print("="*120)
    
    for lang in args.languages:
        if lang not in doc_results:
            continue
        
        print(f"\n{lang.upper()} Documents (n={doc_results[lang]['n']}):")
        print("-" * 120)
        
        # Summary table
        if lang == 'de':
            print(f"\n{'Variant':<8} {'MedianRatio':<12} {'<0.5 Count':<12} {'<0.5 Share%':<12} {'SevereRep':<12} "
                  f"{'MeanLeak':<10} {'Leak3+':<8} {'Fallback':<10} {'TotalSec':<10} {'Sec/Doc':<10} {'ProjHoursByWords':<16}")
            
            for variant_name in ['v1', 'W', 'S', 'P']:
                v = doc_results[lang]['variants'][variant_name]
                print(f"{variant_name:<8} {v['median_ratio']:<12.3f} {v['under_0_5_count']:<12} "
                      f"{v['under_0_5_share']:<12.2f} {v['severe_rep_count']:<12} "
                      f"{v['mean_leak']:<10.4f} {v['leak_3plus_count']:<8} "
                      f"{v['fallback_used']:<10} {v['total_seconds']:<10.1f} "
                      f"{v['seconds_per_doc']:<10.2f} {v['projected_hours']:<16.1f}")
        
        elif lang == 'hi':
            print(f"\n{'Variant':<8} {'MedianRatio':<12} {'<0.5 Count':<12} {'<0.5 Share%':<12} {'SevereRep':<12} "
                  f"{'Devanagari':<12} {'Fallback':<10} {'TotalSec':<10} {'Sec/Doc':<10} {'ProjHoursByWords':<16}")
            
            for variant_name in ['v1', 'W', 'S', 'P']:
                v = doc_results[lang]['variants'][variant_name]
                print(f"{variant_name:<8} {v['median_ratio']:<12.3f} {v['under_0_5_count']:<12} "
                      f"{v['under_0_5_share']:<12.2f} {v['severe_rep_count']:<12} "
                      f"{v['devanagari_count']:<12} "
                      f"{v['fallback_used']:<10} {v['total_seconds']:<10.1f} "
                      f"{v['seconds_per_doc']:<10.2f} {v['projected_hours']:<16.1f}")
        
        # Ratio by bucket
        print(f"\nRatio by Source Length Bucket:")
        print(f"{'Variant':<8} {'101-200':<20} {'201-300':<20} {'>300':<20}")
        
        for variant_name in ['v1', 'W', 'S', 'P']:
            v = doc_results[lang]['variants'][variant_name]
            parts = []
            for bucket in ['101-200', '201-300', '>300']:
                info = v['ratio_by_bucket'][bucket]
                if info['median'] is not None:
                    parts.append(f"{info['median']:.3f} (n={info['n']})")
                else:
                    parts.append(f"N/A (n={info['n']})")
            print(f"{variant_name:<8} {parts[0]:<20} {parts[1]:<20} {parts[2]:<20}")
        
        # Full corpus leakage
        if 'v1_full_corpus' in doc_results[lang]:
            if lang == 'de':
                fc = doc_results[lang]['v1_full_corpus']
                print(f"\nv1 full German corpus leakage (n={fc['n']}):")
                print(f"  Mean leak: {fc['mean_leak']:.4f}")
                print(f"  Documents with 3+ leak words: {fc['leak_3plus_count']}")
            elif lang == 'hi':
                fc = doc_results[lang]['v1_full_corpus']
                print(f"\nv1 full Hindi corpus Devanagari leakage (n={fc['n']}):")
                print(f"  Documents with Devanagari characters: {fc['devanagari_count']}")
        
        # Lowest ratio examples - use stored docs
        print(f"\nLowest Ratio Examples:")
        
        for variant_name in ['v1', 'W', 'S', 'P']:
            v = doc_results[lang]['variants'][variant_name]
            variant_docs = doc_results[lang]['docs'][variant_name]
            
            print(f"\n{variant_name}:")
            for doc_id, ratio in zip(v['lowest_ratio_ids'], v['lowest_ratio_values']):
                vdoc = next((d for d in variant_docs if d['_id'] == doc_id), None)
                if vdoc:
                    print(f"  ID: {doc_id}, Ratio: {ratio:.3f}")
                    print(f"    Source: {vdoc['source'][:250]}")
                    print(f"    Translation: {vdoc['translation'][:250]}")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    with torch.no_grad():
        main()
