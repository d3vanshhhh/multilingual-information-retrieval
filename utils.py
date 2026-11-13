"""
Translation utilities for NLLB-200 model.

Main function:
    translate_batch_to_english(texts, src_lang, model_name, cache): Translate texts to English
        with caching support and sentence-level translation.

Supported models:
    - "nllb-v2": facebook/nllb-200-distilled-600M (sentence-level with loop guard)

Supported languages: 16 (en, de, hi, and 13 others - see NLLB_LANG_CODES)
"""
import re
import time
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import transformers.utils.logging
from collections import Counter

# Set transformers logging to error level
transformers.utils.logging.set_verbosity_error()

# Translation tag and model
TRANSLATION_TAG = "nllb-v2"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Language code mappings for NLLB
NLLB_LANG_CODES = {
    'bg': 'bul_Cyrl', 'bn': 'ben_Beng', 'cs': 'ces_Latn', 'da': 'dan_Latn',
    'de': 'deu_Latn', 'en': 'eng_Latn', 'fa': 'pes_Arab', 'fi': 'fin_Latn',
    'hi': 'hin_Deva', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'no': 'nob_Latn',
    'pt': 'por_Latn', 'ro': 'ron_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn'
}

# Global model cache for lazy loading
_tokenizer = None
_model = None
_device = None

# Module-level stats updated on every call
LAST_STATS = {
    'texts': 0,
    'sentences': 0,
    'loop_fallbacks': 0,
    'empty_sentence_outputs': 0,
    'seconds': 0.0
}


# Abbreviations that should not trigger sentence boundaries
ABBREVIATIONS = {
    'dr.', 'prof.', 'hrsg.', 'nr.', 'ca.', 'bzw.', 'vgl.', 'usw.', 'etc.', 
    'st.', 'jh.', 'geb.', 'gest.', 'ggf.', 'inkl.', 'evtl.', 'max.', 'min.', 
    'mio.', 'mrd.', 'tel.', 'str.', 'bd.', 'vol.', 'no.', 'pp.', 'ed.', 
    'approx.', 'sr.', 'sra.', 'av.', 'tzv.', 'např.', 'atd.', 'apod.', 
    'resp.', 't.ex.', 'f.eks.', 'osv.', 'bl.a.', 'dvs.', 'bijv.', 'enz.', 
    'esim.', 'jne.', 'ym.', 'напр.', 'бр.', 'стр.', 'нпр.', 'год.'
}


def split_sentences(text):
    """
    Split text into sentences with boundary-aware merging.
    
    This is a heuristic splitter that:
    1. Splits on sentence terminators (.!?।॥؟۔…) followed by whitespace
    2. Re-merges false boundaries caused by abbreviations, numbers, or initials
    3. Merges pieces with fewer than 3 words into the previous piece
    
    A false merge only produces a slightly longer translation unit.
    """
    # Split on common sentence terminators
    pieces = re.split(r'(?<=[.!?।॥؟۔…])\s+', text.strip())
    pieces = [p for p in pieces if p]  # Drop empty
    
    if not pieces:
        return []
    
    # Re-merge false boundaries from abbreviations, numbers, and initials
    boundary_aware = [pieces[0]]
    for piece in pieces[1:]:
        if boundary_aware:
            # Get the last token of the previous piece
            prev_tokens = boundary_aware[-1].split()
            if prev_tokens:
                last_token = prev_tokens[-1].lower()
                
                # Check if last token is:
                # (a) 1-3 digits followed by period (e.g., "24.", "1.")
                # (b) single letter followed by period (e.g., "J.", "г.")
                # (c) in ABBREVIATIONS set
                is_false_boundary = (
                    re.match(r'^\d{1,3}\.$', last_token) or
                    re.match(r'^[a-zα-ωа-яґєії]\.$', last_token) or
                    last_token in ABBREVIATIONS
                )
                
                if is_false_boundary:
                    # Merge with previous piece
                    boundary_aware[-1] = boundary_aware[-1] + ' ' + piece
                    continue
        
        # Not a false boundary, add as new piece
        boundary_aware.append(piece)
    
    # Merge pieces with fewer than 3 words into previous
    merged = [boundary_aware[0]]
    for piece in boundary_aware[1:]:
        if len(piece.split()) < 3:
            merged[-1] = merged[-1] + ' ' + piece
        else:
            merged.append(piece)
    
    return merged


def _load_translation_model():
    """Lazily load NLLB model once at module level."""
    global _tokenizer, _model, _device
    
    if _tokenizer is None:
        print(f"[Translation] Loading {TRANSLATION_TAG} model...")
        _tokenizer = NllbTokenizer.from_pretrained(NLLB_MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID, use_safetensors=True)
        
        # Move to GPU with fp16 if available, otherwise keep CPU fp32
        if torch.cuda.is_available():
            _model = _model.half().cuda()
            _device = 'cuda'
            print(f"[Translation] {TRANSLATION_TAG} loaded on GPU with fp16")
        else:
            _device = 'cpu'
            print(f"[Translation] {TRANSLATION_TAG} loaded on CPU with fp32")
        
        _model.eval()
        print(f"[Translation] Model ready")


def _has_severe_repetition(text, min_words=40, threshold=8):
    """Check if text has severe repetition (most frequent 4-gram >= threshold)."""
    words = text.split()
    if len(words) < min_words:
        return False
    
    # Count 4-grams
    ngrams = []
    for i in range(len(words) - 3):
        ngram = tuple(words[i:i+4])
        ngrams.append(ngram)
    
    if not ngrams:
        return False
    
    ngram_counts = Counter(ngrams)
    max_freq = max(ngram_counts.values())
    return max_freq >= threshold


def translate_batch_to_english(texts, src_lang, model_name="nllb-v2", cache=None):
    """
    Translate texts to English with sentence-level translation and loop guard.
    
    Args:
        texts: List of texts to translate
        src_lang: Source language code (2-letter ISO)
        model_name: Must be "nllb-v2"
        cache: Optional TranslationCache instance
    
    Returns:
        List of translated texts in the same order as input
    """
    start_time = time.time()
    
    if model_name != TRANSLATION_TAG:
        raise ValueError(f"Only '{TRANSLATION_TAG}' model is supported, got: {model_name}")
    
    # If already English, return as-is and cache as identity
    if src_lang == 'en':
        if cache is not None:
            for text in texts:
                cache.set(text, src_lang, model_name, text)
        LAST_STATS.update({
            'texts': len(texts),
            'sentences': 0,
            'loop_fallbacks': 0,
            'empty_sentence_outputs': 0,
            'seconds': 0.0
        })
        return texts
    
    # Load model
    _load_translation_model()
    
    # Get language codes and IDs
    src_code = NLLB_LANG_CODES.get(src_lang)
    if src_code is None:
        raise ValueError(f"Language {src_lang} not supported by NLLB")
    
    lang_id = _tokenizer.convert_tokens_to_ids(src_code)
    eng_id = _tokenizer.convert_tokens_to_ids('eng_Latn')
    eos_id = _tokenizer.eos_token_id
    pad_id = _tokenizer.pad_token_id
    
    # Validate language ID
    if lang_id == _tokenizer.unk_token_id:
        raise RuntimeError(f"Language ID for {src_lang} is unk_token_id")
    
    # Check cache and separate hits from misses
    translations = [None] * len(texts)
    cache_miss_indices = []
    cache_miss_texts = []
    
    for i, text in enumerate(texts):
        if cache is not None:
            cached = cache.get(text, src_lang, model_name)
            if cached is not None:
                translations[i] = cached
            else:
                cache_miss_indices.append(i)
                cache_miss_texts.append(text)
        else:
            cache_miss_indices.append(i)
            cache_miss_texts.append(text)
    
    # If no cache misses, return all cached translations
    if not cache_miss_texts:
        LAST_STATS.update({
            'texts': len(texts),
            'sentences': 0,
            'loop_fallbacks': 0,
            'empty_sentence_outputs': 0,
            'seconds': time.time() - start_time
        })
        return translations
    
    # Split all missing texts into sentences
    all_sentences = []   # Flat list of all sentences for batching
    text_sentence_ranges = []  # List of (text_idx, start, end) for each missing text
    
    total_sentences = 0
    for text_idx, text in zip(cache_miss_indices, cache_miss_texts):
        sentences = split_sentences(text)
        start_idx = len(all_sentences)
        all_sentences.extend(sentences)
        end_idx = len(all_sentences)
        text_sentence_ranges.append((text_idx, start_idx, end_idx))
        total_sentences += len(sentences)
    
    # Verify sentence count after flattening
    assert len(all_sentences) == total_sentences
    
    # Tokenize all sentences to get lengths
    sentence_lengths = []
    for sent in all_sentences:
        tokens = _tokenizer(sent, add_special_tokens=False, truncation=True, max_length=510)
        sentence_lengths.append(len(tokens['input_ids']))
    
    # Sort sentences by token length descending
    sorted_indices = sorted(range(len(all_sentences)), key=lambda i: sentence_lengths[i], reverse=True)
    
    # Form batches greedily
    batches = []  # Each batch is a list of (original_sentence_idx, sentence, length)
    current_batch = []
    current_batch_max_len = 0
    
    for sent_idx in sorted_indices:
        sent = all_sentences[sent_idx]
        sent_len = sentence_lengths[sent_idx]
        
        # Check if adding this sentence would exceed limits
        if current_batch:
            new_max_len = max(current_batch_max_len, sent_len)
            would_exceed = (len(current_batch) * new_max_len > 4096) or (len(current_batch) >= 64)
            
            if would_exceed:
                batches.append(current_batch)
                current_batch = [(sent_idx, sent, sent_len)]
                current_batch_max_len = sent_len
            else:
                current_batch.append((sent_idx, sent, sent_len))
                current_batch_max_len = new_max_len
        else:
            current_batch = [(sent_idx, sent, sent_len)]
            current_batch_max_len = sent_len
    
    if current_batch:
        batches.append(current_batch)
    
    # Translate all batches
    sentence_translations = [None] * len(all_sentences)
    loop_fallback_count = 0
    empty_sentence_output_count = 0
    
    for batch in batches:
        batch_indices = [item[0] for item in batch]
        batch_sentences = [item[1] for item in batch]
        batch_lengths = [item[2] for item in batch]
        
        # Build input_ids manually with language ID prefix
        batch_input_ids = []
        for sent in batch_sentences:
            tokens = _tokenizer(sent, add_special_tokens=False, truncation=True, max_length=510)
            ids = [lang_id] + tokens['input_ids'] + [eos_id]
            
            # Validate language ID is present
            if ids[0] != lang_id:
                raise RuntimeError(f"Language ID {lang_id} not at start of sequence")
            
            batch_input_ids.append(ids)
        
        # Pad to batch maximum
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        attention_masks = []
        
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded = ids + [pad_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(_device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(_device)
        
        # Calculate max_new_tokens based on batch max input length
        batch_max_input = max(batch_lengths)
        max_new_tokens = min(512, int(2.0 * batch_max_input) + 10)
        
        # Greedy decoding
        with torch.no_grad():
            generated = _model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=eng_id,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False
            )
        
        batch_translations = _tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Check for severe repetition and retranslate if needed (loop guard)
        for i, (orig_idx, translation) in enumerate(zip(batch_indices, batch_translations)):
            # Check for empty output from non-empty source
            if batch_sentences[i].strip() and not translation.strip():
                empty_sentence_output_count += 1
            
            if _has_severe_repetition(translation, min_words=40, threshold=8):
                # Retranslate this one with beam search and no_repeat_ngram_size
                single_input_ids = input_ids[i:i+1]
                single_attention_mask = attention_mask[i:i+1]
                
                with torch.no_grad():
                    generated_fallback = _model.generate(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        forced_bos_token_id=eng_id,
                        max_new_tokens=max_new_tokens,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        do_sample=False
                    )
                
                translation = _tokenizer.batch_decode(generated_fallback, skip_special_tokens=True)[0]
                loop_fallback_count += 1
            
            sentence_translations[orig_idx] = translation
    
    # Verify all sentences were translated
    assert len(sentence_translations) == len(all_sentences)
    
    # Re-join sentences for each text
    # First verify all sentences were translated
    for i, trans in enumerate(sentence_translations):
        if trans is None:
            raise RuntimeError(f"Sentence at index {i} was not translated")
    
    for text_idx, start_idx, end_idx in text_sentence_ranges:
        # Join this text's sentence translations
        full_translation = ' '.join(sentence_translations[start_idx:end_idx])
        translations[text_idx] = full_translation
        
        # Cache the result
        if cache is not None:
            original_text = texts[text_idx]
            cache.set(original_text, src_lang, model_name, full_translation)
    
    elapsed = time.time() - start_time
    
    # Update module-level stats
    LAST_STATS.update({
        'texts': len(texts),
        'sentences': total_sentences,
        'loop_fallbacks': loop_fallback_count,
        'empty_sentence_outputs': empty_sentence_output_count,
        'seconds': elapsed
    })
    
    return translations
