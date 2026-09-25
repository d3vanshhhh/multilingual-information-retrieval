"""
Translation utilities for NLLB-200 model.

Main function:
    translate_to_english(text, src_lang, model_name, cache): Translate text to English
        with caching support.

Supported models:
    - "nllb": facebook/nllb-200-distilled-600M

Supported languages: 16 (en, de, hi, and 13 others - see NLLB_LANG_CODES)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings


# Language code mappings for NLLB
NLLB_LANG_CODES = {
    'bg': 'bul_Cyrl', 'bn': 'ben_Beng', 'cs': 'ces_Latn', 'da': 'dan_Latn',
    'de': 'deu_Latn', 'en': 'eng_Latn', 'fa': 'pes_Arab', 'fi': 'fin_Latn',
    'hi': 'hin_Deva', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'no': 'nob_Latn',
    'pt': 'por_Latn', 'ro': 'ron_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn'
}

# Global model cache to avoid reloading
_translation_models = {}


def translate_to_english(text, src_lang, model_name="nllb", cache=None):
    """
    Translate text to English using NLLB.
    
    Args:
        text: Text to translate
        src_lang: Source language code (2-letter ISO)
        model_name: Must be "nllb"
        cache: Optional TranslationCache instance
    
    Returns:
        Translated text (or original if translation fails)
    """
    if model_name != "nllb":
        raise ValueError(f"Only 'nllb' model is supported, got: {model_name}")
    
    # Check cache first
    if cache is not None:
        cached = cache.get(text, src_lang, model_name)
        if cached is not None:
            return cached
    
    # If already English, return as-is
    if src_lang == 'en':
        if cache is not None:
            cache.set(text, src_lang, model_name, text)
        return text
    
    try:
        # Load model if not already loaded
        if model_name not in _translation_models:
            print(f"[Translation] Loading {model_name} model...")
            model_id = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Use safetensors to avoid torch.load security issue
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
            
            _translation_models[model_name] = (tokenizer, model)
            print(f"[Translation] {model_name} loaded successfully")
        
        tokenizer, model = _translation_models[model_name]
        
        # Get language codes
        src_code = NLLB_LANG_CODES.get(src_lang)
        tgt_code = NLLB_LANG_CODES['en']
        
        if src_code is None:
            warnings.warn(f"Language {src_lang} not supported by NLLB, returning original")
            return text
        
        # Set source language for tokenizer
        tokenizer.src_lang = src_code
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get the token ID for the target language
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Generate translation
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512
        )
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Cache the result
        if cache is not None:
            cache.set(text, src_lang, model_name, translation)
        
        return translation
    
    except Exception as e:
        warnings.warn(f"Translation failed for {src_lang} -> en: {e}")
        return text


def translate_batch_to_english(texts, src_lang, model_name="nllb", cache=None, batch_size=8):
    """
    Batched GPU translation with fp16 (falls back to fp32 CPU if no CUDA).
    
    Args:
        texts: List of texts to translate
        src_lang: Source language code (2-letter ISO)
        model_name: Must be "nllb"
        cache: Optional TranslationCache instance
        batch_size: Batch size for translation (default: 8)
    
    Returns:
        List of translated texts in the same order as input
    """
    if model_name != "nllb":
        raise ValueError(f"Only 'nllb' model is supported, got: {model_name}")
    
    # If already English, return as-is and cache as identity
    if src_lang == 'en':
        if cache is not None:
            for text in texts:
                cache.set(text, src_lang, model_name, text)
        return texts
    
    # Check cache first, separate cache hits from misses
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
        return translations
    
    try:
        # Load model if not already loaded
        if model_name not in _translation_models:
            print(f"[Translation] Loading {model_name} model...")
            model_id = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
            
            # Move to GPU with fp16 if available, otherwise keep CPU fp32
            if torch.cuda.is_available():
                model = model.half().cuda()
                print(f"[Translation] {model_name} loaded on GPU with fp16")
            else:
                print(f"[Translation] {model_name} loaded on CPU with fp32")
            
            _translation_models[model_name] = (tokenizer, model)
        
        tokenizer, model = _translation_models[model_name]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get language codes
        src_code = NLLB_LANG_CODES.get(src_lang)
        tgt_code = NLLB_LANG_CODES['en']
        
        if src_code is None:
            warnings.warn(f"Language {src_lang} not supported by NLLB, returning originals")
            for i, text in zip(cache_miss_indices, cache_miss_texts):
                translations[i] = text
            return translations
        
        tokenizer.src_lang = src_code
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Process cache misses in batches
        new_translations = []
        for batch_start in range(0, len(cache_miss_texts), batch_size):
            batch_texts = cache_miss_texts[batch_start:batch_start + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512
                )
            
            # Decode batch
            batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Store in cache immediately (for resumability)
            if cache is not None:
                for text, translation in zip(batch_texts, batch_translations):
                    cache.set(text, src_lang, model_name, translation)
            
            new_translations.extend(batch_translations)
        
        # Map new translations back to original positions
        for i, translation in zip(cache_miss_indices, new_translations):
            translations[i] = translation
        
        return translations
    
    except Exception as e:
        warnings.warn(f"Batch translation failed for {src_lang} -> en: {e}")
        # Return originals for failed translations
        for i, text in zip(cache_miss_indices, cache_miss_texts):
            if translations[i] is None:
                translations[i] = text
        return translations
