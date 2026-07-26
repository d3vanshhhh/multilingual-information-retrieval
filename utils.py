"""
Translation utilities for NLLB-200 and M2M100 models.

Main function:
    translate_to_english(text, src_lang, model_name, cache): Translate text to English
        with caching support. Supports NLLB-200 and M2M100 translation models.

Supported models:
    - "nllb": facebook/nllb-200-distilled-600M (faster, 600M params)
    - "m2m100": facebook/m2m100_418M (slower, higher quality)

Supported languages: 16 (en, de, hi, and 13 others - see NLLB_LANG_CODES)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100Tokenizer, M2M100ForConditionalGeneration
import warnings


# Language code mappings for NLLB and M2M100
NLLB_LANG_CODES = {
    'bg': 'bul_Cyrl', 'bn': 'ben_Beng', 'cs': 'ces_Latn', 'da': 'dan_Latn',
    'de': 'deu_Latn', 'en': 'eng_Latn', 'fa': 'pes_Arab', 'fi': 'fin_Latn',
    'hi': 'hin_Deva', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'no': 'nob_Latn',
    'pt': 'por_Latn', 'ro': 'ron_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn'
}

M2M100_LANG_CODES = {
    'bg': 'bg', 'bn': 'bn', 'cs': 'cs', 'da': 'da',
    'de': 'de', 'en': 'en', 'fa': 'fa', 'fi': 'fi',
    'hi': 'hi', 'it': 'it', 'nl': 'nl', 'no': 'no',
    'pt': 'pt', 'ro': 'ro', 'sr': 'sr', 'sv': 'sv'
}

# Global model cache to avoid reloading
_translation_models = {}


def translate_to_english(text, src_lang, model_name="nllb", cache=None):
    """
    Translate text to English using NLLB or M2M100.
    
    Args:
        text: Text to translate
        src_lang: Source language code (2-letter ISO)
        model_name: "nllb" or "m2m100"
        cache: Optional TranslationCache instance
    
    Returns:
        Translated text (or original if translation fails)
    """
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
            if model_name == "nllb":
                model_id = "facebook/nllb-200-distilled-600M"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                # Use safetensors to avoid torch.load security issue
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
            elif model_name == "m2m100":
                model_id = "facebook/m2m100_418M"
                tokenizer = M2M100Tokenizer.from_pretrained(model_id)
                # Use safetensors to avoid torch.load security issue
                model = M2M100ForConditionalGeneration.from_pretrained(model_id, use_safetensors=True)
            else:
                raise ValueError(f"Unknown translation model: {model_name}")
            
            _translation_models[model_name] = (tokenizer, model)
            print(f"[Translation] {model_name} loaded successfully")
        
        tokenizer, model = _translation_models[model_name]
        
        # Get language codes
        if model_name == "nllb":
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
        
        elif model_name == "m2m100":
            src_code = M2M100_LANG_CODES.get(src_lang)
            tgt_code = M2M100_LANG_CODES['en']
            
            if src_code is None:
                warnings.warn(f"Language {src_lang} not supported by M2M100, returning original")
                return text
            
            # Set source language
            tokenizer.src_lang = src_code
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate with target language
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
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
