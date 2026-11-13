"""
Check NLLB tokenizer behavior with different language codes.

Tests whether setting tokenizer.src_lang affects tokenization and translation
for German and Hindi texts.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys


def main():
    print("Loading NLLB model...")
    model_id = "facebook/nllb-200-distilled-600M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
    
    # Move to GPU with fp16 if available, otherwise CPU fp32
    if torch.cuda.is_available():
        model = model.half().cuda()
        device = 'cuda'
        print(f"Model loaded on GPU with fp16")
    else:
        device = 'cpu'
        print(f"Model loaded on CPU with fp32")
    
    model.eval()
    
    print("\n" + "="*120)
    print("PACKAGE VERSIONS:")
    print(f"  transformers: {__import__('transformers').__version__}")
    print(f"  tokenizers: {__import__('tokenizers').__version__}")
    print(f"  torch: {torch.__version__}")
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    if hasattr(tokenizer, 'legacy_behaviour'):
        print(f"  tokenizer.legacy_behaviour: {tokenizer.legacy_behaviour}")
    else:
        print(f"  tokenizer.legacy_behaviour: (attribute not present)")
    
    # Language codes
    deu_code = 'deu_Latn'
    hin_code = 'hin_Deva'
    eng_code = 'eng_Latn'
    
    # Test sentences
    de_samples = [
        "Was ist eine unechte Empore?",
        "Was ist der Inhalt des Buches \"Bänkelsang – Texte – Bilder – Kommentare\" von Wolfgang Braungart?",
        "Was macht den Pas de deux von Odette und Siegfried im 2. Akt von Schwanensee so bedeutend?",
        "Der öffentliche Nahverkehr ist in Hongkong stark ausgebaut.",
        "Die am 1. Oktober 1979 eröffnete Hongkonger U-Bahn heißt Mass Transit Railway (MTR).",
        "Nach den Ausschreitungen und Plünderungen in Stuttgart 2020 wurde die Zusammenarbeit weiterentwickelt."
    ]
    
    hi_samples = [
        "मालकानगिरि में कौन-कौन सी नदियों का संगम होता है?",
        "अश्वगंधा की खेती किस प्रकार की दशाओं में की जा सकती है?"
    ]
    
    all_test_sentences = de_samples + hi_samples
    
    print("\n" + "="*120)
    print("TOKENIZATION TEST:")
    
    for lang_name, lang_code, sample_text in [('de', deu_code, de_samples[0]), ('hi', hin_code, hi_samples[0])]:
        print(f"\n{lang_name.upper()} sample: {sample_text}")
        
        # Tokenize with correct src_lang
        tokenizer.src_lang = lang_code
        tokens_correct = tokenizer(sample_text, return_tensors="pt")
        input_ids_correct = tokens_correct['input_ids'][0].tolist()
        token_strings_correct = tokenizer.convert_ids_to_tokens(input_ids_correct)
        
        # Tokenize with eng_Latn
        tokenizer.src_lang = eng_code
        tokens_eng = tokenizer(sample_text, return_tensors="pt")
        input_ids_eng = tokens_eng['input_ids'][0].tolist()
        token_strings_eng = tokenizer.convert_ids_to_tokens(input_ids_eng)
        
        # Compare
        identical = (input_ids_correct == input_ids_eng)
        
        print(f"  With src_lang={lang_code}:")
        print(f"    input_ids: {input_ids_correct}")
        print(f"    tokens: {token_strings_correct}")
        print(f"  With src_lang={eng_code}:")
        print(f"    input_ids: {input_ids_eng}")
        print(f"    tokens: {token_strings_eng}")
        print(f"  Result: {'IDENTICAL' if identical else 'DIFFERENT'}")
    
    # Print language token IDs
    print(f"\nLanguage token IDs:")
    print(f"  {deu_code}: {tokenizer.convert_tokens_to_ids(deu_code)}")
    print(f"  {hin_code}: {tokenizer.convert_tokens_to_ids(hin_code)}")
    print(f"  {eng_code}: {tokenizer.convert_tokens_to_ids(eng_code)}")
    
    print("\n" + "="*120)
    print("TRANSLATION TEST (greedy):")
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(eng_code)
    
    for i, text in enumerate(all_test_sentences):
        # Determine source language
        if i < len(de_samples):
            src_lang_name = 'de'
            src_lang_code = deu_code
        else:
            src_lang_name = 'hi'
            src_lang_code = hin_code
        
        print(f"\n{src_lang_name.upper()} [{i+1}/{len(all_test_sentences)}]: {text}")
        
        # Translate with correct src_lang
        tokenizer.src_lang = src_lang_code
        inputs_correct = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            generated_tokens_correct = model.generate(
                **inputs_correct,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )
        
        translation_correct = tokenizer.batch_decode(generated_tokens_correct, skip_special_tokens=True)[0]
        
        # Translate with eng_Latn
        tokenizer.src_lang = eng_code
        inputs_eng = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            generated_tokens_eng = model.generate(
                **inputs_eng,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )
        
        translation_eng = tokenizer.batch_decode(generated_tokens_eng, skip_special_tokens=True)[0]
        
        # Compare
        identical = (translation_correct == translation_eng)
        
        print(f"  With src_lang={src_lang_code}: {translation_correct}")
        print(f"  With src_lang={eng_code}: {translation_eng}")
        print(f"  Result: {'IDENTICAL' if identical else 'DIFFERENT'}")
    
    print("\n" + "="*120)
    print("TRANSLATION TEST (beam search, num_beams=4):")
    
    for i, text in enumerate(all_test_sentences):
        # Determine source language
        if i < len(de_samples):
            src_lang_name = 'de'
            src_lang_code = deu_code
        else:
            src_lang_name = 'hi'
            src_lang_code = hin_code
        
        print(f"\n{src_lang_name.upper()} [{i+1}/{len(all_test_sentences)}]: {text}")
        
        # Translate with correct src_lang and beam search
        tokenizer.src_lang = src_lang_code
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4
            )
        
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"  Translation: {translation}")
    
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("Tokenization and translation tests completed.")
    print("See detailed output above for tokenization differences and translation comparisons.")
    print("="*120)


if __name__ == "__main__":
    with torch.no_grad():
        main()
