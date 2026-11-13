"""
Check different NLLB tokenizer initialization methods to find correct src_lang handling.

Tests various ways to set the source language and checks token layout.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizer
import sys


def analyze_token_layout(input_ids, tokenizer, lang_id):
    """Analyze where the language ID appears in the token sequence"""
    ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    
    has_lang_first = False
    has_lang_last = False
    has_unk = False
    
    if len(ids_list) > 0:
        if ids_list[0] == lang_id:
            has_lang_first = True
        if ids_list[-1] == lang_id:
            has_lang_last = True
        if 3 in ids_list:  # <unk> token
            has_unk = True
    
    # Determine position
    if has_lang_first:
        position = "FIRST"
    elif has_lang_last:
        position = "LAST"
    elif lang_id in ids_list:
        position = f"MIDDLE (index {ids_list.index(lang_id)})"
    else:
        position = "ABSENT"
    
    unk_note = " (contains <unk> id=3)" if has_unk else ""
    
    return position + unk_note, has_lang_first or has_lang_last or (lang_id in ids_list)


def create_tokenizer_candidates(model_id, lang_code, tokenizer_baseline):
    """Create different tokenizer candidates"""
    candidates = {}
    
    # A. Baseline (as in utils.py)
    try:
        tok_a = AutoTokenizer.from_pretrained(model_id)
        tok_a.src_lang = lang_code
        candidates['A_baseline'] = tok_a
    except Exception as e:
        print(f"  Candidate A failed: {e}")
        candidates['A_baseline'] = None
    
    # B. AutoTokenizer with src_lang parameter
    try:
        tok_b = AutoTokenizer.from_pretrained(model_id, src_lang=lang_code)
        candidates['B_auto_param'] = tok_b
    except Exception as e:
        print(f"  Candidate B failed: {e}")
        candidates['B_auto_param'] = None
    
    # C. NllbTokenizer with src_lang parameter
    try:
        tok_c = NllbTokenizer.from_pretrained(model_id, src_lang=lang_code)
        candidates['C_nllb_param'] = tok_c
    except Exception as e:
        print(f"  Candidate C failed: {e}")
        candidates['C_nllb_param'] = None
    
    # D. AutoTokenizer with src_lang and legacy_behaviour
    try:
        tok_d = AutoTokenizer.from_pretrained(model_id, src_lang=lang_code, legacy_behaviour=True)
        candidates['D_legacy'] = tok_d
    except Exception as e:
        print(f"  Candidate D failed: {e}")
        candidates['D_legacy'] = None
    
    # E and F will be handled specially during tokenization
    candidates['E_manual_prefix'] = tokenizer_baseline
    candidates['F_manual_suffix'] = tokenizer_baseline
    
    return candidates


def tokenize_with_candidate(text, candidate_name, tokenizer, lang_id, eos_id, device):
    """Tokenize text with a specific candidate method"""
    if candidate_name == 'E_manual_prefix':
        # Manual prefix layout: [lang_id] + tokens + [eos_id]
        tokens_no_special = tokenizer(text, add_special_tokens=False)
        input_ids_list = [lang_id] + tokens_no_special['input_ids'] + [eos_id]
        input_ids = torch.tensor([input_ids_list]).to(device)
        attention_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    elif candidate_name == 'F_manual_suffix':
        # Manual suffix layout: tokens + [eos_id, lang_id]
        tokens_no_special = tokenizer(text, add_special_tokens=False)
        input_ids_list = tokens_no_special['input_ids'] + [eos_id, lang_id]
        input_ids = torch.tensor([input_ids_list]).to(device)
        attention_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    else:
        # Regular tokenization
        return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)


def main():
    print("Loading NLLB model...")
    model_id = "facebook/nllb-200-distilled-600M"
    
    tokenizer_baseline = AutoTokenizer.from_pretrained(model_id)
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
    
    # Language codes
    deu_code = 'deu_Latn'
    hin_code = 'hin_Deva'
    eng_code = 'eng_Latn'
    
    # Get token IDs
    deu_id = tokenizer_baseline.convert_tokens_to_ids(deu_code)
    hin_id = tokenizer_baseline.convert_tokens_to_ids(hin_code)
    eng_id = tokenizer_baseline.convert_tokens_to_ids(eng_code)
    eos_id = tokenizer_baseline.eos_token_id
    
    print(f"\nToken IDs:")
    print(f"  {deu_code}: {deu_id}")
    print(f"  {hin_code}: {hin_id}")
    print(f"  {eng_code}: {eng_id}")
    print(f"  EOS: {eos_id}")
    
    # Test sentences
    test_samples = {
        'de': "Was ist eine unechte Empore?",
        'hi': "मालकानगिरि में कौन-कौन सी नदियों का संगम होता है?"
    }
    
    all_test_sentences = [
        ("de", "Was ist eine unechte Empore?"),
        ("de", 'Was ist der Inhalt des Buches "Bänkelsang – Texte – Bilder – Kommentare" von Wolfgang Braungart?'),
        ("de", "Was macht den Pas de deux von Odette und Siegfried im 2. Akt von Schwanensee so bedeutend?"),
        ("de", "Was ist ein Zwischenschein bei Aktien?"),
        ("de", "Wer hat den Roman geschrieben?"),
        ("de", "Wie hoch ist der Mount Everest?"),
        ("de", "Wo liegt die Stadt Chemnitz?"),
        ("de", "Welche Aufgaben hat der Bundesrat?"),
        ("de", "Der öffentliche Nahverkehr ist in Hongkong stark ausgebaut."),
        ("de", "Die am 1. Oktober 1979 eröffnete Hongkonger U-Bahn heißt Mass Transit Railway (MTR)."),
        ("hi", "मालकानगिरि में कौन-कौन सी नदियों का संगम होता है?"),
        ("hi", "अश्वगंधा की खेती किस प्रकार की दशाओं में की जा सकती है?")
    ]
    
    print("\n" + "="*120)
    print("TOKENIZER CANDIDATES AND TOKEN LAYOUT:")
    print("="*120)
    
    # Store results for each language
    lang_results = {}
    
    for lang, lang_code, lang_id in [('de', deu_code, deu_id), ('hi', hin_code, hin_id)]:
        print(f"\n{lang.upper()} ({lang_code}, id={lang_id}):")
        print("-" * 120)
        
        # Create candidates
        candidates = create_tokenizer_candidates(model_id, lang_code, tokenizer_baseline)
        
        # Test with sample sentence
        sample_text = test_samples[lang]
        print(f"\nSample: {sample_text}\n")
        
        valid_candidates = {}
        
        for cand_name, cand_tok in candidates.items():
            if cand_tok is None:
                continue
            
            try:
                if cand_name in ['E_manual_prefix', 'F_manual_suffix']:
                    result = tokenize_with_candidate(sample_text, cand_name, cand_tok, lang_id, eos_id, device)
                    input_ids = result['input_ids'][0]
                else:
                    tokens = cand_tok(sample_text, return_tensors="pt")
                    input_ids = tokens['input_ids'][0]
                
                input_ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
                token_strings = tokenizer_baseline.convert_ids_to_tokens(input_ids_list)
                
                position, has_lang_id = analyze_token_layout(input_ids_list, cand_tok, lang_id)
                
                print(f"{cand_name}:")
                print(f"  input_ids: {input_ids_list}")
                print(f"  tokens: {token_strings}")
                print(f"  Language ID position: {position}")
                
                if has_lang_id:
                    valid_candidates[cand_name] = cand_tok
                
            except Exception as e:
                print(f"{cand_name}: ERROR - {e}")
        
        lang_results[lang] = {
            'lang_code': lang_code,
            'lang_id': lang_id,
            'valid_candidates': valid_candidates
        }
    
    print("\n" + "="*120)
    print("TRANSLATION TESTS (greedy decoding):")
    print("="*120)
    
    forced_bos_token_id = eng_id
    
    # Translate each sentence with all valid candidates
    for lang, text in all_test_sentences:
        lang_info = lang_results[lang]
        lang_code = lang_info['lang_code']
        lang_id = lang_info['lang_id']
        valid_candidates = lang_info['valid_candidates']
        
        print(f"\n{lang.upper()}: {text}")
        print("-" * 120)
        
        for cand_name, cand_tok in valid_candidates.items():
            try:
                # Tokenize
                inputs = tokenize_with_candidate(text, cand_name, cand_tok, lang_id, eos_id, device)
                
                # Translate
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512
                    )
                
                translation = tokenizer_baseline.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                print(f"  {cand_name}: {translation}")
                
            except Exception as e:
                print(f"  {cand_name}: ERROR - {e}")
    
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("Tokenizer candidate tests completed.")
    print("Check the token layout section to see which candidates properly include the language ID.")
    print("Check the translation tests to see which candidates produce correct translations.")
    print("="*120)


if __name__ == "__main__":
    with torch.no_grad():
        main()
