"""
GPU benchmarking script for translation and embedding operations.
Each section is isolated with explicit memory cleanup.
This is a read-only diagnostic - no existing files are modified.
"""

import yaml
import torch
import gc
import time
import traceback
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100Tokenizer, M2M100ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Language code mappings
NLLB_LANG_CODES = {
    'de': 'deu_Latn', 'en': 'eng_Latn', 'hi': 'hin_Deva'
}

M2M100_LANG_CODES = {
    'de': 'de', 'en': 'en', 'hi': 'hi'
}

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def cleanup_gpu():
    """Aggressively clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_vram_usage(label=""):
    """Print current VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
        print(f"  VRAM {label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

def section_1_gpu_check():
    """Check GPU availability and specifications"""
    print("="*80)
    print("SECTION 1: GPU CHECK")
    print("="*80)
    
    cuda_available = torch.cuda.is_available()
    print(f"\ntorch.cuda.is_available(): {cuda_available}")
    
    if not cuda_available:
        print("\nERROR: No CUDA device available. This script requires GPU.")
        print("Exiting...")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    print(f"torch.cuda.get_device_name(0): {device_name}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024**3)
    print(f"Total VRAM: {total_memory_gb:.2f} GB ({total_memory:,} bytes)")
    
    print()
    return True

def section_2_translation_benchmark(config):
    """Benchmark batched GPU translation with fp16"""
    print("="*80)
    print("SECTION 2: BATCHED TRANSLATION BENCHMARK (GPU, FP16)")
    print("="*80)
    
    torch.cuda.reset_peak_memory_stats()
    
    corpus_name = config['corpus_dataset']
    
    # Load documents
    print("\nLoading corpus documents...")
    corpus_de = load_dataset(corpus_name, 'de', split='test')
    docs_de = [corpus_de[i]['text'] for i in range(100)]
    print(f"Loaded 100 German documents")
    
    corpus_hi = load_dataset(corpus_name, 'hi', split='test')
    docs_hi = [corpus_hi[i]['text'] for i in range(100)]
    print(f"Loaded 100 Hindi documents")
    
    results = {}
    batch_size = 8
    
    for lang, docs, lang_name in [('de', docs_de, 'German'), ('hi', docs_hi, 'Hindi')]:
        for model_name in ['nllb', 'm2m100']:
            print(f"\n{'='*80}")
            print(f"Benchmarking {model_name.upper()} for {lang_name} (batched, fp16)")
            print(f"{'='*80}")
            
            torch.cuda.reset_peak_memory_stats()
            
            try:
                # Load model
                print(f"Loading {model_name} model to GPU with fp16...")
                if model_name == "nllb":
                    model_id = "facebook/nllb-200-distilled-600M"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
                    src_code = NLLB_LANG_CODES[lang]
                    tgt_code = NLLB_LANG_CODES['en']
                    tokenizer.src_lang = src_code
                    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                elif model_name == "m2m100":
                    model_id = "facebook/m2m100_418M"
                    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
                    model = M2M100ForConditionalGeneration.from_pretrained(model_id, use_safetensors=True)
                    src_code = M2M100_LANG_CODES[lang]
                    tgt_code = M2M100_LANG_CODES['en']
                    tokenizer.src_lang = src_code
                    forced_bos_token_id = tokenizer.get_lang_id(tgt_code)
                
                # Move to GPU and convert to fp16
                model = model.half().cuda()
                model.eval()
                
                print(f"Model loaded on GPU")
                print_vram_usage("after model load")
                
                # Batch translation
                print(f"\nTranslating 100 documents in batches of {batch_size}...")
                translations = []
                
                start_time = time.time()
                
                for i in range(0, len(docs), batch_size):
                    batch_texts = docs[i:i+batch_size]
                    
                    # Tokenize batch
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to('cuda')
                    
                    # Generate translations
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=512
                        )
                    
                    # Decode batch
                    batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    translations.extend(batch_translations)
                    
                    if (i + batch_size) % 40 == 0:
                        print(f"  Translated {min(i + batch_size, len(docs))}/100 documents...")
                
                end_time = time.time()
                total_time = end_time - start_time
                time_per_doc = total_time / len(docs)
                
                print(f"\nResults:")
                print(f"  Total time: {total_time:.2f} seconds")
                print(f"  Time per document: {time_per_doc:.4f} seconds")
                print_vram_usage("peak during translation")
                
                # Store results
                results[(lang, model_name)] = time_per_doc
                
                # Print example translations for quality check
                print(f"\nExample translations (quality check):")
                for idx in [0, 1]:
                    print(f"\n  Example {idx + 1}:")
                    print(f"    Input ({lang}):  {docs[idx][:150]}...")
                    print(f"    Output (en): {translations[idx][:150]}...")
                
                # Cleanup
                print(f"\nCleaning up {model_name} model...")
                del model
                del tokenizer
                del inputs
                del generated_tokens
                cleanup_gpu()
                print(f"Cleanup complete")
                
            except Exception as e:
                print(f"\nERROR during {model_name} {lang_name} benchmark:")
                print(traceback.format_exc())
                results[(lang, model_name)] = None
    
    # Projections
    print("\n" + "="*80)
    print("FULL CORPUS TRANSLATION PROJECTIONS")
    print("="*80)
    
    full_corpus_size = 13500
    full_query_size = 1500
    
    total_corpus_hours = 0
    total_query_minutes = 0
    
    print(f"\nFull corpus size: {full_corpus_size} documents per language")
    print(f"Full query size: {full_query_size} queries per language")
    
    for lang, lang_name in [('de', 'German'), ('hi', 'Hindi')]:
        print(f"\n{lang_name}:")
        for model_name in ['nllb', 'm2m100']:
            time_per_doc = results.get((lang, model_name))
            if time_per_doc is not None:
                # Corpus projection
                corpus_total_seconds = time_per_doc * full_corpus_size
                corpus_hours = corpus_total_seconds / 3600
                total_corpus_hours += corpus_hours
                
                # Query projection
                query_total_seconds = time_per_doc * full_query_size
                query_minutes = query_total_seconds / 60
                total_query_minutes += query_minutes
                
                print(f"  {model_name.upper()}:")
                print(f"    Corpus (13500 docs): {corpus_hours:.2f} hours")
                print(f"    Queries (1500 queries): {query_minutes:.2f} minutes")
            else:
                print(f"  {model_name.upper()}: FAILED")
    
    print(f"\n{'='*80}")
    print(f"Total translation time:")
    print(f"  Corpus (4 jobs): {total_corpus_hours:.2f} hours")
    print(f"  Queries (4 jobs): {total_query_minutes:.2f} minutes")
    print(f"{'='*80}")
    
    # Final cleanup
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    return total_corpus_hours, total_query_minutes

def section_3_embedding_benchmark(config):
    """Benchmark GPU embedding generation"""
    print("\n" + "="*80)
    print("SECTION 3: EMBEDDING BENCHMARK (GPU, ISOLATED)")
    print("="*80)
    
    torch.cuda.reset_peak_memory_stats()
    
    corpus_name = config['corpus_dataset']
    
    # Load English documents
    print("\nLoading 500 English corpus documents...")
    corpus_en = load_dataset(corpus_name, 'en', split='test')
    docs_en = [corpus_en[i]['text'] for i in range(500)]
    print(f"Loaded 500 English documents")
    
    results = {}
    batch_size = 32
    
    # Benchmark small model
    print(f"\n{'='*80}")
    print("Benchmarking all-MiniLM-L6-v2 (small model)")
    print(f"{'='*80}")
    
    torch.cuda.reset_peak_memory_stats()
    
    try:
        print("Loading all-MiniLM-L6-v2 to GPU...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model = model.to('cuda')
        
        print("Model loaded on GPU")
        print_vram_usage("after model load")
        
        print(f"\nEncoding 500 documents with batch_size={batch_size}...")
        start_time = time.time()
        
        embeddings = model.encode(
            docs_en,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
            device='cuda'
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_doc = total_time / 500
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per document: {time_per_doc:.4f} seconds")
        print_vram_usage("peak during encoding")
        
        results['all-MiniLM-L6-v2'] = time_per_doc
        
        # Cleanup
        print("\nCleaning up all-MiniLM-L6-v2...")
        del model
        del embeddings
        cleanup_gpu()
        torch.cuda.reset_peak_memory_stats()
        print("Cleanup complete")
        
    except Exception as e:
        print(f"\nERROR during all-MiniLM-L6-v2 benchmark:")
        print(traceback.format_exc())
        results['all-MiniLM-L6-v2'] = None
    
    # Benchmark large model
    print(f"\n{'='*80}")
    print("Benchmarking bge-m3 (large model)")
    print(f"{'='*80}")
    
    torch.cuda.reset_peak_memory_stats()
    
    try:
        print("Loading bge-m3 to GPU...")
        model = SentenceTransformer('BAAI/bge-m3')
        model = model.to('cuda')
        
        print("Model loaded on GPU")
        print_vram_usage("after model load")
        
        print(f"\nEncoding 500 documents with batch_size={batch_size}...")
        start_time = time.time()
        
        embeddings = model.encode(
            docs_en,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
            device='cuda'
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_doc = total_time / 500
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per document: {time_per_doc:.4f} seconds")
        print_vram_usage("peak during encoding")
        
        results['bge-m3'] = time_per_doc
        
        # Cleanup
        print("\nCleaning up bge-m3...")
        del model
        del embeddings
        cleanup_gpu()
        torch.cuda.reset_peak_memory_stats()
        print("Cleanup complete")
        
    except Exception as e:
        print(f"\nERROR during bge-m3 benchmark:")
        print(traceback.format_exc())
        results['bge-m3'] = None
    
    # Projections
    print("\n" + "="*80)
    print("FULL EMBEDDING TIME PROJECTIONS")
    print("="*80)
    
    full_corpus_size = 13500
    num_models = 6
    num_lang_corpus_combos = 5  # 3 native + 2 translated
    
    print(f"\nConfiguration:")
    print(f"  Full corpus size: {full_corpus_size} documents")
    print(f"  Total models: {num_models}")
    print(f"  Language-corpus combinations per model: {num_lang_corpus_combos}")
    print(f"    (3 native: en, de, hi + 2 translated: de->en, hi->en)")
    
    # Model parameter counts (approximate)
    model_params = {
        'all-MiniLM-L6-v2': 22,  # Million parameters
        'all-mpnet-base-v2': 109,
        'stella-base-en-v2': 109,
        'paraphrase-multilingual-MiniLM-L12-v2': 118,
        'multilingual-e5-base': 278,
        'bge-m3': 568
    }
    
    time_smallest = results.get('all-MiniLM-L6-v2')
    time_largest = results.get('bge-m3')
    
    if time_smallest is None or time_largest is None:
        print("\nERROR: Could not complete benchmarks for interpolation")
        return 0.0
    
    # Linear interpolation by parameter count
    param_smallest = model_params['all-MiniLM-L6-v2']
    param_largest = model_params['bge-m3']
    
    model_speeds = {}
    
    for model_name, params in model_params.items():
        if model_name == 'all-MiniLM-L6-v2':
            model_speeds[model_name] = time_smallest
        elif model_name == 'bge-m3':
            model_speeds[model_name] = time_largest
        else:
            # Linear interpolation
            ratio = (params - param_smallest) / (param_largest - param_smallest)
            model_speeds[model_name] = time_smallest + ratio * (time_largest - time_smallest)
    
    print("\n" + "-"*80)
    print("Per-model projections (with parameter counts):")
    print("-"*80)
    
    total_embedding_hours = 0
    
    for model_name in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'stella-base-en-v2',
                       'paraphrase-multilingual-MiniLM-L12-v2', 'multilingual-e5-base', 'bge-m3']:
        time_per_doc = model_speeds[model_name]
        params = model_params[model_name]
        
        # Mark if benchmarked or interpolated
        if model_name in ['all-MiniLM-L6-v2', 'bge-m3']:
            method = "BENCHMARKED"
        else:
            method = "INTERPOLATED"
        
        total_docs = full_corpus_size * num_lang_corpus_combos
        total_seconds = time_per_doc * total_docs
        total_hours = total_seconds / 3600
        total_embedding_hours += total_hours
        
        print(f"\n{model_name} ({method}):")
        print(f"  Parameters: ~{params}M")
        print(f"  Time per doc: {time_per_doc:.4f} seconds")
        print(f"  Total docs to embed: {total_docs:,} ({num_lang_corpus_combos} corpora)")
        print(f"  Estimated time: {total_hours:.2f} hours")
    
    print("\n" + "-"*80)
    print(f"TOTAL EMBEDDING TIME (all 6 models, 5 corpora each): {total_embedding_hours:.2f} hours")
    print("-"*80)
    
    # Note on interpolation
    print("\nNote: Times for all-mpnet-base-v2, stella-base-en-v2,")
    print("      paraphrase-multilingual-MiniLM-L12-v2, and multilingual-e5-base")
    print("      are linearly interpolated based on parameter count between")
    print(f"      the benchmarked models ({param_smallest}M and {param_largest}M params).")
    
    # Final cleanup
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    return total_embedding_hours

def main():
    print("="*80)
    print("GPU BENCHMARKING: BATCHED TRANSLATION + EMBEDDING")
    print("="*80)
    print("\nThis is a read-only diagnostic benchmark with isolated sections.")
    print("No existing files will be modified.\n")
    
    # Load configuration
    config = load_config()
    
    # Section 1: GPU check
    if not section_1_gpu_check():
        return
    
    # Section 2: Translation benchmark
    try:
        translation_corpus_hours, translation_query_minutes = section_2_translation_benchmark(config)
    except Exception as e:
        print(f"\nFATAL ERROR in translation benchmark:")
        print(traceback.format_exc())
        translation_corpus_hours = 0.0
        translation_query_minutes = 0.0
    
    # Section 3: Embedding benchmark
    try:
        embedding_hours = section_3_embedding_benchmark(config)
    except Exception as e:
        print(f"\nFATAL ERROR in embedding benchmark:")
        print(traceback.format_exc())
        embedding_hours = 0.0
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY: PROJECTED FULL-SCALE GPU PROCESSING TIME")
    print("="*80)
    
    translation_query_hours = translation_query_minutes / 60
    total_translation_hours = translation_corpus_hours + translation_query_hours
    
    print(f"\nTranslation (GPU, batched, fp16):")
    print(f"  Corpus translation (4 jobs, 13500 docs each): {translation_corpus_hours:.2f} hours")
    print(f"  Query translation (4 jobs, 1500 queries each): {translation_query_minutes:.2f} minutes ({translation_query_hours:.2f} hours)")
    print(f"  TOTAL TRANSLATION TIME: {total_translation_hours:.2f} hours")
    
    print(f"\nEmbedding generation (GPU):")
    print(f"  6 models × 5 corpora (13500 docs each)")
    print(f"  TOTAL EMBEDDING TIME: {embedding_hours:.2f} hours")
    
    grand_total_hours = total_translation_hours + embedding_hours
    grand_total_days = grand_total_hours / 24
    
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL GPU PROCESSING TIME: {grand_total_hours:.2f} hours ({grand_total_days:.2f} days)")
    print(f"{'='*80}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
