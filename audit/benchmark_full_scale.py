"""
Full-scale benchmarking script for translation and embedding operations.
This is a read-only diagnostic - no existing files are modified.
"""

import yaml
import torch
import os
import time
import shutil
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from utils import translate_to_english
from translation_cache import TranslationCache

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def print_hardware_info():
    """Print hardware information"""
    print("="*80)
    print("HARDWARE CHECK")
    print("="*80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\ntorch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"torch.cuda.get_device_name(0): {device_name}")
    else:
        print("No CUDA device available")
    
    # Check CPU cores
    cpu_count = os.cpu_count()
    print(f"os.cpu_count(): {cpu_count}")
    print()

def benchmark_translation(config):
    """Benchmark translation speed"""
    print("="*80)
    print("TRANSLATION SPEED BENCHMARK")
    print("="*80)
    print("\nLoading corpus documents for translation benchmark...")
    
    corpus_name = config['corpus_dataset']
    
    # Load 20 German documents
    corpus_de = load_dataset(corpus_name, 'de', split='test')
    docs_de = [corpus_de[i]['text'] for i in range(20)]
    print(f"Loaded 20 German documents")
    
    # Load 20 Hindi documents
    corpus_hi = load_dataset(corpus_name, 'hi', split='test')
    docs_hi = [corpus_hi[i]['text'] for i in range(20)]
    print(f"Loaded 20 Hindi documents")
    
    # Create temporary cache directory
    tmp_cache_dir = Path('benchmark_cache_tmp')
    tmp_cache_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Benchmark each combination
    for lang, docs, lang_name in [('de', docs_de, 'German'), ('hi', docs_hi, 'Hindi')]:
        for model_name in ['nllb', 'm2m100']:
            print(f"\n--- Benchmarking {model_name.upper()} for {lang_name} ---")
            
            # Create fresh cache for this benchmark
            cache = TranslationCache(cache_dir=str(tmp_cache_dir))
            
            start_time = time.time()
            
            # Translate all 20 documents
            for i, doc in enumerate(docs):
                translated = translate_to_english(doc, lang, model_name, cache)
                if (i + 1) % 5 == 0:
                    print(f"  Translated {i + 1}/20 documents...")
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_doc = total_time / 20
            
            results[(lang, model_name)] = time_per_doc
            
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Time per document: {time_per_doc:.3f} seconds")
    
    # Clean up temporary cache
    print(f"\nCleaning up temporary cache directory...")
    shutil.rmtree(tmp_cache_dir)
    print(f"Deleted {tmp_cache_dir}")
    
    # Project full corpus translation time
    print("\n" + "="*80)
    print("FULL CORPUS TRANSLATION TIME PROJECTIONS")
    print("="*80)
    
    full_corpus_size = 13500
    total_translation_hours = 0
    
    for lang, lang_name in [('de', 'German'), ('hi', 'Hindi')]:
        print(f"\n{lang_name} corpus (13500 documents):")
        for model_name in ['nllb', 'm2m100']:
            time_per_doc = results[(lang, model_name)]
            total_seconds = time_per_doc * full_corpus_size
            total_hours = total_seconds / 3600
            total_translation_hours += total_hours
            
            print(f"  {model_name.upper()}: {total_hours:.2f} hours")
    
    print(f"\nTotal translation time (all 4 jobs): {total_translation_hours:.2f} hours")
    
    return total_translation_hours

def benchmark_embeddings(config):
    """Benchmark embedding generation speed"""
    print("\n" + "="*80)
    print("EMBEDDING SPEED BENCHMARK")
    print("="*80)
    print("\nLoading corpus documents for embedding benchmark...")
    
    corpus_name = config['corpus_dataset']
    
    # Load 500 English documents
    corpus_en = load_dataset(corpus_name, 'en', split='test')
    docs_en = [corpus_en[i]['text'] for i in range(500)]
    print(f"Loaded 500 English documents")
    
    # Benchmark smallest and largest models
    models_to_test = [
        ('all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2', 'smallest'),
        ('bge-m3', 'BAAI/bge-m3', 'largest')
    ]
    
    results = {}
    
    for model_name, model_path, size_label in models_to_test:
        print(f"\n--- Benchmarking {model_name} ({size_label}) ---")
        print(f"Loading model from {model_path}...")
        
        model = SentenceTransformer(model_path)
        
        print(f"Encoding 500 documents with batch_size=32...")
        start_time = time.time()
        
        embeddings = model.encode(docs_en, batch_size=32, show_progress_bar=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_doc = total_time / 500
        
        results[model_name] = time_per_doc
        
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per document: {time_per_doc:.4f} seconds")
        
        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Project full embedding workload
    print("\n" + "="*80)
    print("FULL EMBEDDING TIME PROJECTIONS")
    print("="*80)
    
    full_corpus_size = 13500
    num_models = 6  # From config: 6 models total
    
    # Language-corpus combinations:
    # 3 native languages (en, de, hi) + 2 translated (de_translated_en, hi_translated_en) = 5 per model
    num_lang_corpus_combos = 5
    
    print(f"\nConfiguration:")
    print(f"  Full corpus size: {full_corpus_size} documents")
    print(f"  Total models: {num_models}")
    print(f"  Language-corpus combinations per model: {num_lang_corpus_combos}")
    print(f"    (3 native: en, de, hi + 2 translated: de->en, hi->en)")
    
    # Use linear interpolation for middle-sized models
    time_per_doc_smallest = results['all-MiniLM-L6-v2']
    time_per_doc_largest = results['bge-m3']
    
    # Assign speeds to all 6 models (smallest to largest)
    # all-MiniLM-L6-v2, all-mpnet-base-v2, stella-base-en-v2,
    # paraphrase-multilingual-MiniLM-L12-v2, multilingual-e5-base, bge-m3
    model_speeds = {
        'all-MiniLM-L6-v2': time_per_doc_smallest,
        'all-mpnet-base-v2': time_per_doc_smallest * 1.5,  # Estimate
        'stella-base-en-v2': time_per_doc_smallest * 1.5,  # Estimate
        'paraphrase-multilingual-MiniLM-L12-v2': time_per_doc_smallest * 2.0,  # Estimate
        'multilingual-e5-base': time_per_doc_largest * 0.7,  # Estimate
        'bge-m3': time_per_doc_largest
    }
    
    print("\n" + "-"*80)
    print("Per-model projections:")
    print("-"*80)
    
    total_embedding_hours = 0
    
    for model_name, time_per_doc in model_speeds.items():
        # Total docs for this model: full_corpus_size * num_lang_corpus_combos
        total_docs = full_corpus_size * num_lang_corpus_combos
        total_seconds = time_per_doc * total_docs
        total_hours = total_seconds / 3600
        total_embedding_hours += total_hours
        
        print(f"\n{model_name}:")
        print(f"  Time per doc: {time_per_doc:.4f} seconds")
        print(f"  Total docs to embed: {total_docs:,} ({num_lang_corpus_combos} corpora)")
        print(f"  Estimated time: {total_hours:.2f} hours")
    
    print("\n" + "-"*80)
    print(f"TOTAL EMBEDDING TIME (all 6 models, 5 corpora each): {total_embedding_hours:.2f} hours")
    print("-"*80)
    
    return total_embedding_hours

def main():
    print("="*80)
    print("FULL-SCALE BENCHMARKING: TRANSLATION + EMBEDDING")
    print("="*80)
    print("\nThis is a read-only diagnostic benchmark.")
    print("No existing files will be modified.\n")
    
    # Load configuration
    config = load_config()
    
    # 1. Hardware check
    print_hardware_info()
    
    # 2. Translation speed benchmark
    translation_hours = benchmark_translation(config)
    
    # 3. Embedding speed benchmark
    embedding_hours = benchmark_embeddings(config)
    
    # 4. Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY: PROJECTED FULL-SCALE PROCESSING TIME")
    print("="*80)
    
    print(f"\nTranslation (4 full corpus jobs):")
    print(f"  German corpus → English (NLLB): included above")
    print(f"  German corpus → English (M2M100): included above")
    print(f"  Hindi corpus → English (NLLB): included above")
    print(f"  Hindi corpus → English (M2M100): included above")
    print(f"  TOTAL TRANSLATION TIME: {translation_hours:.2f} hours")
    
    print(f"\nEmbedding generation (6 models × 5 corpora):")
    print(f"  3 native language corpora: en, de, hi")
    print(f"  2 translated corpora: de→en, hi→en")
    print(f"  6 embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2, stella-base-en-v2,")
    print(f"                      paraphrase-multilingual-MiniLM-L12-v2, multilingual-e5-base, bge-m3")
    print(f"  TOTAL EMBEDDING TIME: {embedding_hours:.2f} hours")
    
    grand_total_hours = translation_hours + embedding_hours
    grand_total_days = grand_total_hours / 24
    
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL PROCESSING TIME: {grand_total_hours:.2f} hours ({grand_total_days:.2f} days)")
    print(f"{'='*80}")
    
    print("\nNote: These are sequential time estimates. Actual time may vary based on:")
    print("  - CPU/GPU utilization and thermal throttling")
    print("  - System load and background processes")
    print("  - Network bandwidth for initial model/dataset downloads")
    print("  - Disk I/O for reading/writing embeddings and translations")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
