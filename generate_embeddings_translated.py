"""
Generate embeddings for translated corpus.
This creates embeddings for the English-translated versions of German and Hindi documents.
"""

import argparse
import yaml
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_translated_corpus(corpus_file):
    """Load translated corpus from JSON file."""
    with open(corpus_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_embeddings_for_translated_corpus(model_name, model_id, translated_corpus, embeddings_dir, original_language):
    """Generate embeddings for translated corpus."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Original language: {original_language} → English (translated)")
    print(f"Documents: {len(translated_corpus)}")
    print(f"{'='*60}")
    
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file name: model_originalLang_translated_en.index
    index_path = embeddings_dir / f"{model_name}_{original_language}_translated_en.index"
    
    if index_path.exists():
        print(f"  Index already exists: {index_path}")
        return True
    
    try:
        # Load model
        print(f"  Loading model: {model_id}")
        
        # Special handling for models that need trust_remote_code=True
        if 'gte-multilingual-base' in model_id:
            model = SentenceTransformer(model_id, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_id)
        
        # Extract texts
        texts = [doc['text'] for doc in translated_corpus]
        print(f"  Encoding {len(texts)} translated documents...")
        
        # Generate embeddings with special handling for gte-multilingual-base
        if 'gte-multilingual-base' in model_id:
            # Encode in smaller batches for gte-multilingual-base
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                batch_size=16,  # Smaller batch size
                convert_to_numpy=True,
                normalize_embeddings=True  # Use built-in normalization
            )
        else:
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
        
        # Normalize embeddings for cosine similarity (skip if already normalized)
        if 'gte-multilingual-base' not in model_id:
            embeddings = embeddings.astype(np.float32)  # Ensure float32 type
            faiss.normalize_L2(embeddings)
        else:
            embeddings = embeddings.astype(np.float32)  # Ensure float32 type
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, str(index_path))
        print(f"  Saved index with {index.ntotal} vectors to {index_path}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for translated corpus")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--translated-corpus-dir', default='translated_corpus', help='Directory with translated corpus')
    parser.add_argument('--model', help='Generate embeddings for specific model only')
    parser.add_argument('--test', action='store_true', help='Test mode (for consistency)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine which models to process
    if args.model:
        if args.model not in config['models']:
            print(f"ERROR: Model '{args.model}' not found in config")
            return
        models_to_process = {args.model: config['models'][args.model]}
    else:
        models_to_process = config['models']
    
    # Find translated corpus files
    translated_corpus_dir = Path(args.translated_corpus_dir)
    corpus_files = list(translated_corpus_dir.glob("corpus_*_translated_en.json"))
    
    if not corpus_files:
        print(f"ERROR: No translated corpus files found in {translated_corpus_dir}")
        print("Expected files like: corpus_de_translated_en.json, corpus_hi_translated_en.json")
        return
    
    print(f"Found {len(corpus_files)} translated corpus files:")
    for f in corpus_files:
        print(f"  - {f.name}")
    
    print(f"\nGenerating embeddings for {len(models_to_process)} model(s)")
    
    total_indices = 0
    
    # Process each model
    for model_name, model_id in models_to_process.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Process each translated corpus
        for corpus_file in corpus_files:
            # Extract original language from filename: corpus_de_translated_en.json -> de
            original_lang = corpus_file.stem.split('_')[1]  # corpus_de_translated_en -> de
            
            print(f"\nLoading translated corpus: {corpus_file.name}")
            translated_corpus = load_translated_corpus(corpus_file)
            
            success = generate_embeddings_for_translated_corpus(
                model_name,
                model_id,
                translated_corpus,
                args.embeddings_dir,
                original_lang
            )
            
            if success:
                total_indices += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total indices generated: {total_indices}")
    
    # Calculate disk usage
    embeddings_path = Path(args.embeddings_dir)
    if embeddings_path.exists():
        translated_indices = list(embeddings_path.glob("*_translated_en.index"))
        if translated_indices:
            total_size = sum(f.stat().st_size for f in translated_indices)
            print(f"Translated corpus embeddings: {total_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()