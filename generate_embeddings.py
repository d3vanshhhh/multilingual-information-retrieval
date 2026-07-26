import argparse
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import time
import os


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_embeddings_for_model(model_name, model_id, config, embeddings_dir, force=False, test_mode=False):
    """Generate embeddings for a specific model and all languages."""
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        print(f"[{model_name}] Loading model from {model_id}...")
        
        # Special handling for models that need trust_remote_code=True
        if 'gte-multilingual-base' in model_id:
            model = SentenceTransformer(model_id, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_id)
            
        print(f"[{model_name}] Model loaded successfully")
    except Exception as e:
        print(f"[{model_name}] ERROR: Failed to load model: {e}")
        print(f"[{model_name}] If this is a PyTorch version issue, try: pip install torch>=2.6.0")
        return 0
    
    languages = config['languages']
    max_docs = config['evaluation']['max_docs_per_lang'] if test_mode else None
    indices_generated = 0
    
    for lang in tqdm(languages, desc=f"Languages ({model_name})"):
        index_path = embeddings_dir / f"{model_name}_{lang}.index"
        
        # Skip if already exists and not forcing
        if index_path.exists() and not force:
            print(f"  [{lang}] Index already exists, skipping (use --force to regenerate)")
            continue
        
        try:
            # Load corpus for this language
            print(f"  [{lang}] Loading corpus...")
            corpus_dataset = load_dataset(
                config['corpus_dataset'],
                lang,
                split='test'
            )
            
            # Limit documents in test mode
            if max_docs and len(corpus_dataset) > max_docs:
                corpus_dataset = corpus_dataset.select(range(max_docs))
            
            # Extract texts
            texts = [doc['text'] for doc in corpus_dataset]
            print(f"  [{lang}] Encoding {len(texts)} documents...")
            
            # Generate embeddings with progress bar
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings.astype(np.float32)  # Ensure float32 type
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            # Save index
            faiss.write_index(index, str(index_path))
            print(f"  [{lang}] Saved index with {index.ntotal} vectors to {index_path}")
            indices_generated += 1
            
        except Exception as e:
            print(f"  [{lang}] ERROR: {e}")
            continue
    
    return indices_generated


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for multilingual IR")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--model', help='Generate embeddings for specific model only')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing indices')
    parser.add_argument('--test', action='store_true', help='Test mode: use limited data')
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
    
    print(f"Starting embedding generation for {len(models_to_process)} model(s)")
    print(f"Test mode: {args.test}")
    print(f"Force regeneration: {args.force}")
    
    start_time = time.time()
    total_indices = 0
    
    # Process each model
    for model_name, model_id in models_to_process.items():
        indices = generate_embeddings_for_model(
            model_name,
            model_id,
            config,
            args.embeddings_dir,
            force=args.force,
            test_mode=args.test
        )
        total_indices += indices
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total indices generated: {total_indices}")
    print(f"Total time: {elapsed/60:.2f} minutes")
    
    # Calculate disk usage
    embeddings_path = Path(args.embeddings_dir)
    if embeddings_path.exists():
        total_size = sum(f.stat().st_size for f in embeddings_path.glob('*.index'))
        print(f"Total disk usage: {total_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
