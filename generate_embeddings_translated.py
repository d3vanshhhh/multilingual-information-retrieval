"""
Generate embeddings for NLLB-translated corpus with resumability and logging.
"""

import argparse
import yaml
import json
import numpy as np
import faiss
import logging
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime


def setup_logging(model_name, language):
    """Set up logging to both console and file"""
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"generate_embeddings_translated_{model_name}_{language}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(f'embeddings_translated_{model_name}_{language}')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_translated_corpus(corpus_file):
    """Load translated corpus from JSON file"""
    with open(corpus_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_embeddings_for_translated_corpus(model_name, model_id, language, embeddings_dir, translation_model='nllb'):
    """
    Generate embeddings for translated corpus with resumability.
    
    Args:
        model_name: Name of the embedding model
        model_id: HuggingFace model ID
        language: Original language code (de, hi)
        embeddings_dir: Directory to save embeddings
        translation_model: Translation model used (default: nllb)
    """
    # Set up logging
    logger, log_file = setup_logging(model_name, language)
    
    start_time = datetime.now()
    
    logger.info(f"{'='*80}")
    logger.info(f"EMBEDDING GENERATION (TRANSLATED CORPUS)")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Language: {language} → English (translated via {translation_model})")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}")
    
    try:
        # Load translated corpus
        corpus_file = Path('translated_corpus') / f"corpus_{language}_translated_en_{translation_model}.json"
        
        if not corpus_file.exists():
            logger.error(f"Translated corpus not found: {corpus_file}")
            return False
        
        logger.info(f"Loading translated corpus: {corpus_file.name}")
        translated_corpus = load_translated_corpus(corpus_file)
        doc_count = len(translated_corpus)
        logger.info(f"Loaded {doc_count} translated documents")
        
        # Output index path
        embeddings_dir = Path(embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        index_path = embeddings_dir / f"{model_name}_{language}_translated_en_{translation_model}.index"
        
        # Check if index already exists and is complete
        if index_path.exists():
            try:
                existing_index = faiss.read_index(str(index_path))
                if existing_index.ntotal == doc_count:
                    logger.info(f"Index already exists with correct count: {index_path}")
                    logger.info(f"  ntotal={existing_index.ntotal}, expected={doc_count}")
                    logger.info(f"Skipping - already complete")
                    end_time = datetime.now()
                    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"Duration: {end_time - start_time}")
                    return True
                else:
                    logger.warning(f"Existing index has wrong count: ntotal={existing_index.ntotal}, expected={doc_count}")
                    logger.info(f"Regenerating index...")
            except Exception as e:
                logger.warning(f"Could not read existing index: {e}")
                logger.info(f"Regenerating index...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_id}")
        model = SentenceTransformer(model_id)
        
        # Explicit device selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        logger.info(f"Model device: {device}")
        
        # Extract texts
        texts = [doc['text'] for doc in translated_corpus]
        
        # Apply multilingual-e5-base prefix if needed
        if model_name == 'multilingual-e5-base':
            texts = [f"passage: {t}" for t in texts]
            logger.info(f"Applying 'passage: ' prefix for multilingual-e5-base")
        
        logger.info(f"Encoding {len(texts)} translated documents...")
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved index: {index_path}")
        logger.info(f"  Vectors: {index.ntotal}")
        logger.info(f"  Dimensions: {dimension}")
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED SUCCESSFULLY")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed time: {elapsed}")
        logger.info(f"{'='*80}")
        
        return True
        
    except Exception as e:
        logger.error(f"\nERROR during embedding generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for NLLB-translated corpus")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--model', help='Generate embeddings for specific model only')
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
    
    print(f"{'='*80}")
    print(f"TRANSLATED CORPUS EMBEDDING GENERATION")
    print(f"{'='*80}")
    print(f"Processing {len(models_to_process)} model(s) for languages: de, hi")
    print(f"Translation model: nllb")
    print(f"{'='*80}\n")
    
    # Languages with translated corpus (skip English)
    languages = ['de', 'hi']
    
    total_success = 0
    total_jobs = len(models_to_process) * len(languages)
    
    # Process each model
    for model_name, model_id in models_to_process.items():
        # Process each language
        for lang in languages:
            success = generate_embeddings_for_translated_corpus(
                model_name,
                model_id,
                lang,
                args.embeddings_dir,
                translation_model='nllb'
            )
            
            if success:
                total_success += 1
            
            print()  # Blank line between jobs
    
    # Final summary
    print(f"{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {total_success}/{total_jobs}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
