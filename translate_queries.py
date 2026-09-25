"""
Translate queries to English with batched GPU translation, resumability, and logging.
"""
import argparse
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from translation_cache import TranslationCache
from utils import translate_batch_to_english


def setup_logging(language, translation_model):
    """Set up logging to both console and file"""
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"translate_queries_{language}_{translation_model}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('translate_queries')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_existing_translations(output_file):
    """Load existing partial/complete translations from checkpoint file"""
    if not output_file.exists():
        return {}
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_queries = json.load(f)
        
        # Create dict mapping query ID to translated query
        return {query['_id']: query for query in existing_queries}
    except Exception as e:
        logging.warning(f"Could not load existing translations from {output_file}: {e}")
        return {}


def save_translated_queries(translated_queries, output_file):
    """Save translated queries (atomic overwrite for checkpointing)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first, then atomic rename
    temp_file = output_file.with_suffix('.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(translated_queries, f, ensure_ascii=False, indent=2)
    
    temp_file.replace(output_file)


def translate_queries_for_language(language, config, translation_model, cache, logger=None):
    """Translate queries with batched translation, resumability, and checkpointing"""
    if language == 'en':
        logger.info(f"[{language}] Skipping - already English")
        return None
    
    start_time = datetime.now()
    logger.info(f"\n{'='*80}")
    logger.info(f"Translating queries: {language} -> English")
    logger.info(f"Translation model: {translation_model}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    
    # Determine output file path (includes translation model in name)
    output_dir = Path('translated_queries')
    output_file = output_dir / f"queries_{language}_translated_en_{translation_model}.json"
    
    # Load queries
    logger.info(f"[{language}] Loading queries...")
    queries_dataset = load_dataset(
        config['query_dataset'],
        language,
        split='test'
    )
    
    full_queries_size = len(queries_dataset)
    logger.info(f"[{language}] Loaded {full_queries_size} queries")
    
    # Check if already complete
    existing_translations = load_existing_translations(output_file)
    if len(existing_translations) == full_queries_size:
        logger.info(f"[{language}] Output file {output_file} already exists with {len(existing_translations)} queries (matches full query set size)")
        logger.info(f"[{language}] Skipping translation - already complete")
        return None
    
    target_count = full_queries_size
    logger.info(f"[{language}] Target query count: {target_count}")
    logger.info(f"[{language}] Already translated (from checkpoint): {len(existing_translations)}")
    
    # Build list of queries to translate
    queries_to_translate = []
    translated_queries_dict = existing_translations.copy()  # Start with existing translations
    
    for query in queries_dataset:
        if query['_id'] not in translated_queries_dict:
            queries_to_translate.append(query)
    
    if not queries_to_translate:
        logger.info(f"[{language}] All {target_count} queries already translated (from checkpoint)")
        # Still save to ensure file is in correct format
        translated_queries_list = [translated_queries_dict[query['_id']] for query in queries_dataset]
        save_translated_queries(translated_queries_list, output_file)
        logger.info(f"[{language}] Saved {len(translated_queries_list)} queries to {output_file}")
        return translated_queries_list
    
    logger.info(f"[{language}] Queries remaining to translate: {len(queries_to_translate)}")
    
    # Translate in batches
    batch_size = 8
    checkpoint_interval = 500
    
    cache_hits_start = cache.hits
    cache_misses_start = cache.misses
    
    texts_to_translate = [query['text'] for query in queries_to_translate]
    
    logger.info(f"[{language}] Translating {len(texts_to_translate)} queries in batches of {batch_size}...")
    
    # Process in chunks for progress tracking and checkpointing
    translated_count = 0
    
    for chunk_start in tqdm(range(0, len(queries_to_translate), checkpoint_interval), 
                            desc=f"Translating {language}",
                            unit="chunk"):
        chunk_end = min(chunk_start + checkpoint_interval, len(queries_to_translate))
        chunk_queries = queries_to_translate[chunk_start:chunk_end]
        chunk_texts = [query['text'] for query in chunk_queries]
        
        # Translate this chunk (already batched internally)
        chunk_translations = translate_batch_to_english(
            chunk_texts,
            language,
            model_name=translation_model,
            cache=cache,
            batch_size=batch_size
        )
        
        # Add translated queries to dict
        for query, translated_text in zip(chunk_queries, chunk_translations):
            translated_query = {
                '_id': query['_id'],
                'text': translated_text,
                'original_language': language
            }
            translated_queries_dict[query['_id']] = translated_query
            translated_count += 1
        
        # Checkpoint every 500 queries
        if translated_count % checkpoint_interval == 0 or chunk_end == len(queries_to_translate):
            # Save in query set order
            translated_queries_list = [translated_queries_dict[query['_id']] for query in queries_dataset 
                                      if query['_id'] in translated_queries_dict]
            save_translated_queries(translated_queries_list, output_file)
            logger.info(f"[{language}] Checkpoint: saved {len(translated_queries_list)}/{target_count} queries")
            cache.save()  # Also save cache
    
    # Final save (ensure query set order)
    translated_queries_list = [translated_queries_dict[query['_id']] for query in queries_dataset]
    save_translated_queries(translated_queries_list, output_file)
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    cache_hits_delta = cache.hits - cache_hits_start
    cache_misses_delta = cache.misses - cache_misses_start
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[{language}] Translation complete!")
    logger.info(f"[{language}] Saved {len(translated_queries_list)} queries to {output_file}")
    logger.info(f"[{language}] End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[{language}] Elapsed time: {elapsed}")
    logger.info(f"[{language}] Cache hits (this run): {cache_hits_delta}")
    logger.info(f"[{language}] Cache misses (this run): {cache_misses_delta}")
    logger.info(f"[{language}] Total cache size: {len(cache.cache)}")
    logger.info(f"{'='*80}")
    
    return translated_queries_list


def main():
    parser = argparse.ArgumentParser(description="Translate queries to English with batched GPU translation")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--language', required=True, help='Language to translate (required)')
    parser.add_argument('--translation-model', required=True, choices=['nllb'], 
                        help='Translation model to use (required)')
    args = parser.parse_args()
    
    # Set up logging
    logger, log_file = setup_logging(args.language, args.translation_model)
    
    logger.info(f"{'='*80}")
    logger.info(f"QUERY TRANSLATION")
    logger.info(f"{'='*80}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Translation model: {args.translation_model}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}")
    
    config = load_config(args.config)
    cache = TranslationCache()
    
    try:
        translated_queries = translate_queries_for_language(
            args.language,
            config,
            args.translation_model,
            cache,
            logger=logger
        )
        
        logger.info("\nQUERY TRANSLATION JOB COMPLETE")
        logger.info(f"Final cache stats: {cache.hits} hits, {cache.misses} misses, {len(cache.cache)} total entries")
        
    except Exception as e:
        logger.error(f"\nERROR during translation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
