"""
Multilingual IR Evaluation - Complete rewrite with graded metrics, E5 prefix support, and resumability.
"""
import argparse
import yaml
import json
import logging
import torch
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from math import log2


def setup_logging():
    """Set up logging to both console and timestamped file"""
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"evaluate_{timestamp}.log"
    
    logger = logging.getLogger('evaluate')
    logger.setLevel(logging.INFO)
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


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_query_prefix(text, model_name):
    """Apply query prefix for multilingual-e5-base"""
    if model_name == 'multilingual-e5-base':
        return f"query: {text}"
    return text


def load_qrels(config, language):
    """Load graded qrels for a language, preserving scores"""
    qrels_dataset = load_dataset(config['qrels_dataset'], language, split='test')
    
    qrels = defaultdict(dict)
    for item in qrels_dataset:
        query_id = item['query-id']
        corpus_id = item['corpus-id']
        score = float(item['score'])
        qrels[query_id][corpus_id] = score
    
    return dict(qrels)


def compute_metrics(retrieved_doc_ids, qrels_for_query, k_values):
    """
    Compute retrieval metrics for a single query.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs in rank order
        qrels_for_query: Dict {doc_id: score} with graded relevance scores
        k_values: List of k values for metrics computation
    
    Returns:
        Dict with MRR, precision@k, success@k, nDCG@k
    """
    metrics = {}
    
    # MRR: first rank where relevance > 0
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if qrels_for_query.get(doc_id, 0) > 0:
            mrr = 1.0 / rank
            break
    metrics['mrr'] = mrr
    
    # Per-k metrics
    for k in k_values:
        top_k = retrieved_doc_ids[:k]
        
        # Precision@k: fraction of top-k that are relevant (binary)
        relevant_count = sum(1 for doc_id in top_k if qrels_for_query.get(doc_id, 0) > 0)
        metrics[f'precision@{k}'] = relevant_count / k
        
        # Success@k: 1.0 if any relevant doc in top-k, else 0.0
        metrics[f'success@{k}'] = 1.0 if relevant_count > 0 else 0.0
        
        # nDCG@k: graded relevance with DCG/IDCG
        dcg = 0.0
        for rank, doc_id in enumerate(top_k, 1):
            relevance = qrels_for_query.get(doc_id, 0)
            dcg += (2 ** relevance - 1) / log2(rank + 1)
        
        # IDCG: ideal ranking (sort relevance scores descending)
        ideal_scores = sorted(qrels_for_query.values(), reverse=True)[:k]
        idcg = sum((2 ** score - 1) / log2(rank + 1) for rank, score in enumerate(ideal_scores, 1))
        
        metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0
    
    return metrics


def evaluate_direct_pipeline(model_name, model_id, language, config, k_values, test_mode, device, logger):
    """
    Evaluate direct (native) pipeline: native queries -> native corpus.
    
    Returns:
        Dict with evaluation results for this language
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"DIRECT PIPELINE: {model_name} / {language}")
    logger.info(f"{'='*80}")
    
    # Load index
    index_path = Path('embeddings') / f"{model_name}_{language}.index"
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        return None
    
    logger.info(f"Loading index: {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info(f"Index loaded: {index.ntotal} vectors, dim={index.d}")
    
    # Load corpus to get doc_ids in index order
    logger.info(f"Loading corpus dataset...")
    corpus_dataset = load_dataset(config['corpus_dataset'], language, split='test')
    doc_ids = [doc['_id'] for doc in corpus_dataset]
    logger.info(f"Corpus loaded: {len(doc_ids)} documents")
    
    # Load queries
    logger.info(f"Loading queries dataset...")
    queries_dataset = load_dataset(config['query_dataset'], language, split='test')
    logger.info(f"Queries loaded: {len(queries_dataset)} total")
    
    # Load qrels
    logger.info(f"Loading qrels...")
    qrels = load_qrels(config, language)
    logger.info(f"Qrels loaded: {len(qrels)} queries with relevance judgments")
    
    # Filter to queries with qrels and apply test mode limit
    queries_to_eval = []
    for query in queries_dataset:
        if query['_id'] in qrels:
            queries_to_eval.append(query)
    
    if test_mode:
        max_queries = config['evaluation']['max_queries_per_lang']
        queries_to_eval = queries_to_eval[:max_queries]
        logger.info(f"TEST MODE: Limited to {len(queries_to_eval)} queries")
    else:
        logger.info(f"Using all {len(queries_to_eval)} queries with qrels")
    
    # Load model and encode queries
    logger.info(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id)
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Extract and prefix query texts
    query_texts = [query['text'] for query in queries_to_eval]
    query_texts_prefixed = [apply_query_prefix(text, model_name) for text in query_texts]
    
    if model_name == 'multilingual-e5-base':
        logger.info(f"Applied 'query: ' prefix for multilingual-e5-base")
    
    logger.info(f"Encoding {len(query_texts_prefixed)} queries...")
    query_embeddings = model.encode(
        query_texts_prefixed,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Normalize query embeddings
    query_embeddings = query_embeddings.astype(np.float32)
    faiss.normalize_L2(query_embeddings)
    logger.info(f"Query embeddings encoded and normalized")
    
    # Batch search
    top_k = max(k_values)
    logger.info(f"Searching index (top-{top_k})...")
    distances, indices = index.search(query_embeddings, top_k)
    logger.info(f"Search complete")
    
    # Compute metrics for each query
    logger.info(f"Computing metrics...")
    all_metrics = []
    per_query_metrics = []  # Collect per-query nDCG@10 for significance testing
    
    for i, query in enumerate(queries_to_eval):
        query_id = query['_id']
        retrieved_indices = indices[i]
        retrieved_doc_ids = [doc_ids[idx] for idx in retrieved_indices]
        
        qrels_for_query = qrels[query_id]
        metrics = compute_metrics(retrieved_doc_ids, qrels_for_query, k_values)
        all_metrics.append(metrics)
        per_query_metrics.append({'query_id': query_id, 'ndcg@10': metrics['ndcg@10']})
    
    # Average metrics across queries
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    logger.info(f"Metrics computed for {len(all_metrics)} queries")
    logger.info(f"  MRR: {avg_metrics['mrr']:.4f}")
    for k in k_values:
        logger.info(f"  P@{k}: {avg_metrics[f'precision@{k}']:.4f}, "
                   f"S@{k}: {avg_metrics[f'success@{k}']:.4f}, "
                   f"nDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
    
    # Save per-query metrics if not in test mode
    if not test_mode:
        per_query_dir = Path('results') / 'per_query'
        per_query_dir.mkdir(parents=True, exist_ok=True)
        per_query_file = per_query_dir / f"{model_name}_direct_{language}.csv"
        per_query_df = pd.DataFrame(per_query_metrics)
        per_query_df.to_csv(per_query_file, index=False)
        logger.info(f"Saved per-query metrics to {per_query_file}")
    
    # Build result row
    result = {
        'language': language,
        'model': model_name,
        'pipeline': 'direct',
        'num_queries': len(queries_to_eval),
        **avg_metrics
    }
    
    return result


def evaluate_translate_pipeline(model_name, model_id, language, config, k_values, test_mode, device, logger):
    """
    Evaluate translate pipeline: translated queries -> translated corpus.
    
    Returns:
        Dict with evaluation results for this language
    """
    if language == 'en':
        logger.info(f"Skipping translate pipeline for English")
        return None
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRANSLATE PIPELINE: {model_name} / {language}")
    logger.info(f"{'='*80}")
    
    # Load translated corpus index
    index_path = Path('embeddings') / f"{model_name}_{language}_translated_en_nllb.index"
    if not index_path.exists():
        logger.error(f"Translated index not found: {index_path}")
        return None
    
    logger.info(f"Loading translated index: {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info(f"Index loaded: {index.ntotal} vectors, dim={index.d}")
    
    # Load translated corpus to get doc_ids in index order
    corpus_path = Path('translated_corpus') / f"corpus_{language}_translated_en_nllb.json"
    logger.info(f"Loading translated corpus: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        translated_corpus = json.load(f)
    doc_ids = [doc['_id'] for doc in translated_corpus]
    logger.info(f"Translated corpus loaded: {len(doc_ids)} documents")
    
    # Load translated queries
    queries_path = Path('translated_queries') / f"queries_{language}_translated_en_nllb.json"
    logger.info(f"Loading translated queries: {queries_path}")
    with open(queries_path, 'r', encoding='utf-8') as f:
        translated_queries = json.load(f)
    translated_query_dict = {q['_id']: q['text'] for q in translated_queries}
    logger.info(f"Translated queries loaded: {len(translated_query_dict)} queries")
    
    # Load original queries dataset to get query IDs and qrels
    logger.info(f"Loading original queries dataset for IDs and qrels...")
    queries_dataset = load_dataset(config['query_dataset'], language, split='test')
    
    # Load qrels
    logger.info(f"Loading qrels...")
    qrels = load_qrels(config, language)
    logger.info(f"Qrels loaded: {len(qrels)} queries with relevance judgments")
    
    # Filter to queries with qrels and translated text, apply test mode limit
    queries_to_eval = []
    for query in queries_dataset:
        if query['_id'] in qrels and query['_id'] in translated_query_dict:
            queries_to_eval.append(query)
    
    if test_mode:
        max_queries = config['evaluation']['max_queries_per_lang']
        queries_to_eval = queries_to_eval[:max_queries]
        logger.info(f"TEST MODE: Limited to {len(queries_to_eval)} queries")
    else:
        logger.info(f"Using all {len(queries_to_eval)} queries with qrels and translations")
    
    # Load model and encode translated queries
    logger.info(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id)
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Extract translated query texts and apply prefix
    query_texts = [translated_query_dict[query['_id']] for query in queries_to_eval]
    query_texts_prefixed = [apply_query_prefix(text, model_name) for text in query_texts]
    
    if model_name == 'multilingual-e5-base':
        logger.info(f"Applied 'query: ' prefix for multilingual-e5-base")
    
    logger.info(f"Encoding {len(query_texts_prefixed)} translated queries...")
    query_embeddings = model.encode(
        query_texts_prefixed,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Normalize query embeddings
    query_embeddings = query_embeddings.astype(np.float32)
    faiss.normalize_L2(query_embeddings)
    logger.info(f"Query embeddings encoded and normalized")
    
    # Batch search
    top_k = max(k_values)
    logger.info(f"Searching index (top-{top_k})...")
    distances, indices = index.search(query_embeddings, top_k)
    logger.info(f"Search complete")
    
    # Compute metrics for each query
    logger.info(f"Computing metrics...")
    all_metrics = []
    per_query_metrics = []  # Collect per-query nDCG@10 for significance testing
    
    for i, query in enumerate(queries_to_eval):
        query_id = query['_id']
        retrieved_indices = indices[i]
        retrieved_doc_ids = [doc_ids[idx] for idx in retrieved_indices]
        
        qrels_for_query = qrels[query_id]
        metrics = compute_metrics(retrieved_doc_ids, qrels_for_query, k_values)
        all_metrics.append(metrics)
        per_query_metrics.append({'query_id': query_id, 'ndcg@10': metrics['ndcg@10']})
    
    # Average metrics across queries
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    logger.info(f"Metrics computed for {len(all_metrics)} queries")
    logger.info(f"  MRR: {avg_metrics['mrr']:.4f}")
    for k in k_values:
        logger.info(f"  P@{k}: {avg_metrics[f'precision@{k}']:.4f}, "
                   f"S@{k}: {avg_metrics[f'success@{k}']:.4f}, "
                   f"nDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
    
    # Save per-query metrics if not in test mode
    if not test_mode:
        per_query_dir = Path('results') / 'per_query'
        per_query_dir.mkdir(parents=True, exist_ok=True)
        per_query_file = per_query_dir / f"{model_name}_translate_{language}.csv"
        per_query_df = pd.DataFrame(per_query_metrics)
        per_query_df.to_csv(per_query_file, index=False)
        logger.info(f"Saved per-query metrics to {per_query_file}")
    
    # Build result row
    result = {
        'language': language,
        'model': model_name,
        'pipeline': 'translate',
        'num_queries': len(queries_to_eval),
        **avg_metrics
    }
    
    return result


def check_if_complete(model_name, pipeline, languages, config, test_mode, force):
    """
    Check if evaluation is already complete for this (model, pipeline) combination.
    
    Args:
        model_name: Name of the model
        pipeline: 'direct' or 'translate'
        languages: List of languages to evaluate
        config: Configuration dict
        test_mode: If True, always return False (test runs don't satisfy resumability)
        force: If True, always return False (force recomputation)
    
    Returns:
        True if evaluation is complete, False otherwise
    """
    # Force mode or test mode runs are never considered complete
    if force or test_mode:
        return False
    
    results_path = Path('results') / 'per_language' / f"{model_name}_{pipeline}.csv"
    
    if not results_path.exists():
        return False
    
    # Load existing results
    df = pd.read_csv(results_path)
    
    # Defensive checks for legacy/stale format
    if 'ndcg@10' not in df.columns:
        return False  # Missing required column - stale format
    
    if 'translation_model' in df.columns:
        return False  # Legacy format with translation_model column - stale
    
    if 'num_queries' not in df.columns:
        return False  # Missing query count column
    
    # Determine expected languages
    expected_languages = set(languages) if pipeline == 'direct' else set(lang for lang in languages if lang != 'en')
    existing_languages = set(df['language'].values)
    
    # Check language coverage
    if existing_languages != expected_languages:
        return False
    
    # Check query counts - must match full query set sizes
    for lang in expected_languages:
        # Load full query dataset to get true count
        queries_dataset = load_dataset(config['query_dataset'], lang, split='test')
        full_query_count = len(queries_dataset)
        
        # Check if this language's row has the correct count
        lang_row = df[df['language'] == lang]
        if lang_row.empty:
            return False
        
        actual_count = lang_row['num_queries'].iloc[0]
        if actual_count != full_query_count:
            return False  # Incomplete or test-mode result
    
    return True


def save_per_language_results(model_name, pipeline, results):
    """Save per-language results to CSV"""
    results_dir = Path('results') / 'per_language'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / f"{model_name}_{pipeline}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    return output_path


def build_summary_csv(logger):
    """Build summary.csv by averaging metrics across languages for each (model, pipeline)"""
    logger.info(f"\n{'='*80}")
    logger.info(f"BUILDING SUMMARY")
    logger.info(f"{'='*80}")
    
    per_language_dir = Path('results') / 'per_language'
    
    if not per_language_dir.exists():
        logger.warning(f"No per-language results found")
        return
    
    # Collect all per-language CSV files
    csv_files = list(per_language_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {per_language_dir}")
        return
    
    summary_rows = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            continue
        
        model = df['model'].iloc[0]
        pipeline = df['pipeline'].iloc[0]
        
        # Extract English nDCG@10 if it exists (direct pipeline only)
        en_ndcg10 = None
        en_row = df[df['language'] == 'en']
        if not en_row.empty:
            en_ndcg10 = en_row['ndcg@10'].iloc[0]
        
        # Exclude English from averaging - we only want to compare over non-English languages
        # where both pipelines (direct and translate) have results
        df_non_en = df[df['language'] != 'en']
        
        if df_non_en.empty:
            logger.warning(f"No non-English results for {model}/{pipeline}")
            continue
        
        # Average all numeric columns across non-English languages only
        numeric_cols = df_non_en.select_dtypes(include=[np.number]).columns
        avg_metrics = df_non_en[numeric_cols].mean().to_dict()
        
        row = {
            'model': model,
            'pipeline': pipeline,
            **{f'avg_{k}': v for k, v in avg_metrics.items() if k != 'num_queries'}
        }
        row['avg_num_queries'] = avg_metrics.get('num_queries', 0)
        row['en_ndcg@10'] = en_ndcg10 if en_ndcg10 is not None else np.nan
        
        summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path('results') / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"  {len(summary_rows)} model-pipeline combinations")
    else:
        logger.warning(f"No summary rows generated")


def build_winloss_csv(languages, logger):
    """Build winloss.csv comparing direct vs translate nDCG@10 for each (model, language)"""
    logger.info(f"\n{'='*80}")
    logger.info(f"BUILDING WIN/LOSS COMPARISON")
    logger.info(f"{'='*80}")
    
    per_language_dir = Path('results') / 'per_language'
    
    if not per_language_dir.exists():
        logger.warning(f"No per-language results found")
        return
    
    # Collect all direct and translate results by model
    direct_files = list(per_language_dir.glob('*_direct.csv'))
    translate_files = list(per_language_dir.glob('*_translate.csv'))
    
    winloss_rows = []
    
    # For each model that has both direct and translate results
    for direct_file in direct_files:
        model_name = direct_file.stem.replace('_direct', '')
        translate_file = per_language_dir / f"{model_name}_translate.csv"
        
        if not translate_file.exists():
            continue
        
        # Load both CSVs
        direct_df = pd.read_csv(direct_file)
        translate_df = pd.read_csv(translate_file)
        
        # For each language in translate (de, hi only)
        for _, translate_row in translate_df.iterrows():
            lang = translate_row['language']
            
            # Find corresponding direct row
            direct_row_data = direct_df[direct_df['language'] == lang]
            
            if direct_row_data.empty:
                continue
            
            direct_ndcg = direct_row_data['ndcg@10'].iloc[0]
            translate_ndcg = translate_row['ndcg@10']
            margin = translate_ndcg - direct_ndcg
            winner = 'Translate' if margin > 0 else 'Direct' if margin < 0 else 'Tie'
            
            winloss_rows.append({
                'Model': model_name,
                'Language': lang,
                'Direct_nDCG@10': direct_ndcg,
                'Translate_nDCG@10': translate_ndcg,
                'Winner': winner,
                'Margin': margin
            })
    
    if winloss_rows:
        winloss_df = pd.DataFrame(winloss_rows)
        winloss_path = Path('results') / 'winloss.csv'
        winloss_df.to_csv(winloss_path, index=False)
        logger.info(f"Win/loss comparison saved to {winloss_path}")
        logger.info(f"  {len(winloss_rows)} (model, language) comparisons")
    else:
        logger.warning(f"No win/loss rows generated")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multilingual IR with graded metrics")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--models', nargs='+', help='Models to evaluate (default: all from config)')
    parser.add_argument('--languages', nargs='+', help='Languages to evaluate (default: all from config)')
    parser.add_argument('--test', action='store_true', help='Test mode: limit queries per language')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
    parser.add_argument('--summary-only', action='store_true', help='Rebuild summary and winloss CSVs from existing per-language results without recomputing')
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging()
    
    # Explicit device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"{'='*80}")
    logger.info(f"MULTILINGUAL IR EVALUATION")
    logger.info(f"{'='*80}")
    
    # Summary-only mode: rebuild summary files without evaluation
    if args.summary_only:
        logger.info(f"SUMMARY-ONLY MODE: Rebuilding summary files from existing per-language results")
        logger.info(f"Log file: {log_file}")
        logger.info(f"{'='*80}")
        
        # Load config to get languages
        config = load_config(args.config)
        languages_to_eval = args.languages if args.languages else config['languages']
        
        # Build summary and winloss CSV files
        build_summary_csv(logger)
        build_winloss_csv(languages_to_eval, logger)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY REBUILD COMPLETE")
        logger.info(f"{'='*80}")
        return
    
    logger.info(f"Device: {device}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"Force mode: {args.force}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}")
    
    # Load config
    config = load_config(args.config)
    
    # Determine models and languages to evaluate
    models_to_eval = args.models if args.models else list(config['models'].keys())
    languages_to_eval = args.languages if args.languages else config['languages']
    k_values = config['evaluation']['k_values']
    
    logger.info(f"Models: {models_to_eval}")
    logger.info(f"Languages: {languages_to_eval}")
    logger.info(f"K values: {k_values}")
    
    # Evaluate each (model, pipeline) combination
    for model_name in models_to_eval:
        model_id = config['models'][model_name]
        
        # DIRECT PIPELINE
        pipeline = 'direct'
        if check_if_complete(model_name, pipeline, languages_to_eval, config, args.test, args.force):
            logger.info(f"\n{'='*80}")
            logger.info(f"SKIPPING {model_name} / {pipeline} - already complete")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"EVALUATING {model_name} / {pipeline}")
            logger.info(f"{'='*80}")
            
            direct_results = []
            for language in languages_to_eval:
                result = evaluate_direct_pipeline(
                    model_name, model_id, language, config,
                    k_values, args.test, device, logger
                )
                if result:
                    direct_results.append(result)
            
            if direct_results:
                output_path = save_per_language_results(model_name, pipeline, direct_results)
                logger.info(f"\nSaved {pipeline} results to {output_path}")
        
        # TRANSLATE PIPELINE
        pipeline = 'translate'
        translate_languages = [lang for lang in languages_to_eval if lang != 'en']
        
        if check_if_complete(model_name, pipeline, translate_languages, config, args.test, args.force):
            logger.info(f"\n{'='*80}")
            logger.info(f"SKIPPING {model_name} / {pipeline} - already complete")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"EVALUATING {model_name} / {pipeline}")
            logger.info(f"{'='*80}")
            
            translate_results = []
            for language in translate_languages:
                result = evaluate_translate_pipeline(
                    model_name, model_id, language, config,
                    k_values, args.test, device, logger
                )
                if result:
                    translate_results.append(result)
            
            if translate_results:
                output_path = save_per_language_results(model_name, pipeline, translate_results)
                logger.info(f"\nSaved {pipeline} results to {output_path}")
    
    # Build summary and winloss CSV files
    build_summary_csv(logger)
    build_winloss_csv(languages_to_eval, logger)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
