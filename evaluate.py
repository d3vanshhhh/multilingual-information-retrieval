import argparse
import yaml
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import warnings
from translation_cache import TranslationCache
from utils import translate_to_english


class Sem8Evaluator:
    """Main evaluation class for Semester VIII multilingual IR project."""
    
    def __init__(self, config_path, embeddings_dir, output_dir, cache_dir):
        self.config = self._load_config(config_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create output directories
        (self.output_dir / "per_language").mkdir(parents=True, exist_ok=True)
        
        # Initialize translation cache
        self.cache = TranslationCache(cache_dir=cache_dir)
        
        # Model cache
        self.loaded_models = {}
    
    def _load_config(self, config_path):
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_language_data(self, language, model_name, test_mode=False):
        """Load queries, qrels, and FAISS index for a language."""
        # Load FAISS index first to know how many docs we have
        index_path = self.embeddings_dir / f"{model_name}_{language}.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        index = faiss.read_index(str(index_path))
        num_docs_in_index = index.ntotal
        
        # Load corpus - MUST match what was used in generate_embeddings
        corpus_dataset = load_dataset(
            self.config['corpus_dataset'],
            language,
            split='test'
        )
        
        # Limit to same number of docs as in the FAISS index
        if len(corpus_dataset) > num_docs_in_index:
            corpus_dataset = corpus_dataset.select(range(num_docs_in_index))
        
        doc_ids = [doc['_id'] for doc in corpus_dataset]
        doc_ids_set = set(doc_ids)
        
        # Verify alignment
        if len(doc_ids) != index.ntotal:
            raise ValueError(f"Mismatch: {len(doc_ids)} doc_ids but {index.ntotal} vectors in index")
        
        # Load qrels
        qrels_dataset = load_dataset(
            self.config['qrels_dataset'],
            language,
            split='test'
        )
        qrels = {}
        for item in qrels_dataset:
            qid = item['query-id']
            corpus_id = item['corpus-id']
            # Only include qrels for docs that are in our corpus
            if corpus_id in doc_ids_set:
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(corpus_id)
        
        # Load queries - only keep queries that have at least one relevant doc in our corpus
        queries_dataset = load_dataset(
            self.config['query_dataset'],
            language,
            split='test'
        )
        queries = {}
        for q in queries_dataset:
            qid = q['_id']
            if qid in qrels and len(qrels[qid]) > 0:
                queries[qid] = q['text']
        
        print(f"  [{language}] Loaded {len(queries)} queries with relevant docs in corpus (out of {len(queries_dataset)} total)")
        
        return queries, qrels, index, doc_ids
    
    def translate_queries(self, queries, language, model_name="nllb"):
        """Batch translate queries to English with caching."""
        translated = {}
        for qid, text in tqdm(queries.items(), desc=f"Translating {language}", leave=False):
            translated[qid] = translate_to_english(text, language, model_name, self.cache)
        return translated
    
    def retrieve(self, query_embedding, faiss_index, k):
        """Retrieve top-k documents using FAISS."""
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = faiss_index.search(query_embedding, k)
        return indices[0], scores[0]
    
    def compute_metrics(self, retrieved_ids, qrels, k_values):
        """Compute nDCG, MRR, Success, and Precision metrics."""
        metrics = {}
        
        # MRR
        reciprocal_rank = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in qrels:
                reciprocal_rank = 1.0 / rank
                break
        metrics['mrr'] = reciprocal_rank
        
        # Compute for each k
        for k in k_values:
            retrieved_k = retrieved_ids[:k]
            relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in qrels]
            
            # Precision@k
            metrics[f'precision@{k}'] = len(relevant_retrieved) / k if k > 0 else 0
            
            # Success@k
            metrics[f'success@{k}'] = 1.0 if len(relevant_retrieved) > 0 else 0.0
            
            # nDCG@k
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved_k, 1):
                if doc_id in qrels:
                    dcg += 1.0 / np.log2(rank + 1)
            
            # Ideal DCG
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(qrels), k) + 1))
            
            metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def evaluate_model(self, model_name, pipeline_type, languages, test_mode=False, translation_model="nllb"):
        """Evaluate a model with specified pipeline on given languages."""
        pipeline_label = f"{pipeline_type}_{translation_model}" if pipeline_type == "translate" else pipeline_type
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} | Pipeline: {pipeline_label}")
        print(f"{'='*60}")
        
        # Load model
        if model_name not in self.loaded_models:
            try:
                model_id = self.config['models'][model_name]
                print(f"Loading model: {model_id}")
                self.loaded_models[model_name] = SentenceTransformer(model_id)
            except Exception as e:
                print(f"ERROR: Failed to load model {model_name}: {e}")
                return None
        
        model = self.loaded_models[model_name]
        results = []
        
        for lang in tqdm(languages, desc="Languages"):
            try:
                # Load data
                queries, qrels, index, doc_ids = self.load_language_data(lang, model_name)
                
                # Limit queries in test mode
                max_queries = self.config['evaluation']['max_queries_per_lang'] if test_mode else None
                if max_queries:
                    query_items = list(queries.items())[:max_queries]
                    queries = dict(query_items)
                
                # Translate if needed
                if pipeline_type == "translate":
                    queries = self.translate_queries(queries, lang, model_name=translation_model)
                
                # Evaluate each query
                lang_metrics = []
                for qid, query_text in tqdm(queries.items(), desc=f"{lang}", leave=False):
                    if qid not in qrels:
                        continue
                    
                    # Encode query
                    query_emb = model.encode(query_text, convert_to_numpy=True)
                    
                    # Retrieve
                    retrieved_indices, scores = self.retrieve(
                        query_emb,
                        index,
                        k=max(self.config['evaluation']['k_values'])
                    )
                    
                    # Map indices to doc IDs
                    retrieved_doc_ids = [doc_ids[idx] for idx in retrieved_indices]
                    
                    # Compute metrics
                    metrics = self.compute_metrics(
                        retrieved_doc_ids,
                        qrels[qid],
                        self.config['evaluation']['k_values']
                    )
                    lang_metrics.append(metrics)
                
                # Average metrics for this language
                if lang_metrics:
                    avg_metrics = {
                        'language': lang,
                        'model': model_name,
                        'pipeline': pipeline_label,
                        'translation_model': translation_model if pipeline_type == "translate" else "N/A",
                        'num_queries': len(lang_metrics)
                    }
                    
                    # Average each metric
                    for key in lang_metrics[0].keys():
                        avg_metrics[key] = np.mean([m[key] for m in lang_metrics])
                    
                    results.append(avg_metrics)
                
            except Exception as e:
                print(f"  ERROR processing {lang}: {e}")
                continue
        
        return pd.DataFrame(results) if results else None
    
    def run_full_evaluation(self, models, languages, test_mode=False):
        """Run evaluation for all model+pipeline combinations."""
        all_results = []
        
        for model_name in models:
            # Direct pipeline (no translation)
            df = self.evaluate_model(model_name, "direct", languages, test_mode)
            if df is not None:
                all_results.append(df)
            
            # Translate pipeline with NLLB
            df = self.evaluate_model(model_name, "translate", languages, test_mode, translation_model="nllb")
            if df is not None:
                all_results.append(df)
            
            # Translate pipeline with M2M100
            df = self.evaluate_model(model_name, "translate", languages, test_mode, translation_model="m2m100")
            if df is not None:
                all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True) if all_results else None
    
    def save_results(self, results_df):
        """Save per-language CSV, summary CSV, and win/loss analysis."""
        if results_df is None or results_df.empty:
            print("No results to save")
            return
        
        # Save per-language results for each model+pipeline
        for (model, pipeline), group in results_df.groupby(['model', 'pipeline']):
            filename = self.output_dir / "per_language" / f"{model}_{pipeline}.csv"
            group.to_csv(filename, index=False)
            print(f"Saved: {filename}")
        
        # Summary: average across all languages
        summary_data = []
        for (model, pipeline), group in results_df.groupby(['model', 'pipeline']):
            summary = {'model': model, 'pipeline': pipeline}
            
            # Average all numeric columns
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'num_queries':
                    summary[f'avg_{col}'] = group[col].mean()
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
        
        # Win/Loss analysis
        self._create_winloss_analysis(results_df)
    
    def _create_winloss_analysis(self, results_df):
        """Create win/loss comparison between direct and translate pipelines."""
        winloss_data = []
        
        for lang in results_df['language'].unique():
            lang_data = results_df[results_df['language'] == lang]
            
            direct = lang_data[lang_data['pipeline'] == 'direct']
            translate_nllb = lang_data[lang_data['pipeline'] == 'translate_nllb']
            translate_m2m100 = lang_data[lang_data['pipeline'] == 'translate_m2m100']
            
            if not direct.empty:
                direct_ndcg = direct['ndcg@10'].mean()
                
                # Compare with NLLB
                if not translate_nllb.empty:
                    nllb_ndcg = translate_nllb['ndcg@10'].mean()
                    winner_nllb = 'Direct' if direct_ndcg > nllb_ndcg else 'Translate_NLLB'
                    margin_nllb = abs(direct_ndcg - nllb_ndcg)
                    
                    winloss_data.append({
                        'Language': lang,
                        'Comparison': 'Direct vs NLLB',
                        'Direct_nDCG': round(direct_ndcg, 4),
                        'Translate_nDCG': round(nllb_ndcg, 4),
                        'Winner': winner_nllb,
                        'Margin': round(margin_nllb, 4)
                    })
                
                # Compare with M2M100
                if not translate_m2m100.empty:
                    m2m100_ndcg = translate_m2m100['ndcg@10'].mean()
                    winner_m2m100 = 'Direct' if direct_ndcg > m2m100_ndcg else 'Translate_M2M100'
                    margin_m2m100 = abs(direct_ndcg - m2m100_ndcg)
                    
                    winloss_data.append({
                        'Language': lang,
                        'Comparison': 'Direct vs M2M100',
                        'Direct_nDCG': round(direct_ndcg, 4),
                        'Translate_nDCG': round(m2m100_ndcg, 4),
                        'Winner': winner_m2m100,
                        'Margin': round(margin_m2m100, 4)
                    })
                
                # Compare NLLB vs M2M100
                if not translate_nllb.empty and not translate_m2m100.empty:
                    winner_trans = 'NLLB' if nllb_ndcg > m2m100_ndcg else 'M2M100'
                    margin_trans = abs(nllb_ndcg - m2m100_ndcg)
                    
                    winloss_data.append({
                        'Language': lang,
                        'Comparison': 'NLLB vs M2M100',
                        'Direct_nDCG': round(nllb_ndcg, 4),
                        'Translate_nDCG': round(m2m100_ndcg, 4),
                        'Winner': winner_trans,
                        'Margin': round(margin_trans, 4)
                    })
        
        winloss_df = pd.DataFrame(winloss_data)
        winloss_path = self.output_dir / "winloss.csv"
        winloss_df.to_csv(winloss_path, index=False)
        print(f"Saved: {winloss_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multilingual IR models")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory with embeddings')
    parser.add_argument('--output-dir', default='results', help='Directory for results')
    parser.add_argument('--cache-dir', default='cache', help='Directory for translation cache')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    parser.add_argument('--languages', nargs='+', help='Specific languages to evaluate')
    parser.add_argument('--test', action='store_true', help='Test mode: limited data')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Sem8Evaluator(
        args.config,
        args.embeddings_dir,
        args.output_dir,
        args.cache_dir
    )
    
    # Determine models and languages
    models = args.models if args.models else list(evaluator.config['models'].keys())
    languages = args.languages if args.languages else evaluator.config['languages']
    
    print(f"Evaluating {len(models)} models on {len(languages)} languages")
    print(f"Test mode: {args.test}")
    
    # Run evaluation
    results = evaluator.run_full_evaluation(models, languages, test_mode=args.test)
    
    # Save results
    if results is not None:
        evaluator.save_results(results)
        print("\nEvaluation complete!")
    else:
        print("\nNo results generated")
    
    # Print cache stats
    evaluator.cache.save()
    evaluator.cache.print_stats()


if __name__ == "__main__":
    main()
