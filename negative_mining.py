import argparse
import yaml
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def mine_hard_negatives(model_name, language, config, embeddings_dir, output_dir, n_negatives=5, test_mode=False):
    """
    Mine hard negatives for a specific model and language.
    
    Hard negatives are documents that are retrieved but not relevant.
    """
    print(f"\n[{model_name}] Mining hard negatives for {language}...")
    
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        model_id = config['models'][model_name]
        model = SentenceTransformer(model_id)
        
        # Load FAISS index
        index_path = embeddings_dir / f"{model_name}_{language}.index"
        if not index_path.exists():
            print(f"  Index not found: {index_path}, skipping")
            return 0
        
        index = faiss.read_index(str(index_path))
        
        # Load queries
        queries_dataset = load_dataset(
            config['query_dataset'],
            language,
            split='train'
        )
        
        # Load qrels
        qrels_dataset = load_dataset(
            config['qrels_dataset'],
            language,
            split='train'
        )
        qrels = {}
        for item in qrels_dataset:
            qid = item['query_id']
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(item['doc_id'])
        
        # Load corpus
        corpus_dataset = load_dataset(
            config['corpus_dataset'],
            language,
            split='train'
        )
        doc_ids = [doc['id'] for doc in corpus_dataset]
        doc_texts = {doc['id']: doc['text'] for doc in corpus_dataset}
        
        # Limit in test mode
        max_queries = config['evaluation']['max_queries_per_lang'] if test_mode else None
        if max_queries:
            queries_dataset = queries_dataset.select(range(min(max_queries, len(queries_dataset))))
        
        # Mine negatives
        mined_data = []
        for query in tqdm(queries_dataset, desc=f"{language}"):
            qid = query['id']
            query_text = query['text']
            
            if qid not in qrels:
                continue
            
            # Encode query
            query_emb = model.encode(query_text, convert_to_numpy=True)
            query_emb = np.array([query_emb]).astype('float32')
            faiss.normalize_L2(query_emb)
            
            # Retrieve top-20
            scores, indices = index.search(query_emb, 20)
            retrieved_doc_ids = [doc_ids[idx] for idx in indices[0]]
            
            # Find hard negatives (retrieved but not relevant)
            hard_negatives = []
            for doc_id in retrieved_doc_ids:
                if doc_id not in qrels[qid]:
                    hard_negatives.append(doc_texts[doc_id])
                    if len(hard_negatives) >= n_negatives:
                        break
            
            # Get a positive example
            positive_docs = [doc_texts[doc_id] for doc_id in qrels[qid] if doc_id in doc_texts]
            if positive_docs and hard_negatives:
                mined_data.append({
                    'query': query_text,
                    'positive': positive_docs[0],
                    'negatives': hard_negatives
                })
        
        # Save to JSONL
        output_file = output_dir / f"{language}_negatives.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in mined_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  Mined {len(mined_data)} examples, saved to {output_file}")
        return len(mined_data)
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def prepare_training_data(negatives_dir, output_path):
    """
    Merge all JSONL files and prepare for sentence-transformers training.
    """
    print("\nPreparing training data...")
    
    negatives_dir = Path(negatives_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Collect all JSONL files
    for jsonl_file in negatives_dir.rglob("*.jsonl"):
        print(f"  Loading {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    # Save merged data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(all_data)} training examples to {output_path}")
    return len(all_data)


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for fine-tuning")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory with embeddings')
    parser.add_argument('--output-dir', default='negatives', help='Output directory')
    parser.add_argument('--model', help='Specific model to mine for')
    parser.add_argument('--language', help='Specific language to mine for')
    parser.add_argument('--n-negatives', type=int, default=5, help='Number of negatives per query')
    parser.add_argument('--test', action='store_true', help='Test mode: limited data')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare training data from existing negatives')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.prepare_only:
        # Just merge existing negatives
        prepare_training_data(args.output_dir, 'training_data/finetune_dataset.jsonl')
        return
    
    # Determine models and languages
    models = [args.model] if args.model else list(config['models'].keys())
    languages = [args.language] if args.language else config['languages']
    
    print(f"Mining hard negatives for {len(models)} model(s) and {len(languages)} language(s)")
    print(f"Test mode: {args.test}")
    
    total_examples = 0
    
    for model_name in models:
        for lang in languages:
            examples = mine_hard_negatives(
                model_name,
                lang,
                config,
                args.embeddings_dir,
                args.output_dir,
                n_negatives=args.n_negatives,
                test_mode=args.test
            )
            total_examples += examples
    
    print(f"\nTotal examples mined: {total_examples}")
    
    # Prepare training data
    if total_examples > 0:
        prepare_training_data(args.output_dir, 'training_data/finetune_dataset.jsonl')


if __name__ == "__main__":
    main()
