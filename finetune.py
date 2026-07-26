import argparse
import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import faiss
from datasets import load_dataset
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_training_data(data_path):
    """Load training data from JSONL file."""
    print(f"Loading training data from {data_path}...")
    
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Create InputExample with query, positive, and negatives
            # sentence-transformers expects: (anchor, positive, negative)
            for negative in item['negatives']:
                examples.append(InputExample(
                    texts=[item['query'], item['positive'], negative]
                ))
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def evaluate_model_on_english(model, config, embeddings_dir):
    """Quick evaluation on English to measure improvement."""
    print("\nEvaluating on English queries...")
    
    try:
        # Load English queries and qrels
        queries_dataset = load_dataset(
            config['query_dataset'],
            'en',
            split='train'
        )
        queries = {q['id']: q['text'] for q in queries_dataset.select(range(50))}
        
        qrels_dataset = load_dataset(
            config['qrels_dataset'],
            'en',
            split='train'
        )
        qrels = {}
        for item in qrels_dataset:
            qid = item['query_id']
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(item['doc_id'])
        
        # Load corpus
        corpus_dataset = load_dataset(
            config['corpus_dataset'],
            'en',
            split='train'
        )
        corpus_dataset = corpus_dataset.select(range(500))
        doc_ids = [doc['id'] for doc in corpus_dataset]
        texts = [doc['text'] for doc in corpus_dataset]
        
        # Generate embeddings
        print("  Encoding corpus...")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Evaluate
        ndcg_scores = []
        for qid, query_text in queries.items():
            if qid not in qrels:
                continue
            
            # Encode and search
            query_emb = model.encode(query_text, convert_to_numpy=True)
            query_emb = np.array([query_emb]).astype('float32')
            faiss.normalize_L2(query_emb)
            
            scores, indices = index.search(query_emb, 10)
            retrieved_doc_ids = [doc_ids[idx] for idx in indices[0]]
            
            # Compute nDCG@10
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved_doc_ids, 1):
                if doc_id in qrels[qid]:
                    dcg += 1.0 / np.log2(rank + 1)
            
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(qrels[qid]), 10) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        print(f"  nDCG@10 on English: {avg_ndcg:.4f}")
        return avg_ndcg
    
    except Exception as e:
        print(f"  Evaluation failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune multilingual embedding model")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--data', default='training_data/finetune_dataset.jsonl', help='Training data path')
    parser.add_argument('--base-model', default='intfloat/multilingual-e5-base', help='Base model to fine-tune')
    parser.add_argument('--output-dir', default='models/finetuned-e5-base', help='Output directory')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--test', action='store_true', help='Test mode: 1 epoch, small data')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Adjust for test mode
    if args.test:
        args.epochs = 1
        args.batch_size = 8
        print("Test mode: 1 epoch, batch size 8")
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    
    # Evaluate before training
    print("\n=== Before Fine-tuning ===")
    ndcg_before = evaluate_model_on_english(model, config, args.embeddings_dir)
    
    # Load training data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        print("Run negative_mining.py first to generate training data")
        return
    
    train_examples = load_training_data(data_path)
    
    # Limit data in test mode
    if args.test and len(train_examples) > 100:
        train_examples = train_examples[:100]
        print(f"Test mode: using only {len(train_examples)} examples")
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Define loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    output_path = Path(args.output_dir)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=100,
        output_path=str(output_path),
        show_progress_bar=True,
        save_best_model=True
    )
    
    print(f"\nModel saved to {output_path}")
    
    # Evaluate after training
    print("\n=== After Fine-tuning ===")
    finetuned_model = SentenceTransformer(str(output_path))
    ndcg_after = evaluate_model_on_english(finetuned_model, config, args.embeddings_dir)
    
    # Summary
    print("\n" + "="*60)
    print("FINE-TUNING SUMMARY")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Epochs: {args.epochs}")
    print(f"nDCG@10 before: {ndcg_before:.4f}")
    print(f"nDCG@10 after: {ndcg_after:.4f}")
    print(f"Improvement: {(ndcg_after - ndcg_before):.4f} ({((ndcg_after/ndcg_before - 1)*100):.1f}%)")
    
    # Generate embeddings for finetuned model
    print("\nGenerating embeddings for finetuned model...")
    print("Run: python generate_embeddings.py --model finetuned-e5-base")


if __name__ == "__main__":
    main()
