"""
Diagnostic script to audit qrels scores and corpus load-order determinism.
This is a read-only audit - no modifications are made to any files.
"""

import yaml
from datasets import load_dataset
from collections import Counter

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def audit_qrels_scores(qrels_name, language):
    """Audit qrels score field for a given language"""
    print(f"\n{'='*80}")
    print(f"QRELS SCORE FIELD AUDIT - Language: {language}")
    print(f"{'='*80}")
    
    try:
        qrels = load_dataset(qrels_name, language, split='test')
        
        # Collect all scores
        scores = [qrel['score'] for qrel in qrels]
        
        # Count distinct values
        score_counts = Counter(scores)
        print(f"\nDistinct score values and their counts:")
        for score_value in sorted(score_counts.keys()):
            print(f"  Score {score_value}: {score_counts[score_value]} rows")
        
        # Min and max scores
        min_score = min(scores)
        max_score = max(scores)
        print(f"\nMinimum score value: {min_score}")
        print(f"Maximum score value: {max_score}")
        
        # Check for scores <= 0
        non_positive_count = sum(1 for s in scores if s <= 0)
        print(f"\nRows with score <= 0: {non_positive_count}")
        if non_positive_count > 0:
            print(f"  WARNING: Found {non_positive_count} rows with non-positive scores!")
        else:
            print(f"  All scores are positive (> 0)")
        
    except Exception as e:
        print(f"\nERROR loading qrels for {language}: {e}")

def audit_corpus_load_order(corpus_name, language):
    """Check corpus load-order determinism for a given language"""
    print(f"\n{'='*80}")
    print(f"CORPUS LOAD-ORDER DETERMINISM - Language: {language}")
    print(f"{'='*80}")
    
    try:
        # First load
        print(f"\nLoading corpus (first load)...")
        corpus_load_1 = load_dataset(corpus_name, language, split='test')
        ids_load_1 = [doc['_id'] for doc in corpus_load_1]
        print(f"First load: {len(ids_load_1)} documents")
        
        # Second load (independent)
        print(f"Loading corpus (second load)...")
        corpus_load_2 = load_dataset(corpus_name, language, split='test')
        ids_load_2 = [doc['_id'] for doc in corpus_load_2]
        print(f"Second load: {len(ids_load_2)} documents")
        
        # Compare lengths first
        if len(ids_load_1) != len(ids_load_2):
            print(f"\nERROR: Load lengths differ!")
            print(f"  First load: {len(ids_load_1)}")
            print(f"  Second load: {len(ids_load_2)}")
            return
        
        # Compare order element by element
        identically_ordered = True
        first_divergence_index = None
        
        for i in range(len(ids_load_1)):
            if ids_load_1[i] != ids_load_2[i]:
                identically_ordered = False
                first_divergence_index = i
                break
        
        if identically_ordered:
            print(f"\n✓ RESULT: Both loads are IDENTICALLY ORDERED")
            print(f"  All {len(ids_load_1)} document IDs match in the same order")
        else:
            print(f"\n✗ RESULT: Loads are NOT identically ordered")
            print(f"  First divergence at index: {first_divergence_index}")
            print(f"  First load ID at index {first_divergence_index}: {ids_load_1[first_divergence_index]}")
            print(f"  Second load ID at index {first_divergence_index}: {ids_load_2[first_divergence_index]}")
            
            # Show a few more divergences if they exist
            divergences = []
            for i in range(len(ids_load_1)):
                if ids_load_1[i] != ids_load_2[i]:
                    divergences.append((i, ids_load_1[i], ids_load_2[i]))
                    if len(divergences) >= 5:
                        break
            
            if len(divergences) > 1:
                print(f"\n  Additional divergences (up to 5):")
                for idx, id1, id2 in divergences[1:]:
                    print(f"    Index {idx}: load1={id1}, load2={id2}")
        
        # Also check if they're the same set (just in different order)
        set_load_1 = set(ids_load_1)
        set_load_2 = set(ids_load_2)
        
        if set_load_1 == set_load_2:
            print(f"\n  Note: The two loads contain the same set of IDs")
            print(f"        (same documents, potentially different order)")
        else:
            only_in_1 = set_load_1 - set_load_2
            only_in_2 = set_load_2 - set_load_1
            print(f"\n  Note: The two loads contain DIFFERENT sets of IDs")
            print(f"        IDs only in first load: {len(only_in_1)}")
            print(f"        IDs only in second load: {len(only_in_2)}")
            if only_in_1:
                print(f"        Examples from first load only: {list(only_in_1)[:5]}")
            if only_in_2:
                print(f"        Examples from second load only: {list(only_in_2)[:5]}")
        
    except Exception as e:
        print(f"\nERROR during corpus load-order check for {language}: {e}")

def main():
    print("="*80)
    print("QRELS SCORES AND CORPUS LOAD-ORDER AUDIT")
    print("="*80)
    print("\nThis is a read-only diagnostic audit.")
    print("Loading datasets from HuggingFace...")
    
    # Load configuration
    config = load_config()
    
    corpus_name = config['corpus_dataset']
    qrels_name = config['qrels_dataset']
    languages = config['languages']
    
    print(f"\nCorpus dataset: {corpus_name}")
    print(f"Qrels dataset: {qrels_name}")
    print(f"Languages: {languages}")
    
    # Process each language
    for lang in languages:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING LANGUAGE: {lang.upper()}")
        print(f"{'#'*80}")
        
        # Audit qrels scores
        audit_qrels_scores(qrels_name, lang)
        
        # Audit corpus load-order determinism
        audit_corpus_load_order(corpus_name, lang)
    
    print("\n\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
