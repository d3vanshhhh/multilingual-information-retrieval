"""
Diagnostic script to audit dataset statistics for multilingual IR project.
This is a read-only audit - no modifications are made to any files.
"""

import yaml
from datasets import load_dataset
from collections import Counter

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def audit_corpus(corpus_name, language):
    """Audit corpus dataset for a given language"""
    print(f"\n{'='*80}")
    print(f"CORPUS AUDIT - Language: {language}")
    print(f"{'='*80}")
    
    try:
        corpus = load_dataset(corpus_name, language, split='test')
        
        # Total document count
        total_docs = len(corpus)
        print(f"\nTotal document count: {total_docs}")
        
        # Feature/column names
        print(f"\nFeature/column names: {list(corpus.features.keys())}")
        
        # Check for empty or null 'text' field
        empty_text_count = 0
        for doc in corpus:
            text = doc.get('text', '')
            if text is None or (isinstance(text, str) and text.strip() == ''):
                empty_text_count += 1
        print(f"\nDocuments with empty or null 'text' field: {empty_text_count}")
        
        # Check for duplicate '_id' values
        ids = [doc['_id'] for doc in corpus]
        id_counts = Counter(ids)
        duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
        print(f"\nNumber of duplicate '_id' values: {len(duplicates)}")
        if duplicates:
            print(f"Example duplicates (first 5): {list(duplicates.items())[:5]}")
        
        return corpus, ids
        
    except Exception as e:
        print(f"\nERROR loading corpus for {language}: {e}")
        return None, []

def audit_queries(query_name, language):
    """Audit query dataset for a given language"""
    print(f"\n{'='*80}")
    print(f"QUERY AUDIT - Language: {language}")
    print(f"{'='*80}")
    
    try:
        queries = load_dataset(query_name, language, split='test')
        
        # Total query count
        total_queries = len(queries)
        print(f"\nTotal query count: {total_queries}")
        
        # Feature/column names
        print(f"\nFeature/column names: {list(queries.features.keys())}")
        
        # Check for empty or null 'text' field
        empty_text_count = 0
        for query in queries:
            text = query.get('text', '')
            if text is None or (isinstance(text, str) and text.strip() == ''):
                empty_text_count += 1
        print(f"\nQueries with empty or null 'text' field: {empty_text_count}")
        
        # Check for duplicate '_id' values
        ids = [query['_id'] for query in queries]
        id_counts = Counter(ids)
        duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
        print(f"\nNumber of duplicate '_id' values: {len(duplicates)}")
        if duplicates:
            print(f"Example duplicates (first 5): {list(duplicates.items())[:5]}")
        
        return queries, ids
        
    except Exception as e:
        print(f"\nERROR loading queries for {language}: {e}")
        return None, []

def audit_qrels(qrels_name, language):
    """Audit qrels dataset for a given language"""
    print(f"\n{'='*80}")
    print(f"QRELS AUDIT - Language: {language}")
    print(f"{'='*80}")
    
    try:
        qrels = load_dataset(qrels_name, language, split='test')
        
        # Total qrels row count
        total_qrels = len(qrels)
        print(f"\nTotal qrels row count: {total_qrels}")
        
        # Feature/column names
        print(f"\nFeature/column names: {list(qrels.features.keys())}")
        
        # Collect unique query-ids and corpus-ids
        query_ids = set()
        corpus_ids = set()
        
        for qrel in qrels:
            query_ids.add(qrel['query-id'])
            corpus_ids.add(qrel['corpus-id'])
        
        print(f"\nNumber of unique query-ids referenced: {len(query_ids)}")
        print(f"Number of unique corpus-ids referenced: {len(corpus_ids)}")
        
        return qrels, query_ids, corpus_ids
        
    except Exception as e:
        print(f"\nERROR loading qrels for {language}: {e}")
        return None, set(), set()

def cross_check_consistency(qrels, qrels_query_ids, qrels_corpus_ids, 
                           corpus_ids, query_ids, language):
    """Cross-check consistency between qrels and full corpus/query sets"""
    print(f"\n{'='*80}")
    print(f"CONSISTENCY CHECK - Language: {language}")
    print(f"{'='*80}")
    
    if qrels is None:
        print("\nSkipping consistency check due to missing qrels data")
        return
    
    # Convert lists to sets for efficient lookup
    corpus_id_set = set(corpus_ids)
    query_id_set = set(query_ids)
    
    # Check for missing corpus-ids
    missing_corpus_ids = []
    for qrel in qrels:
        corpus_id = qrel['corpus-id']
        if corpus_id not in corpus_id_set:
            missing_corpus_ids.append(corpus_id)
    
    print(f"\nQrels rows referencing corpus-id NOT in full corpus: {len(missing_corpus_ids)}")
    if missing_corpus_ids:
        unique_missing = list(set(missing_corpus_ids))[:10]
        print(f"Example missing corpus-ids (first 10): {unique_missing}")
    
    # Check for missing query-ids
    missing_query_ids = []
    for qrel in qrels:
        query_id = qrel['query-id']
        if query_id not in query_id_set:
            missing_query_ids.append(query_id)
    
    print(f"\nQrels rows referencing query-id NOT in full query set: {len(missing_query_ids)}")
    if missing_query_ids:
        unique_missing = list(set(missing_query_ids))[:10]
        print(f"Example missing query-ids (first 10): {unique_missing}")
    
    # Count queries with at least one valid relevant doc
    queries_with_valid_docs = set()
    for qrel in qrels:
        query_id = qrel['query-id']
        corpus_id = qrel['corpus-id']
        # Only count if both ids exist in their respective datasets
        if query_id in query_id_set and corpus_id in corpus_id_set:
            queries_with_valid_docs.add(query_id)
    
    print(f"\nQueries with at least one valid relevant doc in full corpus: {len(queries_with_valid_docs)}")
    print(f"Total unique queries in qrels: {len(qrels_query_ids)}")
    if len(qrels_query_ids) > 0:
        usability_pct = (len(queries_with_valid_docs) / len(qrels_query_ids)) * 100
        print(f"Usability percentage: {usability_pct:.2f}%")

def main():
    print("="*80)
    print("MULTILINGUAL IR DATASET AUDIT")
    print("="*80)
    print("\nThis is a read-only diagnostic audit.")
    print("Loading datasets from HuggingFace...")
    
    # Load configuration
    config = load_config()
    
    corpus_name = config['corpus_dataset']
    query_name = config['query_dataset']
    qrels_name = config['qrels_dataset']
    languages = config['languages']
    
    print(f"\nCorpus dataset: {corpus_name}")
    print(f"Query dataset: {query_name}")
    print(f"Qrels dataset: {qrels_name}")
    print(f"Languages: {languages}")
    
    # Process each language
    for lang in languages:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING LANGUAGE: {lang.upper()}")
        print(f"{'#'*80}")
        
        # Audit corpus
        corpus, corpus_ids = audit_corpus(corpus_name, lang)
        
        # Audit queries
        queries, query_ids = audit_queries(query_name, lang)
        
        # Audit qrels
        qrels, qrels_query_ids, qrels_corpus_ids = audit_qrels(qrels_name, lang)
        
        # Cross-check consistency
        if corpus and queries and qrels:
            cross_check_consistency(qrels, qrels_query_ids, qrels_corpus_ids,
                                  corpus_ids, query_ids, lang)
    
    print("\n\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
