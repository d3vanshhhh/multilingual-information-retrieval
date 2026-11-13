"""
Check the new boundary-aware sentence splitter against the old simple splitter.
Reports differences across all languages in the corpus and queries.
"""
import re
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from datasets import load_dataset
from utils import split_sentences as new_split_sentences

# Old splitter (simple version without boundary awareness)
def old_split_sentences(text):
    """Old splitter: regex split + merge <3 words, no boundary awareness."""
    pieces = re.split(r'(?<=[.!?।॥؟۔…])\s+', text.strip())
    pieces = [p for p in pieces if p]
    
    if not pieces:
        return []
    
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if len(piece.split()) < 3:
            merged[-1] = merged[-1] + ' ' + piece
        else:
            merged.append(piece)
    
    return merged


def print_splits(text, lang_label):
    """Print numbered pieces for a test string."""
    pieces = new_split_sentences(text)
    print(f"\n{lang_label}:")
    for i, piece in enumerate(pieces, 1):
        print(f"  {i}. {piece}")


def compare_on_dataset(lang, file_type, texts, text_ids):
    """
    Compare old vs new splitter on a set of texts.
    Returns: (differ_count, total_count, remerged_boundaries, examples)
    """
    differ_count = 0
    total_texts = len(texts)
    remerged_boundaries = 0
    examples = []
    
    for text_id, text in zip(text_ids, texts):
        old_pieces = old_split_sentences(text)
        new_pieces = new_split_sentences(text)
        
        if old_pieces != new_pieces:
            differ_count += 1
            remerged_boundaries += len(old_pieces) - len(new_pieces)
            
            # Save first 3 differing examples
            if len(examples) < 3:
                # Find first differing boundary position
                context = ""
                for i, (old_p, new_p) in enumerate(zip(old_pieces, new_pieces)):
                    if old_p != new_p:
                        # Extract 80 chars around this boundary in the original text
                        # Find where old_p ends in text
                        pos = text.find(old_p)
                        if pos >= 0:
                            end_pos = pos + len(old_p)
                            start_context = max(0, end_pos - 40)
                            end_context = min(len(text), end_pos + 40)
                            context = text[start_context:end_context]
                        break
                
                examples.append({
                    'id': text_id,
                    'context': context,
                    'old_pieces_at_boundary': old_pieces[:3] if len(old_pieces) > 0 else []
                })
    
    return differ_count, total_texts, remerged_boundaries, examples


def main():
    # Test strings
    print("=== TEST STRINGS ===")
    
    test_strings = [
        ("de", "Am 24. Februar 2014 kündigte Samsung an, dass künftige Smartwatches auf Tizen basieren. Das erste Modell erschien im März."),
        ("de", "Der Name stammt aus dem Italienischen und wird seit dem frühen 16. Jahrhundert (erstmals 1518 belegt) verwendet."),
        ("de", "Was macht den Pas de deux von Odette und Siegfried im 2. Akt von Schwanensee so bedeutend?"),
        ("de", "J. R. R. Tolkien schrieb den Roman. Er erschien 1954."),
        ("de", "Das gilt z. B. für Dr. Müller und ca. 20 weitere Personen. Danach endete die Sitzung."),
        ("cs", "Dne 1. října 1979 byla otevřena nová trať. Provoz byl zahájen v pondělí."),
        ("da", "Den 1. oktober 1979 åbnede metroen. Den er 200 km lang."),
        ("bg", "През 1990 г. страната се промени. Новото правителство встъпи в длъжност."),
        ("fi", "Metro avattiin 1. lokakuuta 1979. Se on 200 kilometriä pitkä."),
        ("hi", "भारत एक देश है। इसकी राजधानी दिल्ली है।"),
    ]
    
    for lang, text in test_strings:
        print_splits(text, lang)
    
    # Load config to get all languages
    print("\n\n=== FULL DATASET COMPARISON ===")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    languages = config['languages']
    
    # Results storage
    results = []
    
    for lang in languages:
        if lang == 'en':
            continue  # Skip English
        
        print(f"\nProcessing {lang}...")
        
        # Load corpus from HuggingFace
        try:
            corpus_ds = load_dataset(config['corpus_dataset'], lang, split='test')
            corpus_texts = [doc['text'] for doc in corpus_ds]
            corpus_ids = [doc['_id'] for doc in corpus_ds]
            
            corpus_differ, corpus_total, corpus_remerged, corpus_examples = compare_on_dataset(
                lang, 'corpus', corpus_texts, corpus_ids
            )
        except Exception as e:
            print(f"  Error loading corpus for {lang}: {e}")
            corpus_differ = corpus_total = corpus_remerged = 0
            corpus_examples = []
        
        # Load queries from HuggingFace
        try:
            queries_ds = load_dataset(config['query_dataset'], lang, split='test')
            query_texts = [q['text'] for q in queries_ds]
            query_ids = [q['_id'] for q in queries_ds]
            query_texts = [q['text'] for q in queries_ds]
            query_ids = [q['_id'] for q in queries_ds]
            
            queries_differ, queries_total, queries_remerged, queries_examples = compare_on_dataset(
                lang, 'queries', query_texts, query_ids
            )
        except Exception as e:
            print(f"  Error loading queries for {lang}: {e}")
            queries_differ = queries_total = queries_remerged = 0
            queries_examples = []
        
        results.append({
            'lang': lang,
            'corpus_differ': corpus_differ,
            'corpus_total': corpus_total,
            'corpus_remerged': corpus_remerged,
            'corpus_examples': corpus_examples,
            'queries_differ': queries_differ,
            'queries_total': queries_total,
            'queries_remerged': queries_remerged,
            'queries_examples': queries_examples
        })
    
    # Print report
    print("\n\n=== REPORT ===")
    print(f"{'Lang':<6} {'Type':<8} {'Differ':>8} {'Total':>8} {'Differ%':>8} {'Remerged':>10}")
    print("-" * 60)
    
    for r in results:
        if r['corpus_total'] > 0:
            corpus_pct = 100.0 * r['corpus_differ'] / r['corpus_total']
            print(f"{r['lang']:<6} {'corpus':<8} {r['corpus_differ']:>8} {r['corpus_total']:>8} {corpus_pct:>7.2f}% {r['corpus_remerged']:>10}")
            
            # Print examples
            for ex in r['corpus_examples']:
                print(f"       Ex: {ex['id']}: ...{ex['context'][:80]}...")
        
        if r['queries_total'] > 0:
            queries_pct = 100.0 * r['queries_differ'] / r['queries_total']
            print(f"{r['lang']:<6} {'queries':<8} {r['queries_differ']:>8} {r['queries_total']:>8} {queries_pct:>7.2f}% {r['queries_remerged']:>10}")
            
            # Print examples
            for ex in r['queries_examples']:
                print(f"       Ex: {ex['id']}: ...{ex['context'][:80]}...")
    
    print("\n=== END REPORT ===")


if __name__ == '__main__':
    main()
