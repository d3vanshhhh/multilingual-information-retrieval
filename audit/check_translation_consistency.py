#!/usr/bin/env python
"""
Check translation consistency across different batching strategies.
GPU script to verify that sentence-level translation produces deterministic results.
"""
import random
import yaml
from pathlib import Path
from datasets import load_dataset
import sys
sys.path.insert(0, '.')
from utils import translate_batch_to_english


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("="*80)
    print("TRANSLATION CONSISTENCY CHECK")
    print("="*80)
    
    config = load_config()
    
    # Load German corpus
    print("\nLoading German corpus...")
    corpus_dataset = load_dataset(config['corpus_dataset'], 'de', split='test')
    corpus_list = list(corpus_dataset)
    
    # Filter to documents with at least 40 words
    eligible_docs = [doc for doc in corpus_list if len(doc['text'].split()) >= 40]
    print(f"Found {len(eligible_docs)} documents with >= 40 words")
    
    # Sample 60 documents
    rng = random.Random(42)
    sampled_docs = rng.sample(eligible_docs, 60)
    doc_texts = [doc['text'] for doc in sampled_docs]
    doc_ids = [doc['_id'] for doc in sampled_docs]
    
    print(f"Sampled 60 documents for testing")
    
    # Test A: All at once
    print("\nTest A: Translating all 60 documents at once...")
    translations_a = translate_batch_to_english(doc_texts, 'de', model_name='nllb-v2', cache=None)
    print(f"  Result: {len(translations_a)} translations")
    
    # Test B: In 6 chunks of 10
    print("\nTest B: Translating in 6 chunks of 10...")
    translations_b = []
    for i in range(0, 60, 10):
        chunk = doc_texts[i:i+10]
        chunk_trans = translate_batch_to_english(chunk, 'de', model_name='nllb-v2', cache=None)
        translations_b.extend(chunk_trans)
    print(f"  Result: {len(translations_b)} translations")
    
    # Test C: One document per call
    print("\nTest C: Translating one document per call...")
    translations_c = []
    for text in doc_texts:
        trans = translate_batch_to_english([text], 'de', model_name='nllb-v2', cache=None)
        translations_c.extend(trans)
    print(f"  Result: {len(translations_c)} translations")
    
    # Compare A vs B
    print("\nComparing A vs B...")
    ab_identical = sum(1 for a, b in zip(translations_a, translations_b) if a == b)
    ab_share = ab_identical / len(translations_a) * 100
    print(f"  Identical: {ab_identical}/{len(translations_a)} ({ab_share:.1f}%)")
    
    ab_diffs = []
    for i, (a, b) in enumerate(zip(translations_a, translations_b)):
        if a != b:
            ab_diffs.append({
                'id': doc_ids[i],
                'a': a[:200],
                'b': b[:200]
            })
    
    # Compare A vs C
    print("\nComparing A vs C...")
    ac_identical = sum(1 for a, c in zip(translations_a, translations_c) if a == c)
    ac_share = ac_identical / len(translations_a) * 100
    print(f"  Identical: {ac_identical}/{len(translations_a)} ({ac_share:.1f}%)")
    
    ac_diffs = []
    for i, (a, c) in enumerate(zip(translations_a, translations_c)):
        if a != c:
            ac_diffs.append({
                'id': doc_ids[i],
                'a': a[:200],
                'c': c[:200]
            })
    
    # Test duplicate handling: [d1, d2, d1]
    print("\nTest: Duplicate handling [d1, d2, d1]...")
    test_texts = [doc_texts[0], doc_texts[1], doc_texts[0]]
    dup_trans = translate_batch_to_english(test_texts, 'de', model_name='nllb-v2', cache=None)
    dup_match = (dup_trans[0] == dup_trans[2])
    print(f"  Position 0 == Position 2: {dup_match}")
    
    # Test empty string
    print("\nTest: Empty string...")
    empty_trans = translate_batch_to_english([""], 'de', model_name='nllb-v2', cache=None)
    empty_ok = (len(empty_trans) == 1 and empty_trans[0] == "")
    print(f"  Result: {repr(empty_trans[0])} (length={len(empty_trans)})")
    print(f"  Correct: {empty_ok}")
    
    # Test output length matches input length
    print("\nTest: Output length matches input length...")
    test_lens = [1, 5, 10, 25]
    len_ok = True
    for test_len in test_lens:
        test_input = doc_texts[:test_len]
        test_output = translate_batch_to_english(test_input, 'de', model_name='nllb-v2', cache=None)
        match = (len(test_output) == len(test_input))
        print(f"  Input={test_len}, Output={len(test_output)}: {match}")
        len_ok = len_ok and match
    
    # Print report
    print("\n" + "="*80)
    print("=== REPORT ===")
    print("="*80)
    
    print(f"\nConsistency Results:")
    print(f"  A vs B (chunks): {ab_identical}/{len(translations_a)} identical ({ab_share:.1f}%)")
    print(f"  A vs C (single): {ac_identical}/{len(translations_a)} identical ({ac_share:.1f}%)")
    
    if ab_diffs:
        print(f"\nA vs B Differences (showing up to 3):")
        for diff in ab_diffs[:3]:
            print(f"  ID: {diff['id']}")
            print(f"    A: {diff['a']}")
            print(f"    B: {diff['b']}")
    
    if ac_diffs:
        print(f"\nA vs C Differences (showing up to 3):")
        for diff in ac_diffs[:3]:
            print(f"  ID: {diff['id']}")
            print(f"    A: {diff['a']}")
            print(f"    C: {diff['c']}")
    
    print(f"\nEdge Cases:")
    print(f"  Duplicate handling [d1, d2, d1]: {dup_match} (positions 0 and 2 equal)")
    print(f"  Empty string handling: {empty_ok} (returns [''])")
    print(f"  Output length matches input: {len_ok}")
    
    print("\n" + "="*80)
    if ab_share == 100.0 and ac_share == 100.0 and dup_match and empty_ok and len_ok:
        print("✅ All consistency checks passed!")
    else:
        print("⚠️  Some consistency checks failed - see above")
    print("="*80)


if __name__ == '__main__':
    main()
