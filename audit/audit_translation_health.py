"""
Audit script for translation health (de and hi).

Validates translation quality by checking:
- Empty translations and identity translations
- Word ratios (expansion/compression)
- Word ratios by source length bucket (to detect truncation)
- Non-Latin character usage
- Repetition patterns (n-gram frequency)
"""
import argparse
import json
import yaml
import re
from pathlib import Path
from collections import Counter
from datasets import load_dataset


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_latin_script(text):
    """Check if character is in Latin script"""
    latin_count = 0
    alpha_count = 0
    
    for ch in text:
        if ch.isalpha():
            alpha_count += 1
            # Latin unicode ranges: Basic Latin (0000-007F), Latin-1 Supplement (0080-00FF),
            # Latin Extended-A (0100-017F), Latin Extended-B (0180-024F)
            if ord(ch) < 0x0250:
                latin_count += 1
    
    if alpha_count == 0:
        return 0.0
    
    return latin_count / alpha_count


def count_word_ngrams(text, n=4):
    """Count word n-grams in text"""
    words = text.split()
    if len(words) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def analyze_translation_file(language, file_type, source_dataset, translated_file, limit=None):
    """Analyze one translation file (corpus or queries)"""
    results = {
        'language': language,
        'file_type': file_type,
        'n': 0,
        'empty_translations': 0,
        'identical_to_source': 0,
        'word_ratios': [],
        'word_ratio_items': [],  # Store (item_id, ratio) pairs
        'ratios_by_bucket': {},
        'ratios_under_0_5_by_bucket': {},
        'non_latin_high': 0,
        'repetition_high': 0,
        'repetition_severe': 0,
        'examples': {
            'lowest_ratio': [],
            'identical': [],
            'highest_non_latin': [],
            'repetition': [],
            'repetition_severe': []
        }
    }
    
    print(f"\n  Analyzing {language} {file_type}...")
    
    # Load translated file
    with open(translated_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
    
    # Create lookup by _id
    translated_lookup = {item['_id']: item['text'] for item in translated_data}
    
    # Apply limit if specified
    if limit:
        source_dataset = source_dataset.select(range(min(limit, len(source_dataset))))
    
    results['n'] = len(source_dataset)
    
    # Buckets for length analysis
    buckets = {
        '<=100': [],
        '101-200': [],
        '201-300': [],
        '>300': []
    }
    
    for item in source_dataset:
        item_id = item['_id']
        source_text = item['text'].strip()
        translated_text = translated_lookup.get(item_id, '').strip()
        
        # Empty translation check
        if not translated_text:
            results['empty_translations'] += 1
            continue
        
        # Identical to source check
        if translated_text == source_text:
            results['identical_to_source'] += 1
            if len(results['examples']['identical']) < 3:
                results['examples']['identical'].append({
                    'id': item_id,
                    'source': source_text[:200],
                    'translation': translated_text[:200]
                })
        
        # Word ratio
        source_words = len(source_text.split())
        translated_words = len(translated_text.split())
        ratio = translated_words / max(1, source_words)
        results['word_ratios'].append(ratio)
        results['word_ratio_items'].append((item_id, ratio))
        
        # Categorize by source length bucket
        if source_words <= 100:
            bucket = '<=100'
        elif source_words <= 200:
            bucket = '101-200'
        elif source_words <= 300:
            bucket = '201-300'
        else:
            bucket = '>300'
        
        buckets[bucket].append(ratio)
        
        # Non-Latin check
        latin_ratio = is_latin_script(translated_text)
        if latin_ratio < 0.80:  # More than 20% non-Latin
            results['non_latin_high'] += 1
            if len(results['examples']['highest_non_latin']) < 3:
                results['examples']['highest_non_latin'].append({
                    'id': item_id,
                    'source': source_text[:200],
                    'translation': translated_text[:200],
                    'latin_ratio': latin_ratio
                })
        
        # Repetition check (only for translations with 40+ words)
        if translated_words >= 40:
            ngrams = count_word_ngrams(translated_text, n=4)
            if ngrams:
                max_freq = max(ngrams.values())
                if max_freq >= 5:
                    results['repetition_high'] += 1
                    if len(results['examples']['repetition']) < 3:
                        results['examples']['repetition'].append({
                            'id': item_id,
                            'source': source_text[:200],
                            'translation': translated_text[:200],
                            'max_4gram_freq': max_freq
                        })
                
                # Severe repetition check (>= 20 occurrences)
                if max_freq >= 20:
                    results['repetition_severe'] += 1
                    if len(results['examples']['repetition_severe']) < 3:
                        results['examples']['repetition_severe'].append({
                            'id': item_id,
                            'source': source_text[:200],
                            'translation': translated_text[:200],
                            'max_4gram_freq': max_freq
                        })
    
    # Compute ratio statistics
    if results['word_ratios']:
        sorted_ratios = sorted(results['word_ratios'])
        n_ratios = len(sorted_ratios)
        
        results['word_ratio_mean'] = sum(sorted_ratios) / n_ratios
        results['word_ratio_median'] = sorted_ratios[n_ratios // 2]
        results['word_ratio_p5'] = sorted_ratios[int(n_ratios * 0.05)]
        results['word_ratio_p95'] = sorted_ratios[int(n_ratios * 0.95)]
        results['word_ratio_under_0_5'] = sum(1 for r in sorted_ratios if r < 0.5)
        results['word_ratio_over_2_5'] = sum(1 for r in sorted_ratios if r > 2.5)
        
        # Save examples for lowest ratio using tracked (item_id, ratio) pairs
        sorted_items = sorted(results['word_ratio_items'], key=lambda x: x[1])[:3]
        for item_id, ratio in sorted_items:
            # Find the item in source dataset
            source_item = next((item for item in source_dataset if item['_id'] == item_id), None)
            if source_item:
                source_text = source_item['text'].strip()
                translated_text = translated_lookup[item_id].strip()
                results['examples']['lowest_ratio'].append({
                    'id': item_id,
                    'source': source_text[:200],
                    'translation': translated_text[:200],
                    'ratio': ratio
                })
        
        # Save examples for highest ratio (>2.5)
        results['examples']['highest_ratio'] = []
        high_ratio_items = [item for item in results['word_ratio_items'] if item[1] > 2.5]
        sorted_high_items = sorted(high_ratio_items, key=lambda x: x[1], reverse=True)[:3]
        for item_id, ratio in sorted_high_items:
            # Find the item in source dataset
            source_item = next((item for item in source_dataset if item['_id'] == item_id), None)
            if source_item:
                source_text = source_item['text'].strip()
                translated_text = translated_lookup[item_id].strip()
                results['examples']['highest_ratio'].append({
                    'id': item_id,
                    'source': source_text,
                    'translation': translated_text,
                    'ratio': ratio
                })
    
    # Compute bucket statistics
    for bucket_name, bucket_ratios in buckets.items():
        if bucket_ratios:
            sorted_bucket = sorted(bucket_ratios)
            under_0_5_count = sum(1 for r in sorted_bucket if r < 0.5)
            results['ratios_by_bucket'][bucket_name] = {
                'median': sorted_bucket[len(sorted_bucket) // 2],
                'count': len(sorted_bucket)
            }
            results['ratios_under_0_5_by_bucket'][bucket_name] = {
                'count': under_0_5_count,
                'share': 100.0 * under_0_5_count / len(sorted_bucket) if len(sorted_bucket) > 0 else 0.0
            }
        else:
            results['ratios_by_bucket'][bucket_name] = {
                'median': None,
                'count': 0
            }
            results['ratios_under_0_5_by_bucket'][bucket_name] = {
                'count': 0,
                'share': 0.0
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Audit translation health for de and hi")
    parser.add_argument('--limit', type=int, default=None,
                       help='Process only first N items per file (for testing)')
    parser.add_argument('--tag', type=str, default='nllb',
                       help='Translation tag to use in file names (default: nllb)')
    args = parser.parse_args()
    
    config = load_config()
    
    languages = ['de', 'hi']
    all_results = []
    
    for lang in languages:
        print(f"\nProcessing {lang}...")
        
        # Load source datasets
        corpus_dataset = load_dataset(config['corpus_dataset'], lang, split='test')
        queries_dataset = load_dataset(config['query_dataset'], lang, split='test')
        
        # Analyze corpus
        corpus_file = Path('translated_corpus') / f'corpus_{lang}_translated_en_{args.tag}.json'
        if corpus_file.exists():
            corpus_results = analyze_translation_file(
                lang, 'corpus', corpus_dataset, corpus_file, args.limit
            )
            all_results.append(corpus_results)
        else:
            print(f"  WARNING: {corpus_file} not found")
        
        # Analyze queries
        queries_file = Path('translated_queries') / f'queries_{lang}_translated_en_{args.tag}.json'
        if queries_file.exists():
            queries_results = analyze_translation_file(
                lang, 'queries', queries_dataset, queries_file, args.limit
            )
            all_results.append(queries_results)
        else:
            print(f"  WARNING: {queries_file} not found")
    
    # Print report
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("="*120)
    
    # Summary table
    print("\nSummary Statistics:")
    print(f"{'Lang':<4} {'Type':<7} {'N':<6} {'Empty':<6} {'Identical':<10} "
          f"{'RatioMean':<10} {'RatioMed':<10} {'P5':<7} {'P95':<7} "
          f"{'<0.5':<6} {'>2.5':<6} {'NonLatin':<9} {'Rep>=5':<7} {'RepSevere':<10}")
    
    for r in all_results:
        print(f"{r['language']:<4} {r['file_type']:<7} {r['n']:<6} "
              f"{r['empty_translations']:<6} {r['identical_to_source']:<10} "
              f"{r.get('word_ratio_mean', 0):<10.3f} {r.get('word_ratio_median', 0):<10.3f} "
              f"{r.get('word_ratio_p5', 0):<7.3f} {r.get('word_ratio_p95', 0):<7.3f} "
              f"{r.get('word_ratio_under_0_5', 0):<6} {r.get('word_ratio_over_2_5', 0):<6} "
              f"{r['non_latin_high']:<9} {r['repetition_high']:<7} {r['repetition_severe']:<10}")
    
    # Ratio by source length bucket
    print("\n" + "-"*120)
    print("Word Ratio by Source Length Bucket:")
    print(f"{'Lang':<4} {'Type':<7} {'<=100':<15} {'101-200':<15} {'201-300':<15} {'>300':<15}")
    
    for r in all_results:
        buckets = r['ratios_by_bucket']
        parts = []
        for bucket in ['<=100', '101-200', '201-300', '>300']:
            info = buckets[bucket]
            if info['median'] is not None:
                parts.append(f"{info['median']:.3f} (n={info['count']})")
            else:
                parts.append(f"N/A (n=0)")
        
        print(f"{r['language']:<4} {r['file_type']:<7} {parts[0]:<15} {parts[1]:<15} {parts[2]:<15} {parts[3]:<15}")
    
    print("\n" + "-"*120)
    print("Count and Share of Docs with Ratio < 0.5 by Source Length Bucket:")
    print(f"{'Lang':<4} {'Type':<7} {'<=100':<20} {'101-200':<20} {'201-300':<20} {'>300':<20}")
    
    for r in all_results:
        buckets = r['ratios_under_0_5_by_bucket']
        parts = []
        for bucket in ['<=100', '101-200', '201-300', '>300']:
            info = buckets[bucket]
            parts.append(f"{info['count']} ({info['share']:.1f}%)")
        
        print(f"{r['language']:<4} {r['file_type']:<7} {parts[0]:<20} {parts[1]:<20} {parts[2]:<20} {parts[3]:<20}")
    
    # Examples
    print("\n" + "="*120)
    print("EXAMPLES:")
    
    for r in all_results:
        lang = r['language']
        ftype = r['file_type']
        
        # Lowest ratio examples
        if r['examples']['lowest_ratio']:
            print(f"\n{lang} {ftype} - Lowest Word Ratio:")
            for ex in r['examples']['lowest_ratio']:
                print(f"  ID: {ex['id']}, Ratio: {ex['ratio']:.3f}")
                print(f"    Source: {ex['source']}")
                print(f"    Translation: {ex['translation']}")
        
        # Highest ratio examples
        if r['examples'].get('highest_ratio'):
            print(f"\n{lang} {ftype} - Highest Word Ratio:")
            for ex in r['examples']['highest_ratio']:
                print(f"  ID: {ex['id']}, Ratio: {ex['ratio']:.3f}")
                print(f"    Source: {ex['source'][:250]}")
                print(f"    Translation: {ex['translation'][:250]}")
        
        # Identical to source examples
        if r['examples']['identical']:
            print(f"\n{lang} {ftype} - Identical to Source:")
            for ex in r['examples']['identical']:
                print(f"  ID: {ex['id']}")
                print(f"    Source: {ex['source']}")
                print(f"    Translation: {ex['translation']}")
        
        # Highest non-Latin examples
        if r['examples']['highest_non_latin']:
            print(f"\n{lang} {ftype} - Highest Non-Latin Share:")
            for ex in r['examples']['highest_non_latin']:
                print(f"  ID: {ex['id']}, Latin ratio: {ex['latin_ratio']:.3f}")
                print(f"    Source: {ex['source']}")
                print(f"    Translation: {ex['translation']}")
        
        # Repetition examples
        if r['examples']['repetition']:
            print(f"\n{lang} {ftype} - High Repetition (>= 5):")
            for ex in r['examples']['repetition']:
                print(f"  ID: {ex['id']}, Max 4-gram freq: {ex['max_4gram_freq']}")
                print(f"    Source: {ex['source']}")
                print(f"    Translation: {ex['translation']}")
        
        # Severe repetition examples
        if r['examples']['repetition_severe']:
            print(f"\n{lang} {ftype} - Severe Repetition (>= 20):")
            for ex in r['examples']['repetition_severe']:
                print(f"  ID: {ex['id']}, Max 4-gram freq: {ex['max_4gram_freq']}")
                print(f"    Source: {ex['source']}")
                print(f"    Translation: {ex['translation']}")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    main()
