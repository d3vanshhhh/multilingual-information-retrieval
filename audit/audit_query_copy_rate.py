"""
Audit script for query translation copy rate.

Analyzes how often translated queries copy the first word from the source,
particularly for common German question words like "Was".
"""
import json
import yaml
import re
from collections import Counter
from pathlib import Path
from datasets import load_dataset


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def strip_punctuation_and_lower(word):
    """Remove punctuation and convert to lowercase"""
    # Remove common punctuation
    word = re.sub(r'^[^\w]+|[^\w]+$', '', word, flags=re.UNICODE)
    return word.lower()


def get_first_word(text):
    """Get first word from text, stripped and lowercased"""
    words = text.split()
    if not words:
        return ''
    return strip_punctuation_and_lower(words[0])


def is_latin_script(text):
    """Check if text contains only Latin alphabetic characters"""
    for ch in text:
        if ch.isalpha():
            # Non-Latin unicode ranges
            if ord(ch) >= 0x0250:
                return False
    return True


def has_non_latin(text):
    """Check if text contains any non-Latin alphabetic character"""
    for ch in text:
        if ch.isalpha():
            # Non-Latin unicode ranges
            if ord(ch) >= 0x0250:
                return True
    return False


def analyze_language(language, config):
    """Analyze query copy rate for one language"""
    results = {
        'language': language,
        'n': 0,
        'first_word_counts': Counter(),
        'first_word_copy_counts': Counter(),
        'was_queries': 0,
        'was_copied': 0,
        'same_first_word': 0,
        'non_latin_translations': 0
    }
    
    print(f"\nProcessing {language}...")
    
    # Load source queries
    queries_dataset = load_dataset(config['query_dataset'], language, split='test')
    
    # Load translated queries
    translated_file = Path('translated_queries') / f'queries_{language}_translated_en_nllb.json'
    if not translated_file.exists():
        print(f"  ERROR: {translated_file} not found")
        return results
    
    with open(translated_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
    
    # Create lookup by _id
    translated_lookup = {item['_id']: item['text'] for item in translated_data}
    
    results['n'] = len(queries_dataset)
    
    # Analyze each query
    for query in queries_dataset:
        query_id = query['_id']
        source_text = query['text'].strip()
        translated_text = translated_lookup.get(query_id, '').strip()
        
        if not translated_text:
            continue
        
        # Get first words
        source_first = get_first_word(source_text)
        translated_first = get_first_word(translated_text)
        
        # Count first word occurrences
        if source_first:
            results['first_word_counts'][source_first] += 1
            
            # Check if translation starts with same first word
            if source_first == translated_first:
                results['first_word_copy_counts'][source_first] += 1
                results['same_first_word'] += 1
        
        # For German: check "Was " specifically (with space to ensure it's the first word)
        if language == 'de' and source_text.startswith('Was '):
            results['was_queries'] += 1
            if translated_text.startswith('Was '):
                results['was_copied'] += 1
        
        # For Hindi: check non-Latin characters in translation
        if language == 'hi' and has_non_latin(translated_text):
            results['non_latin_translations'] += 1
    
    return results


def main():
    config = load_config()
    
    languages = ['de', 'hi']
    all_results = []
    
    for lang in languages:
        try:
            results = analyze_language(lang, config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR processing {lang}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print report
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("="*120)
    
    for r in all_results:
        lang = r['language']
        n = r['n']
        
        print(f"\n{lang.upper()} Query Copy Rate Analysis (n={n}):")
        print("-" * 120)
        
        # Top 12 most frequent first words
        print(f"\nTop 12 Most Frequent First Words (source) and Copy Rate to Translation:")
        print(f"{'First Word':<20} {'Count':<10} {'Copied':<10} {'Copy Rate %':<15}")
        
        most_common = r['first_word_counts'].most_common(12)
        for word, count in most_common:
            copied = r['first_word_copy_counts'].get(word, 0)
            copy_rate = 100.0 * copied / count if count > 0 else 0.0
            print(f"{word:<20} {count:<10} {copied:<10} {copy_rate:<15.2f}")
        
        # For German: "Was " specific analysis
        if lang == 'de':
            print(f"\nGerman 'Was ' Analysis:")
            print(f"  Queries starting with 'Was ': {r['was_queries']}")
            print(f"  Translations also starting with 'Was ': {r['was_copied']}")
            if r['was_queries'] > 0:
                was_rate = 100.0 * r['was_copied'] / r['was_queries']
                print(f"  Copy rate: {was_rate:.2f}%")
        
        # Overall same-first-word rate
        print(f"\nOverall Same First Word:")
        print(f"  Queries with same first word in translation: {r['same_first_word']} / {n}")
        if n > 0:
            same_rate = 100.0 * r['same_first_word'] / n
            print(f"  Rate: {same_rate:.2f}%")
        
        # For Hindi: non-Latin check
        if lang == 'hi':
            print(f"\nHindi Non-Latin Characters in Translations:")
            print(f"  Translations with non-Latin characters: {r['non_latin_translations']} / {n}")
            if n > 0:
                non_latin_rate = 100.0 * r['non_latin_translations'] / n
                print(f"  Rate: {non_latin_rate:.2f}%")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    main()
