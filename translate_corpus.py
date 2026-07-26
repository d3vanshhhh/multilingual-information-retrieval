import argparse
import yaml
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from translation_cache import TranslationCache
from utils import translate_to_english


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def translate_corpus_for_language(language, config, translation_model, cache, test_mode=False):
    if language == 'en':
        print(f"[{language}] Skipping - already English")
        return None
    
    print(f"\nTranslating corpus: {language} -> English")
    print(f"Translation model: {translation_model}")
    
    # Load corpus
    print(f"[{language}] Loading corpus...")
    corpus_dataset = load_dataset(
        config['corpus_dataset'],
        language,
        split='test'
    )
    
    # Limit in test mode
    max_docs = config['evaluation']['max_docs_per_lang'] if test_mode else None
    if max_docs and len(corpus_dataset) > max_docs:
        corpus_dataset = corpus_dataset.select(range(max_docs))
    
    print(f"[{language}] Translating {len(corpus_dataset)} documents...")
    
    # Translate each document
    translated_docs = []
    for doc in tqdm(corpus_dataset, desc=f"Translating {language}"):
        original_text = doc['text']
        translated_text = translate_to_english(
            original_text,
            language,
            model_name=translation_model,
            cache=cache
        )
        
        translated_doc = {
            '_id': doc['_id'],
            'title': doc.get('title', ''),
            'text': translated_text,
            'original_language': language
        }
        translated_docs.append(translated_doc)
    
    return translated_docs


def save_translated_corpus(translated_docs, language, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"corpus_{language}_translated_en.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_docs, f, ensure_ascii=False, indent=2)
    
    print(f"[{language}] Saved {len(translated_docs)} translated documents to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Translate corpus to English")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', default='translated_corpus', help='Directory to save translated corpus')
    parser.add_argument('--translation-model', default='m2m100', choices=['nllb', 'm2m100'], 
                        help='Translation model to use')
    parser.add_argument('--test', action='store_true', help='Test mode: use limited data')
    parser.add_argument('--language', help='Translate specific language only')
    args = parser.parse_args()
    
    config = load_config(args.config)
    cache = TranslationCache()
    
    if args.language:
        languages = [args.language]
    else:
        languages = [lang for lang in config['languages'] if lang != 'en']
    
    print(f"Starting corpus translation for {len(languages)} language(s)")
    print(f"Translation model: {args.translation_model}")
    print(f"Test mode: {args.test}")
    
    for lang in languages:
        try:
            translated_docs = translate_corpus_for_language(
                lang,
                config,
                args.translation_model,
                cache,
                test_mode=args.test
            )
            
            if translated_docs:
                save_translated_corpus(translated_docs, lang, args.output_dir)
        
        except Exception as e:
            print(f"ERROR translating {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nTRANSLATION COMPLETE")
    print(f"Cache stats: {cache.hits} hits, {cache.misses} misses, {len(cache.cache)} total")


if __name__ == "__main__":
    main()
