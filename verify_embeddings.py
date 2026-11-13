"""
Verify that all embedding indices exist, are readable, and have correct document counts.
Checks both native embeddings and translated embeddings.
"""

import faiss
from pathlib import Path
import yaml
import json
from datasets import load_dataset
from utils import TRANSLATION_TAG

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

models = list(config['models'].keys())
languages = config['languages']

print("=" * 100)
print("EMBEDDING INDEX VERIFICATION")
print(f"Translation tag: {TRANSLATION_TAG}")
print("=" * 100)

embeddings_dir = Path('embeddings')
translated_corpus_dir = Path('translated_corpus')
all_good = True
checked_count = 0

# Get translated languages
translated_languages = [lang for lang in languages if lang != 'en']
expected_count = len(models) * (len(languages) + len(translated_languages))

# Get actual corpus sizes for each language
print("\nLoading corpus sizes from datasets...")
corpus_sizes = {}
for lang in languages:
    corpus = load_dataset(config['corpus_dataset'], lang, split='test')
    corpus_sizes[lang] = len(corpus)
    print(f"  {lang}: {len(corpus)} documents")

print()

for model in models:
    print(f"\n{model}")
    print("-" * 100)
    
    # Check native embeddings
    for lang in languages:
        index_file = embeddings_dir / f"{model}_{lang}.index"
        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                dim = index.d
                count = index.ntotal
                size_mb = index_file.stat().st_size / (1024**2)
                expected_count_lang = corpus_sizes[lang]
                
                if count == expected_count_lang:
                    print(f"  {lang:3s}: ✅ {count:5d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB (matches corpus)")
                    checked_count += 1
                else:
                    print(f"  {lang:3s}: ❌ {count:5d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB (expected {expected_count_lang})")
                    all_good = False
                    checked_count += 1
            except Exception as e:
                print(f"  {lang:3s}: ❌ ERROR reading index: {e}")
                all_good = False
        else:
            print(f"  {lang:3s}: ❌ MISSING")
            all_good = False
    
    # Check translated embeddings (skip English)
    for lang in translated_languages:
        index_file = embeddings_dir / f"{model}_{lang}_translated_en_{TRANSLATION_TAG}.index"
        translated_corpus_file = translated_corpus_dir / f"corpus_{lang}_translated_en_{TRANSLATION_TAG}.json"
        
        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                dim = index.d
                count = index.ntotal
                size_mb = index_file.stat().st_size / (1024**2)
                
                # Get expected count from translated corpus file
                if translated_corpus_file.exists():
                    with open(translated_corpus_file, 'r', encoding='utf-8') as f:
                        translated_corpus = json.load(f)
                    expected_translated = len(translated_corpus)
                    
                    if count == expected_translated:
                        print(f"  {lang}→en ({TRANSLATION_TAG}): ✅ {count:5d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB (matches translated corpus)")
                        checked_count += 1
                    else:
                        print(f"  {lang}→en ({TRANSLATION_TAG}): ❌ {count:5d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB (expected {expected_translated})")
                        all_good = False
                        checked_count += 1
                else:
                    print(f"  {lang}→en ({TRANSLATION_TAG}): ⚠️  {count:5d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB (translated corpus file missing, cannot verify count)")
                    checked_count += 1
            except Exception as e:
                print(f"  {lang}→en ({TRANSLATION_TAG}): ❌ ERROR reading index: {e}")
                all_good = False
        else:
            print(f"  {lang}→en ({TRANSLATION_TAG}): ❌ MISSING")
            all_good = False

print("\n" + "=" * 100)
if all_good:
    print("✅ ALL EMBEDDINGS ARE VALID AND COMPLETE!")
else:
    print("⚠️  SOME EMBEDDINGS HAVE ISSUES - SEE ABOVE")
print("=" * 100)

# Summary statistics
translated_lang_count = len(translated_languages)
print("\nSUMMARY:")
total_indices = len(list(embeddings_dir.glob("*.index")))
total_size = sum(f.stat().st_size for f in embeddings_dir.glob("*.index")) / (1024**2)
print(f"Total index files found: {total_indices}")
print(f"Total indices checked: {checked_count}")
print(f"Expected indices: {expected_count} ({len(models)} models × {len(languages) + translated_lang_count} = {len(models) * len(languages)} native + {len(models) * translated_lang_count} translated)")
print(f"Total disk usage: {total_size:.2f} MB")

if checked_count == expected_count and all_good:
    print("\n✅ Perfect! All expected indices present and verified.")
elif checked_count < expected_count:
    print(f"\n⚠️  Missing {expected_count - checked_count} indices!")
elif not all_good:
    print("\n⚠️  Some indices have count mismatches or errors!")
