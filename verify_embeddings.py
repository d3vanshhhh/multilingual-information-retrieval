import faiss
from pathlib import Path
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

models = list(config['models'].keys())
languages = config['languages']

print("=" * 100)
print("EMBEDDING INDEX VERIFICATION")
print("=" * 100)

embeddings_dir = Path('embeddings')
all_good = True

for model in models:
    print(f"\n{model}")
    print("-" * 100)
    
    # Check direct embeddings (en, de, hi)
    for lang in languages:
        index_file = embeddings_dir / f"{model}_{lang}.index"
        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                dim = index.d
                count = index.ntotal
                size_mb = index_file.stat().st_size / (1024**2)
                print(f"  {lang:3s}: ✅ {count:4d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB")
            except Exception as e:
                print(f"  {lang:3s}: ❌ ERROR reading index: {e}")
                all_good = False
        else:
            print(f"  {lang:3s}: ❌ MISSING")
            all_good = False
    
    # Check translated embeddings (de->en, hi->en)
    for lang in ['de', 'hi']:
        index_file = embeddings_dir / f"{model}_{lang}_translated_en.index"
        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                dim = index.d
                count = index.ntotal
                size_mb = index_file.stat().st_size / (1024**2)
                print(f"  {lang}→en: ✅ {count:4d} vectors, dim={dim:4d}, size={size_mb:6.2f}MB")
            except Exception as e:
                print(f"  {lang}→en: ❌ ERROR reading index: {e}")
                all_good = False
        else:
            print(f"  {lang}→en: ❌ MISSING")
            all_good = False

print("\n" + "=" * 100)
if all_good:
    print("✅ ALL EMBEDDINGS ARE VALID AND COMPLETE!")
else:
    print("⚠️  SOME EMBEDDINGS HAVE ISSUES - SEE ABOVE")
print("=" * 100)

# Summary statistics
print("\nSUMMARY:")
total_indices = len(list(embeddings_dir.glob("*.index")))
total_size = sum(f.stat().st_size for f in embeddings_dir.glob("*.index")) / (1024**2)
print(f"Total index files: {total_indices}")
print(f"Total disk usage: {total_size:.2f} MB")
print(f"Expected files: {len(models) * 5} (6 models × 5 files each)")
