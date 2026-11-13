"""
Quick test script to validate the setup and dependencies.
Run this before starting the full pipeline.
"""

import sys
import importlib


def check_imports():
    """Check if all required packages are installed."""
    required_packages = [
        'yaml',
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'sentence_transformers',
        'datasets',
        'faiss',
        'tqdm',
        'plotly'
    ]
    
    print("Checking required packages...")
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True


def check_config():
    """Check if config.yaml is valid."""
    print("\nChecking config.yaml...")
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'models' in config, "Missing 'models' section"
        assert 'languages' in config, "Missing 'languages' section"
        assert 'evaluation' in config, "Missing 'evaluation' section"
        
        print(f"  ✓ Found {len(config['models'])} models")
        print(f"  ✓ Found {len(config['languages'])} languages")
        print(f"  ✓ K values: {config['evaluation']['k_values']}")
        print("\n✅ Config is valid!")
        return True
    except Exception as e:
        print(f"\n❌ Config error: {e}")
        return False


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    from pathlib import Path
    
    required_dirs = [
        'embeddings',
        'results',
        'results/per_language',
        'cache',
        'translated_corpus',
        'translated_queries',
        'logs',
        'audit'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            path.mkdir(parents=True, exist_ok=True)
            print(f"    Created {dir_path}")
    
    print("\n✅ All directories ready!")
    return True


def check_gpu():
    """Check GPU availability for translation and embeddings."""
    print("\nChecking GPU availability...")
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {device_name}")
            print("\n✅ GPU acceleration enabled!")
            return True
        else:
            print(f"  ⚠️  CUDA not available - will use CPU")
            print(f"\n⚠️  Warning: Translation and embedding generation will be significantly slower on CPU.")
            print(f"     This project found CPU-only torch install to be a common pitfall.")
            print(f"     To fix: pip install torch with CUDA support")
            print(f"     Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return True  # Still pass, but with warning
    except Exception as e:
        print(f"\n❌ GPU check error: {e}")
        return False


def test_translation_cache():
    """Test translation cache functionality."""
    print("\nTesting translation cache...")
    try:
        from translation_cache import TranslationCache
        from pathlib import Path
        from utils import TRANSLATION_TAG
        
        # Count entries in nllb-v2 cache files only
        cache_dir = Path('cache')
        cache_pattern = f'translation_cache_{TRANSLATION_TAG}_*.json'
        cache_files = list(cache_dir.glob(cache_pattern))
        
        if not cache_files:
            print(f"  ⚠️  No {TRANSLATION_TAG} cache files found in {cache_dir}")
            print(f"     This is normal for first run - will be created during translation")
            return True
        
        total_entries = 0
        for cache_file in cache_files:
            try:
                # Extract language from filename
                cache_name = cache_file.stem.replace('translation_cache_', '')
                cache = TranslationCache(name=cache_name)
                entry_count = len(cache.cache)
                total_entries += entry_count
                print(f"  ✓ {cache_file.name}: {entry_count} entries")
            except Exception as e:
                print(f"  ⚠️  Could not load {cache_file.name}: {e}")
        
        print(f"  ✓ Total cache entries: {total_entries}")
        print("\n✅ Translation cache ready!")
        return True
    except Exception as e:
        print(f"\n❌ Translation cache error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Multilingual IR - Setup Validation")
    print("="*60)
    
    results = []
    
    results.append(("Package Check", check_imports()))
    results.append(("Config Check", check_config()))
    results.append(("Directory Check", check_directories()))
    results.append(("GPU Check", check_gpu()))
    results.append(("Cache Test", test_translation_cache()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Translate corpus: python translate_corpus.py --language de --translation-model nllb-v2")
        print("                       python translate_corpus.py --language hi --translation-model nllb-v2")
        print("  2. Translate queries: python translate_queries.py --language de --translation-model nllb-v2")
        print("                        python translate_queries.py --language hi --translation-model nllb-v2")
        print("  3. Generate embeddings (native): python generate_embeddings.py")
        print("  4. Generate embeddings (translated): python generate_embeddings_translated.py")
        print("  5. Evaluate: python evaluate.py")
        print("  6. Analyze significance: python significance_analysis.py")
        print("  7. Compare results: python compare_results.py --no-plots")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
