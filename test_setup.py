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
        'streamlit',
        'plotly',
        'sklearn',
        'scipy'
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
        'negatives',
        'training_data',
        'models'
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


def test_translation_cache():
    """Test translation cache functionality."""
    print("\nTesting translation cache...")
    try:
        from translation_cache import TranslationCache
        
        cache = TranslationCache(cache_dir='cache')
        
        # Test set and get
        cache.set("Hello", "en", "test", "Hola")
        result = cache.get("Hello", "en", "test")
        
        assert result == "Hola", "Cache get/set failed"
        print("  ✓ Cache set/get works")
        
        # Test stats
        cache.print_stats()
        
        print("\n✅ Translation cache works!")
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
        print("  1. python generate_embeddings.py --test")
        print("  2. python evaluate.py --test")
        print("  3. python -m streamlit run streamlit_demo_simple.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
