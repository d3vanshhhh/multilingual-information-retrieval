"""
Translation cache with SHA256-based key generation and disk persistence.

Main class:
    TranslationCache: Persistent cache for translation results with auto-save.

Usage:
    cache = TranslationCache(cache_dir='cache')
    result = cache.get(text, src_lang, model) or translate(text)
    cache.set(text, src_lang, model, result)
"""
import json
import hashlib
import os
from pathlib import Path


class TranslationCache:
    """Translation cache with disk persistence and automatic saving."""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.new_translations = 0
        
        self.load()
    
    def _make_key(self, text, src_lang, model):
        """Generate SHA256 hash key for cache lookup."""
        key_string = f"{model}::{src_lang}::{text}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(self, text, src_lang, model):
        """Retrieve translation from cache if it exists."""
        key = self._make_key(text, src_lang, model)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text, src_lang, model, translation):
        """Store translation in cache."""
        key = self._make_key(text, src_lang, model)
        self.cache[key] = translation
        self.new_translations += 1
        
        # Auto-save every 50 new translations
        if self.new_translations % 50 == 0:
            self.save()
    
    def save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"[Cache] Saved {len(self.cache)} translations to {self.cache_file}")
        except Exception as e:
            print(f"[Cache] Warning: Failed to save cache: {e}")
    
    def load(self):
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"[Cache] Loaded {len(self.cache)} translations from {self.cache_file}")
            except Exception as e:
                print(f"[Cache] Warning: Failed to load cache: {e}")
                self.cache = {}
        else:
            print("[Cache] No existing cache found, starting fresh")
    
    def print_stats(self):
        """Print cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print(f"\n[Cache Stats]")
        print(f"  Total lookups: {total}")
        print(f"  Hits: {self.hits} ({hit_rate:.1f}%)")
        print(f"  Misses: {self.misses}")
        print(f"  New translations: {self.new_translations}")
        print(f"  Total cached: {len(self.cache)}")
