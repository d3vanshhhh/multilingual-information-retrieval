# Multilingual Information Retrieval

An investigation into multilingual information retrieval performance across embedding models, languages, and translation pipelines. Framework supports 9 embedding models across 16 languages; current evaluation uses 6 models and 3 languages (en, de, hi) with dual translation approaches.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Validate setup
python test_setup.py

# Generate embeddings (test mode - 5 minutes)
python generate_embeddings.py --test --model multilingual-e5-base
python generate_embeddings.py --test --model all-MiniLM-L6-v2

# Run evaluation
python evaluate.py --test --models multilingual-e5-base all-MiniLM-L6-v2

# Launch interactive interface
python -m streamlit run streamlit_demo_simple.py
```

**Interface URL:** http://localhost:8503

## 📊 Key Results

| Model | nDCG@10 (Direct) | nDCG@10 (Translate-NLLB) |
|-------|------------------|--------------------------|
| multilingual-e5-base | **0.830** | 0.776 |
| all-MiniLM-L6-v2 | 0.514 | 0.543 |
| bge-m3 | **0.836** | 0.812 |

**Top Performers:**
- **bge-m3** achieves highest nDCG@10 (0.836) with direct pipeline
- **multilingual-e5-base** close second (0.830) with direct pipeline
- Translation (NLLB) helps lower-performing models but slightly reduces performance for top models

## 📚 Documentation

For complete documentation, see **[DOCUMENTATION.md](DOCUMENTATION.md)**

Includes:
- Full project overview
- Technical architecture and design decisions
- Results analysis
- Usage guide
- Troubleshooting guide

## 📁 Project Structure

```
multilingual-ir-sem8/
├── config.yaml                  # Configuration
├── translation_cache.py         # Translation cache
├── utils.py                     # Translation utilities
├── generate_embeddings.py       # Generate FAISS indices
├── evaluate.py                  # Evaluation framework
├── streamlit_demo_simple.py     # Interactive search interface
├── compare_results.py           # Results analysis
├── test_setup.py                # Setup validation
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── DOCUMENTATION.md            # Complete documentation
```

## 🎓 Key Features

- ✅ Framework supports 9 embedding models (currently evaluated: 6 models)
- ✅ Framework supports 16 languages (currently configured: 3 — en, de, hi)
- ✅ 2 retrieval pipelines (Direct vs Translate→English with NLLB/M2M100)
- ✅ Translation caching with SHA256 hashing
- ✅ Comprehensive metrics (nDCG, MRR, Precision, Success)
- ✅ Interactive search interface
- ✅ Per-language and summary results
- ✅ Visualizations (HTML charts)

## ⚡ Quick Setup Checklist

**Before running:**
1. Install dependencies: `pip install -r requirements.txt`
2. Verify setup: `python test_setup.py`
3. Check if embeddings exist: `ls embeddings/` (Windows: `dir embeddings`)
4. Check if results exist: `ls results/` (Windows: `dir results`)

**To regenerate everything:**
```bash
python generate_embeddings.py --test --model multilingual-e5-base
python evaluate.py --test --models multilingual-e5-base
python -m streamlit run streamlit_demo_simple.py
```

## 🔧 Configuration

Edit `config.yaml` to customize models, languages, and evaluation parameters.

## 🐛 Troubleshooting

**Streamlit not found:**
```bash
python -m streamlit run streamlit_demo_simple.py
```

**Out of memory:**
```bash
python evaluate.py --test  # Use test mode
```

**Index not found:**
```bash
python generate_embeddings.py --test --model MODEL_NAME
```

For more troubleshooting, see [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting)

## 📈 Project Stats

- **Models**: 6 configured (framework supports 9)
- **Languages**: 3 configured (framework supports 16)
- **Queries**: 56 per language (filtered for test mode)
- **Documents**: 500 per language (test mode)
- **Performance**: bge-m3 and multilingual-e5-base top performers (~0.83 nDCG@10)

## 🔬 Research Evolution

This investigation extends an earlier multilingual IR study with:

- **Extended model coverage**: 5 → 9 embedding models
- **Complete evaluation**: Partial results → full metrics across all models
- **Dual translation support**: NLLB-200 + M2M100
- **Persistent translation cache**: SHA256-based with disk storage
- **Interactive exploration interface**: Real-time search across languages
- **Comprehensive analysis**: Per-language breakdowns, visualizations

## 📞 Support

For issues or questions:
1. Check [DOCUMENTATION.md](DOCUMENTATION.md)
2. Run `python test_setup.py`
3. Review error messages (they're designed to be helpful)

---

**Last Updated:** July 27, 2026
