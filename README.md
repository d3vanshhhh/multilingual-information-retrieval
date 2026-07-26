# Multilingual Information Retrieval - Semester VIII

A comprehensive multilingual information retrieval system. Framework supports 9 embedding models across 16 languages; current demo configuration uses 6 models and 3 languages (en, de, hi) with dual translation pipelines.

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

# Launch demo
python -m streamlit run streamlit_demo_simple.py
```

**Demo URL:** http://localhost:8503

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
- Demo guide for presentations
- Technical details and architecture
- Results analysis
- Troubleshooting guide
- Q&A for expected questions

## 🎯 For Mid-Sem Demo

See **[DEMO_GUIDE.md](DEMO_GUIDE.md)** for:
- 20-minute presentation flow
- Demo queries (copy-paste ready)
- Key talking points
- Expected questions & answers

## 📁 Project Structure

```
multilingual-ir-sem8/
├── config.yaml                  # Configuration
├── translation_cache.py         # Translation cache
├── utils.py                     # Translation utilities
├── generate_embeddings.py       # Generate FAISS indices
├── evaluate.py                  # Evaluation framework
├── streamlit_demo_simple.py     # Interactive demo
├── compare_results.py           # Results analysis
├── test_setup.py                # Setup validation
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── DOCUMENTATION.md            # Complete documentation
├── DEMO_GUIDE.md                # Demo guide
```

## 🎓 Key Features

- ✅ Framework supports 9 embedding models (currently configured: 6 models)
- ✅ Framework supports 16 languages (demo configured for 3: en, de, hi)
- ✅ 2 retrieval pipelines (Direct vs Translate→English with NLLB/M2M100)
- ✅ Translation caching with SHA256 hashing
- ✅ Comprehensive metrics (nDCG, MRR, Precision, Success)
- ✅ Interactive Streamlit demo
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

## 🎉 Semester VII → VIII Improvements

| Feature | Sem VII | Sem VIII |
|---------|---------|----------|
| Models | 5 | 9 |
| Evaluation | Partial | Complete |
| Translation | NLLB only | NLLB + M2M100 |
| Cache | None | Persistent |
| Demo | None | Interactive |
| Results | Basic | Comprehensive |

## 📞 Support

For issues or questions:
1. Check [DOCUMENTATION.md](DOCUMENTATION.md)
2. Run `python test_setup.py`
3. Review error messages (they're designed to be helpful)

---

**B.Tech Semester VIII Project Extension**  
**Last Updated:** July 26, 2026 (Cleanup Pass)
