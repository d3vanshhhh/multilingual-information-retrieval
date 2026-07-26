import streamlit as st
import yaml
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from translation_cache import TranslationCache
from utils import translate_to_english


@st.cache_resource
def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(model_name, config):
    """Load embedding model."""
    try:
        model_id = config['models'][model_name]
        return SentenceTransformer(model_id)
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {e}")
        return None


@st.cache_resource
def load_translation_cache():
    """Load translation cache."""
    return TranslationCache()


@st.cache_data
def load_language_data(language):
    """Load FAISS index and corpus for a language."""
    try:
        config = load_config()
        
        # Load corpus
        corpus_dataset = load_dataset(
            config['corpus_dataset'],
            language,
            split='test'
        )
        
        # Get first 500 docs for demo
        if len(corpus_dataset) > 500:
            corpus_dataset = corpus_dataset.select(range(500))
        
        docs = []
        for doc in corpus_dataset:
            docs.append({
                'id': doc['_id'],
                'title': doc.get('title', 'No title'),
                'text': doc['text'][:300] + '...' if len(doc['text']) > 300 else doc['text']
            })
        
        return docs
    except Exception as e:
        st.error(f"Failed to load data for {language}: {e}")
        return []


def search_documents(query_text, model, docs, language, top_k=10):
    """Search documents using the model."""
    # Encode query
    query_emb = model.encode(query_text, convert_to_numpy=True)
    
    # Encode all documents
    doc_texts = [doc['text'] for doc in docs]
    doc_embs = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Normalize
    query_emb = query_emb / np.linalg.norm(query_emb)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(doc_embs, query_emb)
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'title': docs[idx]['title'],
            'text': docs[idx]['text'],
            'similarity': float(similarities[idx]),
            'language': language
        })
    
    return results


def main():
    st.set_page_config(page_title="Multilingual IR Demo", layout="wide", page_icon="🌍")
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🌍 Multilingual Information Retrieval System")
    st.markdown("**Semester VIII B.Tech Project Extension**")
    st.markdown("---")
    
    config = load_config()
    cache = load_translation_cache()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selector
    available_models = [
        'multilingual-e5-base',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2'
    ]
    model_name = st.sidebar.selectbox(
        "🤖 Select Model",
        available_models,
        index=0,
        help="Multilingual models: e5-base, paraphrase-multilingual | Monolingual: MiniLM, mpnet"
    )
    
    # Pipeline selector
    pipeline = st.sidebar.radio(
        "🔄 Retrieval Pipeline",
        ["Direct", "Translate→English"],
        index=0,
        help="Direct: Search in original language | Translate: Translate to English first"
    )
    
    # Translation model selector (only show if Translate pipeline is selected)
    translation_model = "nllb"
    if pipeline == "Translate→English":
        translation_model = st.sidebar.selectbox(
            "🌐 Translation Model",
            ["nllb", "m2m100"],
            index=0,
            help="NLLB: Faster, 600M params | M2M100: Slower, 1.9GB"
        )
    
    # Top-k slider
    top_k = st.sidebar.slider(
        "📊 Number of Results",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Language selector
    query_language = st.sidebar.selectbox(
        "🌐 Query Language",
        config['languages'],
        index=config['languages'].index('en')
    )
    
    # Info boxes
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Info")
    st.sidebar.info(f"""
    **Embedding Models:** {len(config['models'])}  
    **Languages Supported:** {len(config['languages'])}  
    **Translation Models:** NLLB-200, M2M100
    """)
    
    st.sidebar.markdown("### 💾 Cache Stats")
    st.sidebar.success(f"""
    **Cache Hits:** {cache.hits}  
    **Cache Misses:** {cache.misses}  
    **Total Cached:** {len(cache.cache)}
    """)
    
    # Main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_text = st.text_input(
            "🔍 Enter your search query",
            placeholder="Type your query here...",
            help="Enter a query in any supported language"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("💡 Example Queries"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**English:**")
            st.code("machine learning algorithms")
            st.code("artificial intelligence")
        with col2:
            st.markdown("**German:**")
            st.code("maschinelles Lernen")
            st.code("künstliche Intelligenz")
        with col3:
            st.markdown("**Hindi:**")
            st.code("कंप्यूटर विज्ञान")
            st.code("मशीन लर्निंग")
    
    if search_button and query_text:
        # Load model
        with st.spinner(f"Loading model {model_name}..."):
            model = load_model(model_name, config)
        
        if model is None:
            st.error("Failed to load model. Please try again.")
            return
        
        # Process query based on pipeline
        processed_query = query_text
        translation_info = None
        
        if pipeline == "Translate→English" and query_language != 'en':
            with st.spinner("🔄 Translating query..."):
                processed_query = translate_to_english(
                    query_text,
                    query_language,
                    model_name=translation_model,
                    cache=cache
                )
                translation_info = {
                    'original': query_text,
                    'translated': processed_query,
                    'from': query_language,
                    'to': 'en',
                    'model': translation_model
                }
        
        # Show translation if applicable
        if translation_info:
            st.info(f"**Original ({translation_info['from']}):** {translation_info['original']}")
            st.success(f"**Translated ({translation_info['to']}) using {translation_info['model'].upper()}:** {translation_info['translated']}")
        
        # Search across all languages
        st.markdown("---")
        st.subheader("📄 Search Results")
        
        all_results = []
        
        # Create tabs for each language
        tabs = st.tabs([f"🌐 {lang.upper()}" for lang in config['languages']])
        
        for idx, lang in enumerate(config['languages']):
            with tabs[idx]:
                with st.spinner(f"Searching {lang.upper()} documents..."):
                    # Load documents
                    docs = load_language_data(lang)
                    
                    if not docs:
                        st.warning(f"No documents available for {lang}")
                        continue
                    
                    # Search
                    results = search_documents(processed_query, model, docs, lang, top_k)
                    
                    # Display results
                    for result in results:
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric("Rank", f"#{result['rank']}")
                                st.metric("Score", f"{result['similarity']:.3f}")
                            with col2:
                                st.markdown(f"**{result['title']}**")
                                st.markdown(result['text'])
                            st.markdown("---")
        
        # Summary
        st.markdown("---")
        st.subheader("📊 Search Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_name)
        with col2:
            st.metric("Pipeline", pipeline)
        with col3:
            st.metric("Query Language", query_language.upper())
        with col4:
            st.metric("Results per Language", top_k)
    
    elif search_button:
        st.warning("⚠️ Please enter a query to search")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Multilingual Information Retrieval System | Semester VIII B.Tech Project</p>
    <p>Supports 9 models across 16 languages with dual translation pipelines</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
