import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
    /* Ensure Streamlit metrics don't introduce extra white bars/gaps */
    div[data-testid="stMetric"] {
        background: transparent;
        padding: 0;
        margin: 0 0 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        with open('model_svm.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure model_svm.pkl and tfidf_vectorizer.pkl files are in the same directory as this app.")
        return None, None

def clean_text(text):
    """Clean and preprocess text"""
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags and special characters
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of the given text"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform text using TF-IDF
    text_vector = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.decision_function(text_vector)[0]
    
    # Convert probability to confidence score
    confidence = abs(probability)
    
    return prediction, confidence, cleaned_text

def main():
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Header Section
    st.markdown('<h1 class="main-header">🎬 Analisis Sentimen Film IMDB</h1>', unsafe_allow_html=True)
    st.markdown("### Sistem analisis sentimen untuk mengklasifikasikan ulasan film menjadi positif atau negatif menggunakan Machine Learning")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 📊 Informasi Model")
    st.sidebar.info("""
    **Model**: Support Vector Machine (SVM)  
    **Akurasi**: 86.89%  
    **Fitur**: TF-IDF Vectorization  
    **Data Latih**: Ulasan Film IMDB
    """)
    
    st.sidebar.markdown("## 🎯 Cara Penggunaan")
    st.sidebar.markdown("""
    1. Masukkan ulasan film di area input
    2. Klik tombol 'Analisis Sentimen'
    3. Lihat hasil prediksi dan tingkat kepercayaan
    4. Coba ulasan yang berbeda untuk menguji model
    """)
    
    st.sidebar.markdown("## 📝 Contoh Ulasan")
    sample_reviews = [
        "Film ini sangat fantastis! Akting yang bagus dan plot yang menakjubkan.",
        "Film yang buruk, membuang waktu. Membosankan dan dibuat dengan buruk.",
        "Film ini biasa saja, tidak ada yang istimewa tapi juga tidak buruk.",
        "Penampilan luar biasa dari para pemain. Sangat direkomendasikan!",
        "Sekuel yang mengecewakan. Yang asli jauh lebih baik."
    ]
    
    for i, sample in enumerate(sample_reviews):
        if st.sidebar.button(f"Contoh {i+1}", key=f"sample_{i}"):
            st.session_state.sample_text = sample
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input Area
        st.markdown('<div class="sub-header">📝 Area Input</div>', unsafe_allow_html=True)
        st.markdown("**Masukkan atau tempel teks ulasan film untuk dianalisis:**")
        
        # Text input
        review_text = st.text_area(
            "Ulasan Film:",
            height=200,
            placeholder="Masukkan ulasan film Anda di sini...",
            help="Ketik atau tempel ulasan film untuk menganalisis sentimennya",
            value=st.session_state.get('sample_text', '')
        )
        
        # Clear sample text after use
        if hasattr(st.session_state, 'sample_text'):
            del st.session_state.sample_text
        
        # Analysis Button
        if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Menganalisis sentimen..."):
                    prediction, confidence, cleaned_text = predict_sentiment(review_text, model, vectorizer)
                
                # Output Area
                st.markdown('<div class="sub-header">📊 Area Output</div>', unsafe_allow_html=True)
                
                # Prediction result
                col_pred, col_conf = st.columns(2)
                
                with col_pred:
                    if prediction == 1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="positive">😊 SENTIMEN POSITIF</h3>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="negative">😞 SENTIMEN NEGATIF</h3>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col_conf:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Tingkat Kepercayaan", f"{confidence:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show cleaned text
                with st.expander("🔍 Lihat Teks yang Diproses"):
                    st.text(cleaned_text)
                
                # Additional insights
                st.markdown("### 💡 Wawasan")
                if confidence > 0.5:
                    st.success("Prediksi dengan kepercayaan tinggi - model sangat yakin dengan sentimen ini!")
                elif confidence > 0.3:
                    st.warning("Prediksi dengan kepercayaan sedang - sentimen cukup jelas.")
                else:
                    st.info("Prediksi dengan kepercayaan rendah - sentimen mungkin ambigu.")
                
            else:
                st.warning("Silakan masukkan ulasan film untuk dianalisis.")
    
    with col2:
        st.markdown('<div class="sub-header">📈 Performa Model</div>', unsafe_allow_html=True)
        
        # Performance metrics
        st.metric("Akurasi", "86.89%")
        st.metric("Presisi", "85.0%")
        st.metric("Recall", "89.0%")
        st.metric("F1-Score", "87.0%")
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Dibuat dengan menggunakan Streamlit dan Scikit-learn</p>
        <p>Model dilatih pada Dataset Ulasan Film IMDB</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
