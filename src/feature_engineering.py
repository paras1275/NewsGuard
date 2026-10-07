import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
# Defensive optional imports with fallbacks
try:
    from gensim.models import Word2Vec  # type: ignore
except ImportError:
    Word2Vec = None

try:
    import textstat  # type: ignore
except ImportError:
    textstat = None

try:
    from textblob import TextBlob  # type: ignore
except ImportError:
    TextBlob = None


from src.config import TFIDF_PARAMS, W2V_PARAMS, TFIDF_VECTORIZER_PATH, WORD2VEC_MODEL_PATH, SCALER_PATH
from src.preprocessing import preprocess_article, tokenize_text
import joblib


class AuxiliaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts Readability, Sentiment, and Statistical/Linguistic features from raw text.
    Returns a dense numpy array of numerical features.
    """
    def __init__(self):
        self.feature_names = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "sentence_count",
            "avg_sentence_length",
            "sentiment_polarity",
            "sentiment_subjectivity",
            "char_length",
            "word_count",
            "unique_word_ratio",
            "punctuation_count",
            "exclamation_count",
            "question_count",
            "uppercase_word_ratio",
            "avg_word_length"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text) if text is not None else ""
            words = text_str.split()
            word_count = len(words)
            char_len = len(text_str)
            
            # Readability (TextStat)
            try:
                fre = textstat.flesch_reading_ease(text_str)
                fkg = textstat.flesch_kincaid_grade(text_str)
                sent_cnt = max(1, textstat.sentence_count(text_str))
                avg_sent_len = word_count / sent_cnt
            except Exception:
                fre, fkg, sent_cnt, avg_sent_len = 50.0, 8.0, 1.0, float(word_count)

            # Sentiment (TextBlob)
            try:
                blob = TextBlob(text_str)
                pol = blob.sentiment.polarity
                subj = blob.sentiment.subjectivity
            except Exception:
                pol, subj = 0.0, 0.0

            # Statistical & Linguistic
            unique_ratio = len(set(words)) / max(1, word_count)
            punct_cnt = sum(1 for c in text_str if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            excl_cnt = text_str.count('!')
            quest_cnt = text_str.count('?')
            upper_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(1, word_count)
            avg_word_len = sum(len(w) for w in words) / max(1, word_count)

            vec = [
                fre, fkg, sent_cnt, avg_sent_len,
                pol, subj,
                char_len, word_count, unique_ratio,
                punct_cnt, excl_cnt, quest_cnt,
                upper_ratio, avg_word_len
            ]
            features.append(vec)

        return np.array(features, dtype=np.float32)

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Gensim Word2Vec Document Vectorizer.
    Calculates average word vectors for each document text.
    Handles out-of-vocabulary words gracefully.
    """
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.w2v_model = None

    def fit(self, X_tokens: List[List[str]], y=None):
        """Fits Word2Vec model on tokenized text list."""
        self.w2v_model = Word2Vec(
            sentences=X_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs,
            seed=42
        )
        return self

    def transform(self, X_tokens: List[List[str]]) -> np.ndarray:
        """Computes mean Word2Vec embedding vector per document."""
        embeddings = []
        for tokens in X_tokens:
            if not tokens or self.w2v_model is None:
                vec = np.zeros(self.vector_size, dtype=np.float32)
            else:
                vectors = [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
                if vectors:
                    vec = np.mean(vectors, axis=0)
                else:
                    vec = np.zeros(self.vector_size, dtype=np.float32)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

class NewsFeaturePipeline:
    """
    Combined Feature Pipeline orchestrating:
    1. Preprocessing
    2. TF-IDF Vectorizer
    3. Word2Vec Document Embeddings
    4. Auxiliary Features (Readability + Sentiment + Linguistic)
    5. Feature Scaling for dense features
    """
    def __init__(self):
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        self.aux_extractor = AuxiliaryFeatureExtractor()
        self.w2v = Word2VecVectorizer(vector_size=W2V_PARAMS["vector_size"])
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, raw_texts: List[str], labels: List[int] = None):
        print("[FeaturePipeline] Preprocessing raw training text...")
        processed_texts = [preprocess_article(t) for t in raw_texts]
        tokenized_texts = [t.split() for t in processed_texts]

        print("[FeaturePipeline] Fitting TF-IDF Vectorizer...")
        self.tfidf.fit(processed_texts)

        print("[FeaturePipeline] Training Word2Vec Model...")
        self.w2v.fit(tokenized_texts)

        print("[FeaturePipeline] Extracting and scaling Auxiliary + Word2Vec features...")
        aux_features = self.aux_extractor.transform(raw_texts)
        w2v_features = self.w2v.transform(tokenized_texts)
        dense_combined = np.hstack([aux_features, w2v_features])

        self.scaler.fit(dense_combined)
        self.is_fitted = True
        return self

    def transform(self, raw_texts: List[str]) -> csr_matrix:
        if not self.is_fitted:
            raise ValueError("FeaturePipeline is not fitted yet! Call fit() first.")
        
        processed_texts = [preprocess_article(t) for t in raw_texts]
        tokenized_texts = [t.split() for t in processed_texts]

        # 1. TF-IDF sparse matrix
        tfidf_mat = self.tfidf.transform(processed_texts)

        # 2. Dense auxiliary + Word2Vec
        aux_features = self.aux_extractor.transform(raw_texts)
        w2v_features = self.w2v.transform(tokenized_texts)
        dense_combined = np.hstack([aux_features, w2v_features])
        scaled_dense = self.scaler.transform(dense_combined)

        # 3. Stack sparse and scaled dense matrices
        combined_matrix = hstack([tfidf_mat, csr_matrix(scaled_dense)]).tocsr()
        return combined_matrix

    def fit_transform(self, raw_texts: List[str], labels: List[int] = None) -> csr_matrix:
        self.fit(raw_texts, labels)
        return self.transform(raw_texts)

    def save(self):
        joblib.dump(self.tfidf, TFIDF_VECTORIZER_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        if self.w2v.w2v_model is not None:
            self.w2v.w2v_model.save(str(WORD2VEC_MODEL_PATH))
        print(f"[FeaturePipeline] Exported TF-IDF, Word2Vec, and Scaler artifacts to {TFIDF_VECTORIZER_PATH.parent}")

    def load(self):
        if not TFIDF_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer artifact not found at '{TFIDF_VECTORIZER_PATH}'. "
                f"Please run model training pipeline first using 'python run.py --train'."
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found at '{SCALER_PATH}'. "
                f"Please run model training pipeline first using 'python run.py --train'."
            )
        self.tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        if WORD2VEC_MODEL_PATH.exists() and Word2Vec is not None:
            try:
                w2v_mod = Word2Vec.load(str(WORD2VEC_MODEL_PATH))
                self.w2v.w2v_model = w2v_mod
            except Exception:
                pass
        self.is_fitted = True
        return self

def analyze_discriminative_terms(tfidf_vec: TfidfVectorizer, X_tfidf: csr_matrix, y: np.ndarray, top_n: int = 15) -> dict:
    """Computes top discriminative terms for Real vs Fake news."""
    feature_names = np.array(tfidf_vec.get_feature_names_out())
    fake_mean = np.mean(X_tfidf[y == 0], axis=0).A1
    real_mean = np.mean(X_tfidf[y == 1], axis=0).A1
    
    diff = real_mean - fake_mean
    top_real_idx = np.argsort(diff)[::-1][:top_n]
    top_fake_idx = np.argsort(diff)[:top_n]
    
    return {
        "vocab_size": len(feature_names),
        "top_real_terms": list(zip(feature_names[top_real_idx], real_mean[top_real_idx])),
        "top_fake_terms": list(zip(feature_names[top_fake_idx], fake_mean[top_fake_idx]))
    }

def build_features_for_model(raw_texts: List[str], feature_pipeline: str, pipeline: NewsFeaturePipeline) -> csr_matrix:
    """
    Shared helper function for constructing features for evaluation and live prediction.
    If feature_pipeline is 'tfidf_only', transforms raw texts using the saved TF-IDF vectorizer only.
    Otherwise, uses the full combined feature pipeline (TF-IDF + Auxiliary + Word2Vec).
    """
    if feature_pipeline == "tfidf_only":
        processed_texts = [preprocess_article(t) for t in raw_texts]
        return pipeline.tfidf.transform(processed_texts)
    else:
        return pipeline.transform(raw_texts)

