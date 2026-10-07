import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "news_dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"

# Model File Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FAKE_NEWS_MODEL_PATH = MODELS_DIR / "fake_news_model.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec.model"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Create required directories
for d in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def validate_model_files() -> tuple:
    """
    Validates existence of required model artifacts.
    Returns (is_complete: bool, missing_files: list[str]).
    """
    missing = []
    if not TFIDF_VECTORIZER_PATH.exists():
        missing.append(f"models/{TFIDF_VECTORIZER_PATH.name}")
    
    if not (FAKE_NEWS_MODEL_PATH.exists() or BEST_MODEL_PATH.exists()):
        missing.append("models/fake_news_model.joblib")

    if not SCALER_PATH.exists():
        missing.append(f"models/{SCALER_PATH.name}")

    return len(missing) == 0, missing

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Train / Validation / Test Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# TF-IDF Settings
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 3000,
    "sublinear_tf": True
}

# Word2Vec Settings
W2V_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "workers": 4,
    "epochs": 10
}

# Credibility Score Thresholds
CREDIBILITY_LEVELS = [
    (0.0, 30.0, "Very Low Credibility", "#EF4444"),
    (30.0, 50.0, "Low Credibility", "#F97316"),
    (50.0, 70.0, "Moderate Credibility", "#FBBF24"),
    (70.0, 85.0, "High Credibility", "#3B82F6"),
    (85.0, 100.0, "Very High Credibility", "#10B981")
]

def get_credibility_category(score: float):
    """Categorizes a 0-100 credibility score into risk levels."""
    for low, high, label, color in CREDIBILITY_LEVELS:
        if low <= score <= high:
            return label, color
    return "High Credibility" if score >= 70 else "Low Credibility", "#3B82F6"
