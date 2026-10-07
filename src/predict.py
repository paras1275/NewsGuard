import time
import json
from typing import Dict, List
import joblib
import pandas as pd
from src.config import BEST_MODEL_PATH, FAKE_NEWS_MODEL_PATH, METADATA_PATH, get_credibility_category, validate_model_files
from src.feature_engineering import NewsFeaturePipeline, build_features_for_model
from src.explain import explain_prediction

class NewsPredictor:
    """
    End-to-end prediction engine for NewsGuard.
    Loads trained model & pipeline, computes prediction, credibility score, and explanations.
    """
    def __init__(self):
        self.is_ready, self.missing_files = validate_model_files()
        self.pipeline = None
        self.model = None
        self.metadata = {}

        if not self.is_ready:
            print(f"[NewsPredictor] ⚠️ Startup Validation Alert: The following required model files are missing: {self.missing_files}")
            print("[NewsPredictor] Please execute the training pipeline first: `python run.py --train`")
        else:
            try:
                self.pipeline = NewsFeaturePipeline().load()
                model_path = FAKE_NEWS_MODEL_PATH if FAKE_NEWS_MODEL_PATH.exists() else BEST_MODEL_PATH
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"[NewsPredictor] Error loading model artifacts: {e}")
                self.is_ready = False

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)

    def predict_single(self, text: str) -> Dict:
        """
        Predicts credibility and label for a single news article text.
        Returns detailed result dictionary.
        """
        start_time = time.time()

        if not self.is_ready:
            raise FileNotFoundError("Prediction unavailable because the trained model could not be loaded.")

        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input article text cannot be empty.")

        # Get SHAP / feature explanation
        explanation = explain_prediction(text)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "label": explanation["prediction"],
            "credibility_score": explanation["credibility_score"],
            "confidence": explanation["confidence_category"],
            "confidence_color": explanation["confidence_color"],
            "top_features": explanation["top_positive_features"] if explanation["prediction"] == "REAL" else explanation["top_negative_features"],
            "positive_signals": explanation["top_positive_features"],
            "negative_signals": explanation["top_negative_features"],
            "model_used": self.metadata.get("best_model_name", "Best Trained Model"),
            "execution_time_ms": elapsed_ms
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predicts credibility and labels for a batch of news articles."""
        return [self.predict_single(t) for t in texts]

if __name__ == "__main__":
    predictor = NewsPredictor()
    sample = "Scientists at Nature Astronomy confirmed atmospheric water signatures on exoplanet."
    res = predictor.predict_single(sample)
    print("Single Prediction Result:")
    print(json.dumps(res, indent=2))
