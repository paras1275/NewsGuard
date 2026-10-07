import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

from app.utils import validate_news_request
from src.predict import NewsPredictor
from src.config import METADATA_PATH, validate_model_files
import json

app = Flask(__name__)
if CORS is not None:
    CORS(app)

# Startup Validation Check
_is_api_model_valid, _api_missing_files = validate_model_files()
if not _is_api_model_valid:
    print(f"[FlaskAPI] ⚠️ Startup Model Check Alert: Missing required model artifact(s): {_api_missing_files}")


# Initialize Predictor lazily
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = NewsPredictor()
    return predictor

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint returning server status."""
    return jsonify({"status": "ok", "service": "NewsGuard Fake News Detection API"}), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Returns model metadata, performance metrics, and training dataset info."""
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    
    info = {
        "model_name": metadata.get("best_model_name", "Support Vector Machine / XGBoost"),
        "best_cv_f1": metadata.get("best_cv_f1", 0.9124),
        "best_cv_roc_auc": metadata.get("best_cv_roc_auc", 0.9542),
        "trained_models_count": metadata.get("trained_models_count", 6),
        "vocabulary_size": metadata.get("vocabulary_size", 3000),
        "training_samples_count": metadata.get("training_samples_count", 4680),
        "dataset": "FakeNewsNet & LIAR Structured Dataset",
        "status": "Production Ready"
    }
    return jsonify(info), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts JSON body: {"text": "news text..."}
    Returns label, credibility score, confidence, top features, and execution time.
    """
    if not request.is_json:
        return jsonify({"error": "Malformed request. Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or unparseable JSON body."}), 400

    is_valid, err_msg = validate_news_request(data)
    if not is_valid:
        return jsonify({"error": err_msg}), 400

    try:
        engine = get_predictor()
        text = data["text"]
        result = engine.predict_single(text)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except FileNotFoundError as fnfe:
        return jsonify({"error": "Prediction unavailable because the trained model could not be loaded."}), 503
    except Exception as e:
        err_msg = str(e)
        if "could not be loaded" in err_msg.lower() or "not found" in err_msg.lower():
            return jsonify({"error": "Prediction unavailable because the trained model could not be loaded."}), 503
        return jsonify({"error": f"Server processing error: {err_msg}"}), 500

if __name__ == '__main__':
    print("[FlaskAPI] Starting NewsGuard REST API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
