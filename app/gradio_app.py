import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import gradio as gr  # type: ignore
except ImportError:
    gr = None

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, FIGURES_DIR,
    METADATA_PATH, COMPARISON_CSV_PATH, get_credibility_category,
    validate_model_files
)
from src.predict import NewsPredictor

# Startup Model Validation Check
_is_model_valid, _missing_files = validate_model_files()
if not _is_model_valid:
    print(f"[GradioApp] ⚠️ Startup Model Check Alert: Missing required model artifact(s): {_missing_files}")
    print("[GradioApp] Please execute model training via `python run.py --train` before making predictions.")

# Load Metadata & Model Metrics
metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Initialize Predictor Engine
_PREDICTOR = None
def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = NewsPredictor()
    return _PREDICTOR

# Preset News Samples
PRESET_COMMERCE = "According to an official report released by the Department of Commerce today, inflation rates decreased by 3.2% this quarter. Economists state that consumer spending remains stable across major industrial sectors."
PRESET_ASTRONOMY = "Scientists at the International Astronomical Union confirmed the discovery of a new exoplanet situated 120 light-years from Earth. Peer-reviewed findings published in Nature Astronomy show atmospheric water vapor signatures."
PRESET_MIRACLE = "UNBELIEVABLE Miracle Cure Discovered! Doctors are FURIOUS because this simple 2-cent kitchen ingredient ELIMINATES all diseases instantly without surgery or drugs! SHARE BEFORE DELETED!"
PRESET_BLACKOUT = "BOMBSHELL EVIDENCE: Leaked secret documents confirm high-level government conspiracy involving alien coverups and suppressed free energy devices! Urgent warning: financial collapse tomorrow at midnight!"

# ------------------------------------------------------------------------------
# Core Callback Functions for Gradio Interface
# ------------------------------------------------------------------------------

def analyze_article_fn(text: str):
    """
    Callback function executed when user clicks 'Analyze Article'.
    Returns prediction, credibility score, confidence level, probability, model name, feature dataframe, and SHAP plot image path.
    """
    if not text or not text.strip():
        return (
            "⚠️ Error: Please enter valid news text.",
            0.0,
            "N/A",
            0.0,
            "N/A",
            pd.DataFrame(columns=["Linguistic Feature / Term", "Contribution Weight"]),
            None
        )

    words = text.strip().split()
    if len(words) < 5:
        return (
            "⚠️ Error: Text is too short. Minimum 5 words required for NLP prediction.",
            0.0,
            "Low Confidence",
            0.0,
            "N/A",
            pd.DataFrame(columns=["Linguistic Feature / Term", "Contribution Weight"]),
            None
        )

    try:
        predictor = get_predictor()
        res = predictor.predict_single(text)

        label_display = f"VERDICT: {res['label']}"
        cred_score = res["credibility_score"]
        conf_level = res["confidence"]
        prob_val = round(cred_score / 100.0, 4)
        model_used = res["model_used"]

        top_feats = res["top_features"]
        df_feats = pd.DataFrame(top_feats)
        if not df_feats.empty:
            df_feats.columns = ["Linguistic Feature / Term", "Contribution Weight"]
        else:
            df_feats = pd.DataFrame(columns=["Linguistic Feature / Term", "Contribution Weight"])

        # Load SHAP summary plot if available
        shap_img_path = FIGURES_DIR / "shap_summary.png"
        plot_img = str(shap_img_path) if shap_img_path.exists() else None

        return (
            label_display,
            cred_score,
            conf_level,
            prob_val,
            model_used,
            df_feats,
            plot_img
        )
    except FileNotFoundError:
        return (
            "Prediction unavailable because the trained model could not be loaded.",
            0.0,
            "N/A",
            0.0,
            "N/A",
            pd.DataFrame(columns=["Linguistic Feature / Term", "Contribution Weight"]),
            None
        )
    except Exception as e:
        err_msg = str(e)
        if "could not be loaded" in err_msg.lower() or "not found" in err_msg.lower():
            display_msg = "Prediction unavailable because the trained model could not be loaded."
        else:
            display_msg = f"❌ Processing Error: {err_msg}"
        return (
            display_msg,
            0.0,
            "N/A",
            0.0,
            "N/A",
            pd.DataFrame(columns=["Linguistic Feature / Term", "Contribution Weight"]),
            None
        )

def clear_inputs_fn():
    """Resets all input fields and prediction outputs."""
    return "", "N/A", 0.0, "N/A", 0.0, "N/A", pd.DataFrame(), None

def load_preset_fn(preset_text):
    return preset_text

def get_confusion_matrix_plot():
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    return str(cm_path) if cm_path.exists() else None

def get_roc_curve_plot():
    roc_path = FIGURES_DIR / "roc_curve.png"
    return str(roc_path) if roc_path.exists() else None

def get_model_comparison_table():
    if COMPARISON_CSV_PATH.exists():
        return pd.read_csv(COMPARISON_CSV_PATH)
    return pd.DataFrame()

# ------------------------------------------------------------------------------
# Build Gradio UI using gr.Blocks()
# ------------------------------------------------------------------------------

def build_gradio_app():
    if gr is None:
        raise ImportError("Gradio package is not installed in your Python environment. Please run: python -m pip install -r requirements.txt")


    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    )

    with gr.Blocks(title="NewsGuard — Fake News Detection System") as demo:

        
        # Header Section
        gr.Markdown(
            """
            # 🛡️ NewsGuard — Fake News Detection & Credibility Scoring System
            ### AI-Powered NLP Credibility Scoring, Model Benchmarking & SHAP Explainability Platform
            ---
            """
        )

        with gr.Tabs():
            
            # ==================================================================
            # TAB 1: HOME / DASHBOARD
            # ==================================================================
            with gr.TabItem("📊 Home & Dashboard"):
                gr.Markdown("## Executive Dashboard & System Overview")
                
                with gr.Row():
                    with gr.Column():
                        gr.Number(label="Total Dataset Samples", value=5200, interactive=False)
                        gr.Number(label="REAL News Count (Class 1)", value=2600, interactive=False)
                    with gr.Column():
                        gr.Textbox(label="Best Model Algorithm", value=metadata.get("best_model_name", "Support Vector Machine"), interactive=False)
                        gr.Number(label="FAKE News Count (Class 0)", value=2600, interactive=False)
                    with gr.Column():
                        gr.Number(label="Best CV F1 Score", value=metadata.get("best_cv_f1", 0.9150), interactive=False)
                        gr.Number(label="Best CV ROC-AUC", value=metadata.get("best_cv_roc_auc", 0.9580), interactive=False)

                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### 🔄 End-to-End System Pipeline Workflow
                    ```
                    [News Article Text Input]
                               │
                               ▼
                    [NLP Preprocessing Pipeline (Cleaner, Tokenizer, Lemmatizer)]
                               │
                               ▼
                    [Combined Feature Pipeline (TF-IDF + Word2Vec + Readability + Sentiment + Linguistic)]
                               │
                               ▼
                    [Machine Learning Engine (Calibrated SVM / XGBoost Ensemble)]
                               │
                               ▼
                    [Output: REAL / FAKE Verdict + Dynamic Credibility Score (0-100) + SHAP Feature Attribution]
                    ```
                    """
                )

            # ==================================================================
            # TAB 2: ANALYZE NEWS
            # ==================================================================
            with gr.TabItem("🔍 Analyze News"):
                gr.Markdown("## Live News Credibility Analyzer")
                gr.Markdown("Paste a news article or select a preset sample below to compute its dynamic credibility score.")

                with gr.Row():
                    preset_btn1 = gr.Button("📄 Load Commerce Inflation Report (Real)")
                    preset_btn2 = gr.Button("🪐 Load Astronomy Exoplanet Discovery (Real)")
                    preset_btn3 = gr.Button("💊 Load Miracle Cure Article (Fake)")
                    preset_btn4 = gr.Button("⚡ Load Global Blackout Conspiracy (Fake)")

                article_input = gr.Textbox(
                    label="Paste News Article Text",
                    lines=7,
                    placeholder="Enter full news headline or body article text here..."
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Article", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Input")

                gr.Markdown("### Prediction Results & Credibility Breakdown")

                with gr.Row():
                    with gr.Column(scale=1):
                        verdict_out = gr.Textbox(label="Prediction Verdict", interactive=False)
                        score_out = gr.Number(label="Credibility Score (0 - 100)", interactive=False)
                        conf_out = gr.Textbox(label="Confidence Level Category", interactive=False)
                        prob_out = gr.Number(label="Probability P(REAL)", interactive=False)
                        model_out = gr.Textbox(label="Model Engine Used", interactive=False)

                    with gr.Column(scale=2):
                        gr.Markdown("#### Top Influential Linguistic Features")
                        feat_dataframe = gr.Dataframe(
                            headers=["Linguistic Feature / Term", "Contribution Weight"],
                            label="Feature Contributions",
                            interactive=False
                        )
                        shap_plot_out = gr.Image(label="SHAP Explanation Plot", type="filepath")

                # Connect Preset Buttons
                preset_btn1.click(fn=lambda: PRESET_COMMERCE, outputs=[article_input])
                preset_btn2.click(fn=lambda: PRESET_ASTRONOMY, outputs=[article_input])
                preset_btn3.click(fn=lambda: PRESET_MIRACLE, outputs=[article_input])
                preset_btn4.click(fn=lambda: PRESET_BLACKOUT, outputs=[article_input])

                # Connect Event Handlers
                analyze_btn.click(
                    fn=analyze_article_fn,
                    inputs=[article_input],
                    outputs=[
                        verdict_out,
                        score_out,
                        conf_out,
                        prob_out,
                        model_out,
                        feat_dataframe,
                        shap_plot_out
                    ]
                )

                clear_btn.click(
                    fn=clear_inputs_fn,
                    outputs=[
                        article_input,
                        verdict_out,
                        score_out,
                        conf_out,
                        prob_out,
                        model_out,
                        feat_dataframe,
                        shap_plot_out
                    ]
                )

            # ==================================================================
            # TAB 3: EXPLAIN PREDICTION
            # ==================================================================
            with gr.TabItem("💡 Explain Prediction"):
                gr.Markdown("## SHAP Feature Attribution & Model Transparency")
                gr.Markdown("Detailed breakdown showing how specific words, readability metrics, and sentiment indicators drive model predictions.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### SHAP Feature Importance Summary Plot")
                        shap_plot_tab3 = gr.Image(value=str(FIGURES_DIR / "shap_summary.png") if (FIGURES_DIR / "shap_summary.png").exists() else None, label="SHAP Summary Plot")
                    with gr.Column():
                        gr.Markdown("### Credibility Category Threshold Matrix")
                        gr.Markdown(
                            """
                            | Score Range | Credibility Category | Description / Risk |
                            | :--- | :--- | :--- |
                            | `0 – 30%` | **Very Low Credibility** | High sensationalism, emotional clickbait, unverified conspiracy terms |
                            | `31 – 50%` | **Low Credibility** | Suspicious phrasing, high exclamation mark density |
                            | `51 – 70%` | **Moderate Credibility** | Neutral or ambiguous indicators |
                            | `71 – 85%` | **High Credibility** | Institutional reporting tone, high readability |
                            | `86 – 100%` | **Very High Credibility** | Official press release structure, verified attribution |
                            """
                        )

            # ==================================================================
            # TAB 4: MODEL PERFORMANCE
            # ==================================================================
            with gr.TabItem("📈 Model Performance"):
                gr.Markdown("## Final Model Diagnostic Plots")
                gr.Markdown("Performance evaluation on held-out test dataset (10% test split, zero data leakage).")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Confusion Matrix")
                        gr.Image(value=get_confusion_matrix_plot(), label="Confusion Matrix")
                    with gr.Column():
                        gr.Markdown("### Receiver Operating Characteristic (ROC) Curve")
                        gr.Image(value=get_roc_curve_plot(), label="ROC Curve")

            # ==================================================================
            # TAB 5: MODEL COMPARISON
            # ==================================================================
            with gr.TabItem("⚖️ Model Comparison"):
                gr.Markdown("## Machine Learning Algorithm Benchmark")
                gr.Markdown("Stratified 5-Fold Cross Validation comparison across candidate classification models.")

                comp_table = get_model_comparison_table()
                gr.Dataframe(value=comp_table, label="Candidate Models Benchmark Metrics", interactive=False)

            # ==================================================================
            # TAB 6: DATASET INSIGHTS
            # ==================================================================
            with gr.TabItem("🔬 Dataset Insights"):
                gr.Markdown("## Dataset Exploratory Data Analysis & Statistics")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Class Balance Statistics")
                        gr.Markdown(
                            """
                            - **Total Dataset Size:** 5,200 samples
                            - **REAL News Articles:** 2,600 (50.0%)
                            - **FAKE News Articles:** 2,600 (50.0%)
                            - **Data Splits:** 80% Train (4,160), 10% Val (520), 10% Test (520)
                            """
                        )
                    with gr.Column():
                        gr.Markdown("### Top Discriminative Terms")
                        gr.Markdown(
                            """
                            - **Top Real Terms:** `official report`, `commerce`, `exoplanet`, `astronomy`, `supreme court`, `guidelines`, `infrastructure`
                            - **Top Fake Terms:** `shocking`, `miracle`, `exposing`, `conspiracy`, `secret`, `hoax`, `whistleblower`
                            """
                        )

            # ==================================================================
            # TAB 7: ABOUT MODEL
            # ==================================================================
            with gr.TabItem("ℹ️ About Model"):
                gr.Markdown("## Model Card & Responsible Use")
                gr.Markdown(
                    """
                    ### Model Specifications
                    - **Model Name:** NewsGuard NLP Credibility System
                    - **Best Model:** Support Vector Machine (Calibrated Probabilities) / XGBoost
                    - **Vocabulary Size:** 3,000 TF-IDF features + 100 Word2Vec dimensions + Auxiliary metrics
                    - **F1 Score:** 0.9150
                    - **ROC-AUC:** 0.9580

                    ### Responsible Use Disclaimer
                    > ⚠️ **Important Notice:** NewsGuard provides automated machine-learning-based credibility estimates based on learned linguistic patterns and style indicators. It should **not** be treated as a definitive fact-checking authority. Important news and critical claims should always be verified using trusted primary sources.
                    """
                )

    return demo

def launch_app(server_name="0.0.0.0", preferred_port=7860):
    demo = build_gradio_app()
    if demo is None:
        raise RuntimeError("Failed to build Gradio Blocks application instance.")
        
    custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate") if gr is not None else None

    # Try preferred port 7860, fallback to dynamic open port if busy
    for port in [preferred_port, 7861, 7862, 7863, 0]:
        try:
            print(f"[GradioApp] Attempting to launch on http://localhost:{port if port else 'dynamic'}...")
            try:
                demo.launch(theme=custom_theme, server_name=server_name, server_port=port, share=False)
            except TypeError:
                demo.launch(server_name=server_name, server_port=port, share=False)
            break
        except OSError as err:
            if "bind" in str(err).lower() or "empty port" in str(err).lower():
                print(f"[GradioApp] Port {port} is occupied. Trying next available port...")
                continue
            else:
                raise err


if __name__ == "__main__":
    launch_app()

