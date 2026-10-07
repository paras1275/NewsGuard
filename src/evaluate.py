import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from src.config import (
    PROCESSED_DATA_DIR, BEST_MODEL_PATH, METADATA_PATH, FIGURES_DIR,
    COMPARISON_CSV_PATH
)
from src.feature_engineering import NewsFeaturePipeline, build_features_for_model

# Set plot style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

def evaluate_model_on_test():
    """
    Evaluates the final best model on held-out test dataset and saves all evaluation plots.
    """
    print("\n[Evaluate] Loading held-out test dataset...")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_test_raw = test_df["text"].tolist()
    y_test = test_df["label"].values

    # Load fitted feature pipeline, best model, and metadata
    pipeline = NewsFeaturePipeline().load()
    best_model = joblib.load(BEST_MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    best_model_name = metadata.get("best_model_name", type(best_model).__name__)
    feature_pipeline = metadata.get("feature_pipeline", "tfidf_only" if best_model_name == "Multinomial Naive Bayes" else "combined")

    X_test = build_features_for_model(X_test_raw, feature_pipeline, pipeline)

    expected = getattr(best_model, "n_features_in_", metadata.get("expected_features", X_test.shape[1]))
    print(f"[Evaluate] Selected model: {best_model_name}")
    print(f"[Evaluate] Feature pipeline: {feature_pipeline}")
    print(f"[Evaluate] Expected features: {expected}")
    print(f"[Evaluate] Generated features: {X_test.shape[1]}")

    if expected is not None and X_test.shape[1] != expected:
        raise ValueError(
            f"Feature pipeline mismatch: model expects {expected} features, "
            f"but generated {X_test.shape[1]} features using pipeline '{feature_pipeline}'."
        )

    # Predict class & probabilities
    y_pred = best_model.predict(X_test)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        d_func = best_model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-d_func))
    else:
        y_prob = y_pred.astype(float)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "Test Accuracy": round(acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall": round(rec, 4),
        "Test F1 Score": round(f1, 4),
        "Test ROC-AUC": round(auc, 4)
    }

    print("\n=== Final Test Dataset Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["FAKE (0)", "REAL (1)"], yticklabels=["FAKE (0)", "REAL (1)"], ax=ax)
    ax.set_title("Confusion Matrix — NewsGuard Test Evaluation", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved Confusion Matrix to {cm_path}")

    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved ROC Curve to {roc_path}")

    # 3. Plot Probability Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.kdeplot(y_prob[y_test == 1], color="#10B981", label="True REAL Articles", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(y_prob[y_test == 0], color="#EF4444", label="True FAKE Articles", fill=True, alpha=0.4, ax=ax)
    ax.set_title("Predicted Probability Distribution (Credibility Score Basis)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Probability P(REAL)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend()
    plt.tight_layout()
    prob_path = FIGURES_DIR / "prob_dist.png"
    plt.savefig(prob_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved Probability Distribution plot to {prob_path}")

    # 4. Plot Model Comparison Chart (if comparison csv exists)
    if os.path.exists(COMPARISON_CSV_PATH):
        comp_df = pd.read_csv(COMPARISON_CSV_PATH)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=comp_df, x="F1 Score", y="Model", palette="Blues_r", ax=ax)
        ax.set_title("Cross-Validation Model Benchmark (F1 Score)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0.7, 1.0)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.4f}", (width - 0.04, p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        plt.tight_layout()
        comp_plot_path = FIGURES_DIR / "model_comparison.png"
        plt.savefig(comp_plot_path, dpi=300)
        plt.close()
        print(f"[Evaluate] Saved Model Comparison chart to {comp_plot_path}")

    return metrics

if __name__ == "__main__":
    evaluate_model_on_test()
