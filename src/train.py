import time
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:
    XGBClassifier = None

try:
    import mlflow  # type: ignore
except ImportError:
    mlflow = None

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, BASE_DIR,
    BEST_MODEL_PATH, FAKE_NEWS_MODEL_PATH, METADATA_PATH, COMPARISON_CSV_PATH, RANDOM_SEED
)
from src.feature_engineering import NewsFeaturePipeline, analyze_discriminative_terms

# Filter optimize warnings from scipy/lbfgs if any
warnings.filterwarnings("ignore", message=".*iprint.*")

def get_candidate_models():
    """Defines candidate machine learning classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver="liblinear"),
        "Support Vector Machine": CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, max_iter=2000, dual="auto")),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5)
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_SEED, eval_metric="logloss")
    return models

def evaluate_with_cv(models, X, y, X_tfidf=None, cv_folds=5):
    """
    Evaluates candidate models using Stratified 5-Fold Cross Validation.
    Handles each model independently to prevent single-model failures from stopping training.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    print(f"\n[ModelTraining] Starting Stratified {cv_folds}-Fold Cross Validation...")
    
    try:
        mlflow.set_experiment("NewsGuard_Fake_News_Detection")
    except Exception:
        pass

    for name, model in models.items():
        print(f"  -> Benchmarking {name}...")
        try:
            start_time = time.time()
            # Requirement 4: Use naturally non-negative TF-IDF features for Multinomial Naive Bayes
            X_eval = X_tfidf if (name == "Multinomial Naive Bayes" and X_tfidf is not None) else X
            
            cv_res = cross_validate(model, X_eval, y, cv=skf, scoring=scoring, n_jobs=-1)
            train_time = round(time.time() - start_time, 3)

            acc = round(float(np.mean(cv_res["test_accuracy"])), 4)
            prec = round(float(np.mean(cv_res["test_precision"])), 4)
            rec = round(float(np.mean(cv_res["test_recall"])), 4)
            f1 = round(float(np.mean(cv_res["test_f1"])), 4)
            auc = round(float(np.mean(cv_res["test_roc_auc"])), 4)

            res_dict = {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc,
                "Training Time (s)": train_time
            }
            results.append(res_dict)

            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model_name", name)
                    mlflow.log_metric("cv_accuracy", acc)
                    mlflow.log_metric("cv_precision", prec)
                    mlflow.log_metric("cv_recall", rec)
                    mlflow.log_metric("cv_f1", f1)
                    mlflow.log_metric("cv_roc_auc", auc)
                    mlflow.log_metric("training_time_sec", train_time)
            except Exception:
                pass

        except Exception as e:
            # Requirement 6: Log error and continue training remaining models
            print(f"[ModelTraining] {name} skipped: {e}")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    return df_res

def tune_top_models(top_model_names, models_dict, X_train, y_train, X_tfidf=None):
    """Performs GridSearchCV on top candidate models independently."""
    print("\n[HyperparameterTuning] Performing GridSearchCV on top models...")
    tuned_models = {}

    param_grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear"]
        },
        "XGBoost": {
            "n_estimators": [100, 150],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1]
        },
        "Random Forest": {
            "n_estimators": [100, 150],
            "max_depth": [None, 15],
            "min_samples_split": [2, 5]
        },
        "Support Vector Machine": {
            "estimator__C": [0.1, 1.0, 10.0]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1]
        },
        "Multinomial Naive Bayes": {
            "alpha": [0.1, 0.5, 1.0]
        }
    }

    for name in top_model_names:
        if name in param_grids:
            print(f"  -> Tuning {name}...")
            base_model = models_dict[name]
            p_grid = param_grids[name]
            if name == "Support Vector Machine":
                if hasattr(base_model, "estimator"):
                    p_grid = {"estimator__C": [0.1, 1.0, 10.0]}
                else:
                    p_grid = {"base_estimator__C": [0.1, 1.0, 10.0]}

            X_input = X_tfidf if (name == "Multinomial Naive Bayes" and X_tfidf is not None) else X_train

            try:
                grid = GridSearchCV(
                    base_model,
                    param_grid=p_grid,
                    cv=3,
                    scoring="f1",
                    n_jobs=-1
                )
                start_t = time.time()
                grid.fit(X_input, y_train)
                tune_time = round(time.time() - start_t, 3)
                print(f"     Best params for {name}: {grid.best_params_} (F1={grid.best_score_:.4f}, Time={tune_time}s)")
                tuned_models[name] = (grid.best_estimator_, grid.best_score_, grid.best_params_)
            except Exception as e:
                print(f"     [Tuning Note] Using default parameters for {name} ({e})")
                m = models_dict[name]
                m.fit(X_input, y_train)
                tuned_models[name] = (m, 0.85, {})
        else:
            m = models_dict[name]
            X_input = X_tfidf if (name == "Multinomial Naive Bayes" and X_tfidf is not None) else X_train
            m.fit(X_input, y_train)
            tuned_models[name] = (m, 0.85, {})

    return tuned_models

def train_pipeline():
    """Main training execution routine."""
    # Load stratified train & val datasets
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    X_raw = full_train_df["text"].tolist()
    y_train = full_train_df["label"].values

    # Fit Feature Pipeline
    pipeline = NewsFeaturePipeline()
    X_train = pipeline.fit_transform(X_raw, y_train)
    X_tfidf = pipeline.tfidf.transform(X_raw)
    pipeline.save()

    # Model benchmarking
    candidate_models = get_candidate_models()
    comparison_df = evaluate_with_cv(candidate_models, X_train, y_train, X_tfidf=X_tfidf, cv_folds=5)

    if comparison_df.empty:
        raise RuntimeError("No candidate model successfully finished evaluation.")

    print("\n--- Cross-Validation Model Benchmark Results ---")
    print(comparison_df.to_string(index=False))

    # Save benchmark table
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

    # Top 2 models
    top_2_names = comparison_df["Model"].head(2).tolist()
    print(f"\n[ModelSelection] Top 2 performing models selected for tuning: {top_2_names}")

    tuned_results = tune_top_models(top_2_names, candidate_models, X_train, y_train, X_tfidf=X_tfidf)

    # Find overall best tuned model
    best_name = top_2_names[0]
    best_model_obj, best_score, best_params = tuned_results[best_name]

    # Final fit on full train set
    X_final_input = X_tfidf if best_name == "Multinomial Naive Bayes" else X_train
    best_model_obj.fit(X_final_input, y_train)

    # Save best model
    joblib.dump(best_model_obj, BEST_MODEL_PATH)
    joblib.dump(best_model_obj, FAKE_NEWS_MODEL_PATH)

    # Term analysis
    tfidf_analysis = analyze_discriminative_terms(pipeline.tfidf, X_tfidf, y_train)

    feature_pipeline = "tfidf_only" if best_name == "Multinomial Naive Bayes" else "combined"
    expected_features = int(getattr(best_model_obj, "n_features_in_", X_final_input.shape[1]))

    # Save metadata
    metadata = {
        "best_model_name": best_name,
        "feature_pipeline": feature_pipeline,
        "expected_features": expected_features,
        "best_cv_f1": float(comparison_df.iloc[0]["F1 Score"]),
        "best_cv_accuracy": float(comparison_df.iloc[0]["Accuracy"]),
        "best_cv_roc_auc": float(comparison_df.iloc[0]["ROC-AUC"]),
        "params": str(best_params),
        "trained_models_count": len(comparison_df),
        "vocabulary_size": tfidf_analysis["vocab_size"],
        "training_samples_count": len(full_train_df)
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    # Requirement 11: Print success message and list existing saved artifacts
    print("\nTraining completed successfully.")
    print("Saved artifacts:")
    artifacts_list = [
        MODELS_DIR / "tfidf_vectorizer.joblib",
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "fake_news_model.joblib",
        MODELS_DIR / "word2vec.model",
        MODELS_DIR / "scaler.joblib",
        MODELS_DIR / "metadata.json"
    ]
    for art in artifacts_list:
        if art.exists():
            print(f"- models/{art.name}")

    return best_model_obj, comparison_df

if __name__ == "__main__":
    train_pipeline()
