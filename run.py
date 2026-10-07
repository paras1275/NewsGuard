import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_and_split_data
from src.train import train_pipeline
from src.evaluate import evaluate_model_on_test
from src.explain import generate_shap_summary_plot

def run_training_pipeline():
    print("==================================================")
    print("  NewsGuard — Training & Model Evaluation Pipeline")
    print("==================================================")
    
    print("\nPhase 1: Loading & Splitting Dataset...")
    load_and_split_data()

    print("\nPhase 2-6: Preprocessing, Feature Engineering & Cross-Validation Training...")
    train_pipeline()

    print("\nPhase 7: Final Test Evaluation & Generating Figures...")
    evaluate_model_on_test()

    print("\nPhase 8: Generating SHAP Explainability Visuals...")
    generate_shap_summary_plot()

    print("\n[COMPLETE] NewsGuard Training Pipeline completed successfully!")

def run_flask_api():
    print("Starting NewsGuard Flask REST API...")
    from app.flask_api import app
    app.run(host='0.0.0.0', port=5000, debug=False)

def run_gradio_ui():
    print("Launching NewsGuard Gradio Web Dashboard...")
    from app.gradio_app import launch_app
    launch_app(server_name="0.0.0.0", preferred_port=7860)


def run_tests():
    print("Executing Pytest test suite...")
    subprocess.run(["pytest", "-v"])

def main():
    parser = argparse.ArgumentParser(description="NewsGuard — Master CLI Tool")
    parser.add_argument("--train", action="store_true", help="Execute complete model training & evaluation pipeline")
    parser.add_argument("--eval", action="store_true", help="Execute test set evaluation & update figure plots")
    parser.add_argument("--api", action="store_true", help="Launch Flask REST API server")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio Web Dashboard")
    parser.add_argument("--test", action="store_true", help="Run Pytest automated test suite")


    args = parser.parse_args()

    if args.train:
        run_training_pipeline()
    elif args.eval:
        evaluate_model_on_test()
    elif args.api:
        run_flask_api()
    elif args.ui:
        run_gradio_ui()

    elif args.test:
        run_tests()
    else:
        # Default: Execute full training pipeline
        run_training_pipeline()

if __name__ == "__main__":
    main()
