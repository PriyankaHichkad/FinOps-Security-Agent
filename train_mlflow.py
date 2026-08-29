#!/usr/bin/env python3
"""
FinOps-Security-Agent — Standalone MLflow Experimentation & Training Pipeline
Executes multi-model training (LightGBM, XGBoost, Random Forest, Logistic Regression),
logs hyperparameters, PR-AUC / ROC-AUC metrics, and promotes the Champion Model to MLflow.
"""

import sys
from src.ml_engine import MLEngine, logger

def main():
    print("=" * 70)
    print("🚀 FinOps-Security-Agent — MLflow Experimentation Pipeline")
    print("=" * 70)
    
    print("\n1. Initializing ML Engine...")
    import os
    import mlflow
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(base_dir, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("FinGuard_Fraud_ML_Benchmark")
    engine = MLEngine()

    print("\n2. Regenerating NeurIPS 2022 Benchmark Dataset & Retraining Pipeline...")
    engine._generate_synthetic_baf_data()
    engine.train_pipeline()

    print("\n" + "=" * 70)
    print("📊 MLFLOW EXPERIMENT COMPARISON MATRIX")
    print("=" * 70)
    for model in engine.comparison_matrix:
        print(f"Model: {model['model_name']:<22} | PR-AUC: {model['pr_auc']:<6} | ROC-AUC: {model['roc_auc']:<6} | Status: {model['status']}")

    print("\n" + "=" * 70)
    print("🏆 PRODUCTION CHAMPION MODEL HYPERPARAMETERS")
    print("=" * 70)
    champ_name = type(engine.model).__name__
    print(f"Architecture: {champ_name}")
    print(f"Retained PCA Components: {engine.pca_metrics.get('retained_components')} (Cumulative Variance: {engine.pca_metrics.get('cumulative_variance_explained')*100:.2f}%)")
    print(f"Champion PR-AUC: {engine.pca_metrics.get('pr_auc'):.4f}")
    
    if hasattr(engine.model, "get_params"):
        print("\nTrained Hyperparameters:")
        for k, v in engine.model.get_params().items():
            print(f"  • {k:<25}: {v}")

    print("\n" + "=" * 70)
    print("🌐 HOW TO VIEW EXPERIMENTS IN MLFLOW UI")
    print("=" * 70)
    print("Run the following command in your terminal to view the interactive dashboard:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("\nThen open your browser at: http://localhost:5000")
    print("You will see the 'FinGuard_Fraud_ML_Benchmark' experiment with candidate comparisons & champion hyperparameters!")
    print("=" * 70)

if __name__ == "__main__":
    main()
