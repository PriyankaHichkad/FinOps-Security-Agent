#!/usr/bin/env python3
"""
FinOps-Security-Agent — Financial Backtesting & Loss Simulator Engine
Inspired by Yves Hilpisch (Artificial Intelligence in Finance, Ch. 10 & 11).
Simulates dollar savings ($) achieved by the Champion ML model vs unmitigated baselines.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.logger import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "champion_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "artifacts", "pca_transformer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022_dataset", "Base.csv")
BACKTEST_RESULTS_PATH = os.path.join(BASE_DIR, "artifacts", "backtest_results.json")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

class FinOpsBacktestEngine:
    """
    Event-based financial loss simulator. Evaluates economic net dollar savings ($)
    across decision thresholds to determine the optimal economic decision threshold.
    """
    def __init__(self, avg_fraud_loss=2500.0, false_positive_cost=25.0, decision_cost=0.05):
        self.avg_fraud_loss = avg_fraud_loss
        self.false_positive_cost = false_positive_cost
        self.decision_cost = decision_cost
        self.load_artifacts()

    def load_artifacts(self):
        """Loads trained Champion model, scaler, and PCA transformer."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.pca = joblib.load(PCA_PATH)
            logger.info("Successfully loaded Champion Model, Scaler, and PCA for Financial Backtesting.")
        except Exception as e:
            logger.error(f"Error loading artifacts for backtesting: {e}")
            self.model = None

    def run_backtest(self, sample_size=100000):
        """Runs vectorized event-based backtesting on Out-of-Time test data."""
        if self.model is None:
            logger.error("Champion model not loaded. Aborting backtest.")
            return {}

        if not os.path.exists(DATA_PATH):
            raise FinGuardException(f"Real NeurIPS 2022 dataset Base.csv is required for backtesting. File not found at: {DATA_PATH}")

        logger.info(f"Reading real NeurIPS 2022 dataset for Financial Backtest simulation ({DATA_PATH})...")
        df = pd.read_csv(DATA_PATH)

        if len(df) > sample_size and "fraud_bool" in df.columns:
            df = df.groupby("fraud_bool", group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=42)
            )

        # Apply Out-of-Time split (Months 6-7 for testing)
        if "month" in df.columns and len(df[df["month"] > 5]) > 0:
            test_df = df[df["month"] > 5].copy()
        else:
            test_df = df.sample(frac=0.2, random_state=42).copy()

        from src.ml_engine import _engineer_ratio_features, ALL_FEATURE_COLUMNS
        test_df = _engineer_ratio_features(test_df)
        expected_cols = list(self.scaler.feature_names_in_) if hasattr(self.scaler, "feature_names_in_") else ALL_FEATURE_COLUMNS
        for c in expected_cols:
            if c not in test_df.columns:
                test_df[c] = 0.0

        X_test = test_df[expected_cols].fillna(test_df[expected_cols].median(numeric_only=True))
        y_test = test_df["fraud_bool"].values

        X_sc = self.scaler.transform(X_test)
        X_pca = self.pca.transform(X_sc)

        y_proba = self.model.predict_proba(X_pca)[:, 1]

        # Calculate Unmitigated Exposure Loss (0% fraud caught)
        total_fraud_incidents = np.sum(y_test == 1)
        unmitigated_baseline_loss = total_fraud_incidents * self.avg_fraud_loss

        threshold_grid = np.linspace(0.05, 0.95, 19)
        simulation_results = []
        best_net_savings = -float("inf")
        optimal_threshold = 0.50
        optimal_metrics = {}

        for tau in threshold_grid:
            y_pred = (y_proba >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            gross_savings = tp * self.avg_fraud_loss
            fp_investigation_cost = fp * self.false_positive_cost
            execution_cost = len(y_test) * self.decision_cost

            net_dollars_saved = gross_savings - fp_investigation_cost - execution_cost
            roi_percentage = (net_dollars_saved / max(unmitigated_baseline_loss, 1.0)) * 100

            res_entry = {
                "threshold": round(float(tau), 2),
                "true_positives_caught": int(tp),
                "false_positives_flagged": int(fp),
                "uncaught_fraud_fn": int(fn),
                "gross_fraud_prevented_usd": round(float(gross_savings), 2),
                "false_alarm_investigation_cost_usd": round(float(fp_investigation_cost), 2),
                "net_dollars_saved_usd": round(float(net_dollars_saved), 2),
                "roi_percentage": round(float(roi_percentage), 2)
            }
            simulation_results.append(res_entry)

            if net_dollars_saved > best_net_savings:
                best_net_savings = net_dollars_saved
                optimal_threshold = round(float(tau), 2)
                optimal_metrics = res_entry

        summary = {
            "total_test_transactions": len(y_test),
            "total_fraud_incidents": int(total_fraud_incidents),
            "unmitigated_baseline_exposure_usd": round(float(unmitigated_baseline_loss), 2),
            "optimal_economic_threshold": optimal_threshold,
            "optimal_net_dollars_saved_usd": round(float(best_net_savings), 2),
            "optimal_roi_percentage": optimal_metrics.get("roi_percentage", 0.0),
            "optimal_metrics": optimal_metrics,
            "threshold_grid_simulation": simulation_results
        }

        os.makedirs(os.path.dirname(BACKTEST_RESULTS_PATH), exist_ok=True)
        with open(BACKTEST_RESULTS_PATH, "w") as f:
            json.dump(summary, f, indent=2)

        # Log Backtesting Results to MLflow
        if HAS_MLFLOW and mlflow:
            try:
                db_path = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
                mlflow.set_tracking_uri(f"sqlite:///{db_path}")
                mlflow.set_experiment("FinGuard_Fraud_ML_Benchmark")
                mlflow.end_run()
                with mlflow.start_run(run_name="💰_FINANCIAL_BACKTEST_SIMULATOR"):
                    mlflow.set_tag("stage", "Economic_Backtest")
                    mlflow.log_param("avg_fraud_loss_usd", self.avg_fraud_loss)
                    mlflow.log_param("false_positive_cost_usd", self.false_positive_cost)
                    mlflow.log_param("optimal_economic_threshold", optimal_threshold)
                    mlflow.log_metric("unmitigated_baseline_exposure_usd", unmitigated_baseline_loss)
                    mlflow.log_metric("optimal_net_dollars_saved_usd", best_net_savings)
                    mlflow.log_metric("optimal_roi_percentage", optimal_metrics.get("roi_percentage", 0.0))
                    mlflow.log_artifact(BACKTEST_RESULTS_PATH)
                mlflow.end_run()
                logger.info("Successfully logged Financial Backtest simulation to MLflow.")
            except Exception as e_ml:
                logger.warning(f"MLflow backtest logging notice: {e_ml}")

        return summary

def main():
    print("=" * 70)
    print("💰 FinOps-Security-Agent — Financial Backtest Loss Simulator")
    print("Inspired by Yves Hilpisch (Artificial Intelligence in Finance)")
    print("=" * 70)

    engine = FinOpsBacktestEngine()
    results = engine.run_backtest()

    if results:
        print(f"\nTotal Test Transactions Analyzed : {results['total_test_transactions']:,}")
        print(f"Total Uncaught Baseline Exposure : ${results['unmitigated_baseline_exposure_usd']:,.2f}")
        print(f"Optimal Economic Threshold (τ*)   : {results['optimal_economic_threshold']}")
        print(f"Optimal Net Dollars Saved ($)    : ${results['optimal_net_dollars_saved_usd']:,.2f}")
        print(f"Return on Investment (ROI %)     : {results['optimal_roi_percentage']:.2f}%")
        print("\nThreshold Grid Financial Backtest Summary:")
        print("-" * 70)
        for r in results['threshold_grid_simulation'][::3]:
            print(f"Threshold τ={r['threshold']:<4} | Net Saved: ${r['net_dollars_saved_usd']:<12,.2f} | Caught: {r['true_positives_caught']:<5} | False Alarms: {r['false_positives_flagged']:<5}")
        print("=" * 70)

if __name__ == "__main__":
    main()
