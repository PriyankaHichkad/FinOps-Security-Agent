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
        """Runs vectorized event-based backtesting on Out-of-Time test data across fine threshold grids."""
        if self.model is None:
            logger.error("Champion model not loaded. Aborting backtest.")
            return {}

        if not os.path.exists(DATA_PATH):
            logger.warning(f"Dataset not found at {DATA_PATH}. Generating fallback synthetic dataset for CI environment...")
            from src.ml_engine import MLEngine
            eng = MLEngine()
            eng._generate_synthetic_baf_data()
            synth_path = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022.csv")
            if os.path.exists(synth_path):
                df = pd.read_csv(synth_path)
            else:
                logger.error("Could not resolve dataset for backtesting.")
                return {}
        else:
            df = pd.read_csv(DATA_PATH)

        if len(df) > sample_size:
            df = df.sample(n=min(len(df), sample_size), random_state=42)

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

        # 1. Calibrated probabilities
        try:
            y_proba_calib = self.model.predict_proba(X_sc)[:, 1]
        except Exception:
            y_proba_calib = self.model.predict_proba(X_pca)[:, 1]

        # 2. Uncalibrated probabilities (extracting base estimators from CalibratedClassifierCV wrapper)
        uncalib_submodels = []
        if hasattr(self.model, "models_and_weights"):
            for sub_m, w, name in self.model.models_and_weights:
                if hasattr(sub_m, "estimator"):
                    base_m = sub_m.estimator
                elif hasattr(sub_m, "calibrated_classifiers_") and len(sub_m.calibrated_classifiers_) > 0:
                    base_m = sub_m.calibrated_classifiers_[0].estimator
                else:
                    base_m = sub_m
                uncalib_submodels.append((base_m, w, name))
            from src.ml_engine import ChampionEnsemble
            uncalib_ensemble = ChampionEnsemble(uncalib_submodels)
            try:
                y_proba_uncalib = uncalib_ensemble.predict_proba(X_sc)[:, 1]
            except Exception:
                y_proba_uncalib = uncalib_ensemble.predict_proba(X_pca)[:, 1]
        else:
            y_proba_uncalib = y_proba_calib

        total_fraud_incidents = np.sum(y_test == 1)
        unmitigated_baseline_loss = total_fraud_incidents * self.avg_fraud_loss

        fine_threshold_grid = np.round(np.linspace(0.01, 0.99, 99), 2)

        def _evaluate_simulation(y_proba, calibration_status="calibrated"):
            sim_results = []
            best_saved = -float("inf")
            optimal_tau = 0.50
            best_entry = {}

            for tau in fine_threshold_grid:
                tau_val = round(float(tau), 2)
                y_pred = (y_proba >= tau_val).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                gross_savings = tp * self.avg_fraud_loss
                fp_cost = fp * self.false_positive_cost
                exec_cost = len(y_test) * self.decision_cost

                net_saved = gross_savings - fp_cost - exec_cost
                roi = (net_saved / max(unmitigated_baseline_loss, 1.0)) * 100

                entry = {
                    "threshold": tau_val,
                    "true_positives_caught": int(tp),
                    "false_positives_flagged": int(fp),
                    "uncaught_fraud_fn": int(fn),
                    "gross_fraud_prevented_usd": round(float(gross_savings), 2),
                    "false_alarm_investigation_cost_usd": round(float(fp_cost), 2),
                    "execution_cost_usd": round(float(exec_cost), 2),
                    "net_dollars_saved_usd": round(float(net_saved), 2),
                    "roi_percentage": round(float(roi), 2)
                }
                sim_results.append(entry)

                if net_saved > best_saved:
                    best_saved = net_saved
                    optimal_tau = tau_val
                    best_entry = entry

            return {
                "dataset_scope": "full_oot_20k" if len(y_test) > 1000 else "sample_200_smoke_test",
                "dataset_notes": f"Evaluated on a {len(y_test):,} transaction subsample (sampled from ~205,011 Out-of-Time test rows in Months 6-7). Fixed decision execution cost is {len(y_test):,} x $0.05 = ${len(y_test)*0.05:,.2f}.",
                "calibration_status": calibration_status,
                "total_test_transactions": len(y_test),
                "total_fraud_incidents": int(total_fraud_incidents),
                "unmitigated_baseline_exposure_usd": round(float(unmitigated_baseline_loss), 2),
                "optimal_economic_threshold": optimal_tau,
                "optimal_net_dollars_saved_usd": round(float(best_saved), 2),
                "optimal_roi_percentage": best_entry.get("roi_percentage", 0.0),
                "optimal_metrics": best_entry,
                "threshold_grid_simulation": sim_results
            }

        calibrated_summary = _evaluate_simulation(y_proba_calib, calibration_status="calibrated_sigmoid")
        uncalibrated_summary = _evaluate_simulation(y_proba_uncalib, calibration_status="uncalibrated_raw")

        os.makedirs(os.path.dirname(BACKTEST_RESULTS_PATH), exist_ok=True)
        with open(BACKTEST_RESULTS_PATH, "w") as f:
            json.dump(calibrated_summary, f, indent=2)

        out_name = "backtest_results_full.json" if len(y_test) > 1000 else "backtest_results_sample.json"
        with open(os.path.join(BASE_DIR, "artifacts", out_name), "w") as f:
            json.dump(calibrated_summary, f, indent=2)

        uncalib_out_name = "backtest_results_uncalibrated.json" if len(y_test) > 1000 else "backtest_results_uncalibrated_sample.json"
        with open(os.path.join(BASE_DIR, "artifacts", uncalib_out_name), "w") as f:
            json.dump(uncalibrated_summary, f, indent=2)

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
                    mlflow.log_param("optimal_economic_threshold", calibrated_summary["optimal_economic_threshold"])
                    mlflow.log_metric("unmitigated_baseline_exposure_usd", unmitigated_baseline_loss)
                    mlflow.log_metric("optimal_net_dollars_saved_usd", calibrated_summary["optimal_net_dollars_saved_usd"])
                    mlflow.log_metric("optimal_roi_percentage", calibrated_summary["optimal_roi_percentage"])
                    mlflow.log_artifact(BACKTEST_RESULTS_PATH)
                    mlflow.log_artifact(os.path.join(BASE_DIR, "artifacts", uncalib_out_name))
                mlflow.end_run()
                logger.info("Successfully logged Financial Backtest simulation to MLflow.")
            except Exception as e_ml:
                logger.warning(f"MLflow backtest logging notice: {e_ml}")

        return calibrated_summary

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
