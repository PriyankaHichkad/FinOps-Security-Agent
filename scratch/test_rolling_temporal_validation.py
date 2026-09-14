#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_KAGGLE_PATH = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022_dataset", "Base.csv")
SINGLE_CSV_PATH = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022.csv")

DATA_PATH = REAL_KAGGLE_PATH if os.path.exists(REAL_KAGGLE_PATH) else SINGLE_CSV_PATH

print(f"Loading Real BAF Dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Total Dataset shape: {df.shape}")

FEATURE_COLUMNS = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4week", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "keep_alive_session",
    "device_distinct_emails_8w", "device_fraud_count", "month"
]

v6 = df["velocity_6h"] if "velocity_6h" in df.columns else 0.0
v24 = df["velocity_24h"] if "velocity_24h" in df.columns else 0.0
v4w = df["velocity_4week"] if "velocity_4week" in df.columns else 0.0
inc = df["income"] if "income" in df.columns else 0.5
credit = df["proposed_credit_limit"] if "proposed_credit_limit" in df.columns else 500.0
bank_m = df["bank_months_count"] if "bank_months_count" in df.columns else 12.0
age = df["customer_age"] if "customer_age" in df.columns else 35.0
device_fraud = df["device_fraud_count"] if "device_fraud_count" in df.columns else 0.0
dob_emails = df["date_of_birth_distinct_emails_4w"] if "date_of_birth_distinct_emails_4w" in df.columns else 0.0
email_free = df["email_is_free"] if "email_is_free" in df.columns else 0.0

df["velocity_acceleration_6h_24h"] = v6 / (v24 + 1.0)
df["velocity_acceleration_24h_4w"] = v24 / (v4w + 1.0)
df["credit_to_income_ratio"] = credit / (inc + 0.01)
df["bank_tenure_to_age_ratio"] = bank_m / (age * 12.0 + 1.0)
df["risk_interaction_score"] = device_fraud * v6
df["dob_email_risk"] = dob_emails * (email_free + 1.0)

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "velocity_acceleration_6h_24h", "velocity_acceleration_24h_4w",
    "credit_to_income_ratio", "bank_tenure_to_age_ratio",
    "risk_interaction_score", "dob_email_risk"
]

existing_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
X_all = df[existing_cols].copy().fillna(df[existing_cols].median(numeric_only=True))
y_all = df["fraud_bool"].values
months = df["month"].values

db_path = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("FinGuard_Fraud_ML_Benchmark")

def focal_loss_obj(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = p * y_true + (1.0 - p) * (1.0 - y_true)
    alpha_t = alpha * y_true + (1.0 - alpha) * (1.0 - y_true)
    grad = alpha_t * (1.0 - p_t) ** gamma * (p - y_true)
    hess = alpha_t * (1.0 - p_t) ** gamma * p * (1.0 - p)
    return grad, hess

def evaluate_predictions(y_true, y_proba):
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
    idx_5 = np.argmin(np.abs(fpr_arr - 0.05))
    recall_at_5_fpr = float(tpr_arr[idx_5])
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_val = float(auc(rec_c, prec_c))
    roc_auc_val = float(roc_auc_score(y_true, y_proba))
    return recall_at_5_fpr, pr_auc_val, roc_auc_val

windows = [
    {"name": "Window 1 (Train 0-4 -> Test M5)", "train_months": [0,1,2,3,4], "test_months": [5]},
    {"name": "Window 2 (Train 0-5 -> Test M6)", "train_months": [0,1,2,3,4,5], "test_months": [6]},
    {"name": "Window 3 (Train 0-6 -> Test M7)", "train_months": [0,1,2,3,4,5,6], "test_months": [7]},
    {"name": "Full OOT (Train 0-5 -> Test M6-7)", "train_months": [0,1,2,3,4,5], "test_months": [6,7]}
]

rolling_results = []

for w in windows:
    w_name = w["name"]
    train_mask = np.isin(months, w["train_months"])
    test_mask = np.isin(months, w["test_months"])

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    n_train_frauds = int(np.sum(y_train == 1))
    n_test_frauds = int(np.sum(y_test == 1))

    print(f"\n========================================================")
    print(f"📊 {w_name}")
    print(f"   Train: {len(X_train):,} rows ({n_train_frauds:,} frauds) | Test: {len(X_test):,} rows ({n_test_frauds:,} frauds)")
    print(f"========================================================")

    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    pos_w = (len(y_train) - n_train_frauds) / max(n_train_frauds, 1)

    # 1. XGBoost + Focal Loss
    xgb_focal = XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        objective=focal_loss_obj, scale_pos_weight=pos_w, max_delta_step=1.0,
        random_state=42, eval_metric="logloss"
    )
    xgb_focal.fit(X_tr_sc, y_train)
    p_xgb_focal = xgb_focal.predict_proba(X_te_sc)[:, 1]
    rec5_xf, pr_xf, roc_xf = evaluate_predictions(y_test, p_xgb_focal)

    print(f"  XGBoost + Focal Loss : Recall@5%FPR={rec5_xf:.4f} | PR-AUC={pr_xf:.4f} | ROC-AUC={roc_xf:.4f}")

    # 2. Standard XGBoost (Logloss)
    xgb_std = XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        scale_pos_weight=pos_w, random_state=42, eval_metric="logloss"
    )
    xgb_std.fit(X_tr_sc, y_train)
    p_xgb_std = xgb_std.predict_proba(X_te_sc)[:, 1]
    rec5_xs, pr_xs, roc_xs = evaluate_predictions(y_test, p_xgb_std)

    print(f"  Standard XGBoost     : Recall@5%FPR={rec5_xs:.4f} | PR-AUC={pr_xs:.4f} | ROC-AUC={roc_xs:.4f}")

    # 3. LightGBM
    lgb = LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
        scale_pos_weight=pos_w, random_state=42, verbosity=-1
    )
    lgb.fit(X_tr_sc, y_train)
    p_lgb = lgb.predict_proba(X_te_sc)[:, 1]
    rec5_lgb, pr_lgb, roc_lgb = evaluate_predictions(y_test, p_lgb)

    print(f"  LightGBM             : Recall@5%FPR={rec5_lgb:.4f} | PR-AUC={pr_lgb:.4f} | ROC-AUC={roc_lgb:.4f}")

    # 4. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_train)
    p_lr = lr.predict_proba(X_te_sc)[:, 1]
    rec5_lr, pr_lr, roc_lr = evaluate_predictions(y_test, p_lr)

    print(f"  Logistic Regression : Recall@5%FPR={rec5_lr:.4f} | PR-AUC={pr_lr:.4f} | ROC-AUC={roc_lr:.4f}")

    w_res = {
        "window": w_name,
        "test_size": len(X_test),
        "test_frauds": n_test_frauds,
        "xgb_focal": {"recall_5_fpr": rec5_xf, "pr_auc": pr_xf, "roc_auc": roc_xf},
        "xgb_std": {"recall_5_fpr": rec5_xs, "pr_auc": pr_xs, "roc_auc": roc_xs},
        "lightgbm": {"recall_5_fpr": rec5_lgb, "pr_auc": pr_lgb, "roc_auc": roc_lgb},
        "logistic_regression": {"recall_5_fpr": rec5_lr, "pr_auc": pr_lr, "roc_auc": roc_lr}
    }
    rolling_results.append(w_res)

    with mlflow.start_run(run_name=f"[Rolling_CV] XGBoost Focal Loss ({w_name})"):
        mlflow.log_param("window", w_name)
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("test_frauds", n_test_frauds)
        mlflow.log_metric("recall_at_5_percent_fpr", rec5_xf)
        mlflow.log_metric("pr_auc", pr_xf)
        mlflow.log_metric("roc_auc", roc_xf)

summary_path = os.path.join(BASE_DIR, "artifacts", "rolling_cv_summary.json")
with open(summary_path, "w") as f:
    json.dump(rolling_results, f, indent=2)

print("\n========================================================")
print("✅ Rolling Temporal Validation Complete! Summary saved to artifacts/rolling_cv_summary.json")
print("========================================================")
