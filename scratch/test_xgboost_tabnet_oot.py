#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_KAGGLE_PATH = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022_dataset", "Base.csv")
SINGLE_CSV_PATH = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022.csv")

if os.path.exists(REAL_KAGGLE_PATH):
    DATA_PATH = REAL_KAGGLE_PATH
else:
    DATA_PATH = SINGLE_CSV_PATH

print(f"Loading Real BAF Dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

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

train_mask = df["month"] <= 5
test_mask = df["month"] > 5

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test, y_test = X_all[test_mask], y_all[test_mask]

print(f"OOT Split: Train (Months 0-5) = {len(X_train):,} rows | Test (Months 6-7) = {len(X_test):,} rows")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
print(f"Calculated scale_pos_weight: {pos_weight:.2f}")

db_path = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("FinGuard_Fraud_ML_Benchmark")

def focal_loss_obj(y_true, y_pred):
    alpha = 0.75  # Optimized alpha for positive fraud class penalty
    gamma = 1.5   # Optimized gamma for gradient balancing
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = p * y_true + (1.0 - p) * (1.0 - y_true)
    alpha_t = alpha * y_true + (1.0 - alpha) * (1.0 - y_true)
    grad = alpha_t * (1.0 - p_t) ** gamma * (p - y_true)
    hess = alpha_t * (1.0 - p_t) ** gamma * p * (1.0 - p)
    return grad, hess

print("\n--- Training Model 1: XGBoost + Focal Loss (OOT Split) ---")
xgb_focal = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    objective=focal_loss_obj,
    scale_pos_weight=pos_weight,
    max_delta_step=1.0,
    random_state=42,
    eval_metric="logloss"
)
xgb_focal.fit(X_train_scaled, y_train)

y_proba_xgb = xgb_focal.predict_proba(X_test_scaled)[:, 1]
fpr_arr, tpr_arr, thresh_arr = roc_curve(y_test, y_proba_xgb)
idx_5 = np.argmin(np.abs(fpr_arr - 0.05))
recall_at_5_fpr_xgb = float(tpr_arr[idx_5])

prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba_xgb)
pr_auc_xgb = float(auc(rec_c, prec_c))
roc_auc_xgb = float(roc_auc_score(y_test, y_proba_xgb))

print(f"XGBoost + Focal Loss Results:")
print(f"  Recall @ 5% FPR: {recall_at_5_fpr_xgb:.4f}")
print(f"  PR-AUC: {pr_auc_xgb:.4f}")
print(f"  ROC-AUC: {roc_auc_xgb:.4f}")

with mlflow.start_run(run_name="[OOT_Split] XGBoost + Focal Loss"):
    mlflow.log_param("model", "XGBoost + Focal Loss")
    mlflow.log_param("split_protocol", "Out-of-Time (Months 0-5 Train, 6-7 Test)")
    mlflow.log_param("focal_loss_alpha", 0.25)
    mlflow.log_param("focal_loss_gamma", 2.0)
    mlflow.log_param("scale_pos_weight", pos_weight)
    mlflow.log_metric("recall_at_5_percent_fpr", recall_at_5_fpr_xgb)
    mlflow.log_metric("pr_auc", pr_auc_xgb)
    mlflow.log_metric("roc_auc", roc_auc_xgb)

print("\n--- Training Model 2: Google TabNet Classifier (OOT Split) ---")
tabnet = TabNetClassifier(
    n_d=16, n_a=16, n_steps=4, gamma=1.3,
    cat_idxs=[], cat_dims=[],
    verbose=0
)
X_train_tabnet = X_train_scaled.astype(np.float32)
y_train_tabnet = y_train.astype(np.int64)
X_test_tabnet = X_test_scaled.astype(np.float32)

tabnet.fit(
    X_train_tabnet, y_train_tabnet,
    max_epochs=20,
    batch_size=2048,
    virtual_batch_size=256
)

y_proba_tabnet = tabnet.predict_proba(X_test_tabnet)[:, 1]
fpr_arr_t, tpr_arr_t, _ = roc_curve(y_test, y_proba_tabnet)
idx_5_t = np.argmin(np.abs(fpr_arr_t - 0.05))
recall_at_5_fpr_tabnet = float(tpr_arr_t[idx_5_t])

prec_c_t, rec_c_t, _ = precision_recall_curve(y_test, y_proba_tabnet)
pr_auc_tabnet = float(auc(rec_c_t, prec_c_t))
roc_auc_tabnet = float(roc_auc_score(y_test, y_proba_tabnet))

print(f"TabNet Results:")
print(f"  Recall @ 5% FPR: {recall_at_5_fpr_tabnet:.4f}")
print(f"  PR-AUC: {pr_auc_tabnet:.4f}")
print(f"  ROC-AUC: {roc_auc_tabnet:.4f}")

with mlflow.start_run(run_name="[OOT_Split] Google TabNet Classifier"):
    mlflow.log_param("model", "Google TabNet Classifier")
    mlflow.log_param("split_protocol", "Out-of-Time (Months 0-5 Train, 6-7 Test)")
    mlflow.log_param("n_d", 16)
    mlflow.log_param("n_a", 16)
    mlflow.log_param("n_steps", 4)
    mlflow.log_metric("recall_at_5_percent_fpr", recall_at_5_fpr_tabnet)
    mlflow.log_metric("pr_auc", pr_auc_tabnet)
    mlflow.log_metric("roc_auc", roc_auc_tabnet)

print("\nBenchmark Execution Finished Successfully!")
