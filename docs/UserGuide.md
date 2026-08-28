# FinOps-Security-Agent — User Guide

> **Enterprise User & API Guide for Autonomous Financial Decisioning and Security Compliance.**

Welcome to the **FinOps-Security-Agent** User Guide. This guide provides comprehensive instructions on using the Streamlit BI Analytics Dashboard, sending REST API decision requests, configuring approved vendors, and inspecting the cryptographic audit log.

---

## 📋 Table of Contents
1. [Quick Start Guide](#-quick-start-guide)
2. [Streamlit BI Analytics Dashboard](#-streamlit-bi-analytics-dashboard)
   - [Tab 1: Live Interactive Event Simulator](#tab-1-live-interactive-event-simulator)
   - [Tab 2: Batch CSV Ingestion & Audit](#tab-2-batch-csv-ingestion--audit)
   - [Tab 3: MLflow Candidate & PCA Analytics](#tab-3-mlflow-candidate--pca-analytics)
   - [Tab 4: Cryptographic Audit Ledger Inspector](#tab-4-cryptographic-audit-ledger-inspector)
3. [REST API Endpoint Reference](#-rest-api-endpoint-reference)
   - [`POST /decide`](#post-decide)
   - [`GET /audit/verify`](#get-auditverify)
   - [`GET /metrics`](#get-metrics)
   - [`GET /health`](#get-health)
4. [Vendor Master Management](#-vendor-master-management)
5. [FAQ & Troubleshooting](#-faq--troubleshooting)

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.9+ installed
- Terminal access

### Launch Commands
```bash
# 1. Clone repository
git clone https://github.com/PriyankaHichkad/FinOps-Security-Agent.git
cd FinOps-Security-Agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Analytics Dashboard
streamlit run app_analytics.py

# 4. Launch FastAPI REST Server (in a separate terminal window)
uvicorn main:app --reload --port 8000
```

---

## 🖥 Streamlit BI Analytics Dashboard

The Streamlit Dashboard runs live at `http://localhost:8501`.

### Tab 1: Live Interactive Event Simulator
Simulate incoming invoice applications in real time:
- **Inputs**:
  - `Applicant Name` (e.g. `Alice Johnson`)
  - `Email Address` (e.g. `alicejohnson@gmail.com`)
  - `Vendor Name` (Select from `Acme Corp`, `TechData Inc`, `Global Logistics LLC`, `Apex Cloud Systems`, or custom write-in)
  - `Invoice Amount` ($)
  - `PO Number` (e.g. `PO-1001`)
  - `Access Hour` (0-23)
  - `Transaction Notes` (Freeform string)
- **Output Verdict Cards**:
  - 🟢 **`AUTO_APPROVE`**: Transaction meets all policy rules, passes LightGBM fraud risk evaluation, and exhibits normal UEBA behavior.
  - 🟡 **`ROUTE_TO_HUMAN_REVIEW`**: High dollar impact ($10,000+ cap) or unapproved vendor requiring manager sign-off.
  - 🔴 **`AUTO_BLOCK`**: Prompt injection attempt detected (`UNSAFE`) or high ML fraud probability (>0.85).

---

### Tab 2: Batch CSV Ingestion & Audit
Batch process hundreds of financial records at once:
1. Download the **1-Click Sample CSV Template**.
2. Drag and drop your company's CSV file.
3. View the live synthesized batch decision table with row-by-row risk scores and SHA-256 hashes.

---

### Tab 3: MLflow Candidate & PCA Analytics
Inspect model performance and feature compression:
- **MLflow Model Benchmark Table**: Compares **LightGBM**, **XGBoost**, **Random Forest**, and **Logistic Regression** across PR-AUC, ROC-AUC, Precision, Recall, and F1 Score.
- **PCA Scree Plot**: Interactive visualization of individual and cumulative explained variance across principal components.

---

### Tab 4: Cryptographic Audit Ledger Inspector
- View the immutable SHA-256 cryptographic chain ($H_i = \text{SHA256}(H_{i-1} \parallel R_i)$).
- Click **`[ Verify Ledger Integrity ]`** to run real-time cryptographic validation proving no past record was tampered with.

---

## 🔌 REST API Endpoint Reference

FastAPI runs live at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.

### `POST /decide`
Evaluates a single transaction application.

#### Request Example (`curl`):
```bash
curl -X 'POST' \
  'http://localhost:8000/decide' \
  -H 'Content-Type: application/json' \
  -d '{
  "event_id": "EVT-9001",
  "applicant_name": "Sarah Connor",
  "email": "sarah.connor@cyberdyne.com",
  "vendor_name": "Acme Corp",
  "invoice_amount": "$2,450.00",
  "po_number": "PO-1001",
  "velocity_6h": 1,
  "access_hour": 14,
  "notes": "Routine software subscription renewal"
}'
```

#### Response Example:
```json
{
  "event_id": "EVT-9001",
  "final_verdict": "AUTO_APPROVE",
  "risk_level": "LOW_RISK",
  "ml_fraud_probability": 0.0421,
  "name_email_similarity": 0.9412,
  "finops_evidence": {
    "po_matched": true,
    "vendor_status": "APPROVED",
    "cap_check": "IN_POLICY"
  },
  "security_evidence": {
    "prompt_injection_status": "SAFE",
    "ueba_anomaly_score": 0.12
  },
  "audit_hash": "a4b8f72c3d10e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"
}
```

---

### `GET /audit/verify`
Validates the complete SHA-256 hash chain.

#### Request:
```bash
curl -X 'GET' 'http://localhost:8000/audit/verify'
```

#### Response:
```json
{
  "is_valid": true,
  "total_records": 12,
  "status": "TAMPER_EVIDENT_VALIDATED"
}
```

---

## ❓ FAQ & Troubleshooting

> [!NOTE]
> **Q: What happens if an invoice amount has currency formatting like `"$12,500.00"`?**  
> **A**: The system's flexible type sanitizer automatically converts `"$12,500.00"` into float `12500.0` before rule evaluation.

> [!IMPORTANT]
> **Q: How are new vendors onboarded?**  
> **A**: When an invoice for an unapproved vendor arrives, the system routes it to `ROUTE_TO_HUMAN_REVIEW`. Upon manager approval, `tool_register_new_vendor()` dynamically registers the vendor into `data/vendor_master.json`.
