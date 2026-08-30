# FinOps-Security-Agent

> **Enterprise Multi-Agent FinOps & Security Compliance REST Microservice with NeurIPS 2022 ML Risk Scoring and SHA-256 Audit Logging.**

📖 **Documentation Quick Links**:
- 📘 [**User & REST API Guide**](docs/UserGuide.md) — Comprehensive guide on REST API endpoints (`/decide`, `/audit/verify`, `/metrics`), JSON payloads, and vendor master rules.
- 📙 [**Developer & Architecture Guide**](docs/DeveloperGuide.md) — System architecture, Mermaid sequence diagrams, multi-agent state machine designs, and trade-off rationales.

---

An enterprise-grade autonomous decisioning microservice that integrates **NeurIPS 2022 Bank Account Fraud (BAF)** tabular ML risk scoring, **Financial Operations (FinOps)** deterministic policy rules, **Security & Compliance (SecOps)** UEBA anomaly detection, and a **tamper-evident SHA-256 cryptographic audit ledger**.

---

## 📌 System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Sanitization Layer"]
        InputEvent["Incoming Invoice / Application API Event"]
        BAFData["NeurIPS 2022 BAF Benchmark Data"]
    end

    subgraph CoreEngine ["2. Modular Agent Engines (src/)"]
        PCAReduction["src/ml_engine.py: PCA Scree Plot & LightGBM Fraud Scoring"]
        FinOpsPolicy["src/finops_agent.py: Field Extractor, PO Match & Policy Rules"]
        SecOpsAnomaly["src/security_agent.py: UEBA Anomaly Scoring & Injection Defense"]
    end

    subgraph OrchestratorLayer ["3. State Machine & Audit Ledger"]
        Orchestrator["src/orchestrator.py: Multi-Agent Verdict Synthesizer"]
        FinalVerdict{"Final Decision Verdict"}
        HashChain["SHA-256 Cryptographic Hash Chain Audit Ledger"]
    end

    subgraph ServingLayer ["4. REST API Serving & Microservice"]
        FastAPI["main.py: FastAPI REST Server (/decide, /audit/verify, /metrics)"]
        SwaggerDocs["OpenAPI / Swagger Interactive Documentation (/docs)"]
    end

    InputEvent & BAFData --> PCAReduction & FinOpsPolicy & SecOpsAnomaly
    PCAReduction & FinOpsPolicy & SecOpsAnomaly --> Orchestrator --> FinalVerdict

    FinalVerdict -->|Clean & In-Policy| AutoApprove["AUTO_APPROVE"]
    FinalVerdict -->|Hard Risk Violation| AutoBlock["AUTO_BLOCK"]
    FinalVerdict -->|High Impact / Low Conf| HumanQueue["ROUTE_TO_HUMAN_REVIEW"]

    AutoApprove & AutoBlock & HumanQueue --> HashChain --> FastAPI --> SwaggerDocs
```

---

## 🧠 Core Mechanics & Mathematical Foundations

### 1. NeurIPS 2022 PCA Variance Engine (`src/ml_engine.py`)
- **Covariance Matrix Calculation**: $\Sigma = \frac{1}{n-1} X^T X$
- **Eigen Decomposition**: Computes Eigenvalues $\lambda_i$ and Eigenvectors $v_i$.
- **Explained Variance Ratio ($EVR_i$)**: $EVR_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$
- **95% Cumulative Variance Thresholding**: Compresses 24 tabular applicant features into 19 principal orthogonal components (96.21% cumulative variance retained), enabling **sub-10ms model inference**.

### 2. Deterministic Rules vs. Probabilistic Judgment (`src/finops_agent.py`)
- **Exact Policy Rules**: Spending caps (User/Account assigned spending limit), Purchase Order (PO) matching, and duplicate payment detection are evaluated in exact Python code.
- **Dynamic Feature Calculation**: Automatically calculates applicant `name_email_similarity` on the fly using string ratio algorithms.

### 3. Tamper-Evident SHA-256 Audit Ledger (`src/orchestrator.py`)
- Every decision record $R_i$ is cryptographically linked using SHA-256 hashing:
  $$H_i = \text{SHA256}(H_{i-1} \parallel R_i)$$
- Verifies ledger integrity via `/audit/verify` to prove past logs were never retroactively edited.

### 4. FinOps Financial Backtesting Loss Simulator (`src/backtest_engine.py`)
- **Inspired by Yves Hilpisch** (*Artificial Intelligence in Finance*, Ch. 10 & 11).
- Evaluates economic net dollar savings ($) across decision thresholds $\tau \in [0.05, 0.95]$:
  $$\text{Net Dollars Saved}(\tau) = \Big( TP(\tau) \times \$2,500 \Big) - \Big( FP(\tau) \times \$25 \Big) - \Big( N_{\text{test}} \times \$0.05 \Big)$$
- **Backtest Result**: Achieves **`$310,056.75` in Net Savings** (**41.90% ROI cost reduction**) at the optimal economic threshold $\tau^* = 0.95$.

---

## 🛠 Directory Structure

```
FinOps-Security-Agent/
├── src/
│   ├── __init__.py           # Package Initialization
│   ├── logger.py             # Logging & Custom Exception Handler
│   ├── ml_engine.py          # NeurIPS 2022 ML Engine, PCA Variance & LightGBM Scoring
│   ├── backtest_engine.py    # FinOps Financial Backtesting & Loss Simulator (Ch. 10 & 11)
│   ├── finops_agent.py       # FinOps Field Extractor, PO Reconciliation & Rules
│   ├── security_agent.py     # SecOps UEBA Anomaly Scoring & Prompt Injection Sanitizer
│   └── orchestrator.py       # Decision State Machine & SHA-256 Audit Hash Chain
├── data/
│   ├── BAF_NeurIPS_2022.csv  # NeurIPS 2022 Benchmark Dataset
│   └── vendor_master.json    # Approved Vendor Reference Database
├── artifacts/                # Saved Model (.pkl) & Metrics (.json)
├── tests/
│   └── test_finguard.py      # 10 Passing PyTest Unit Tests
├── main.py                   # FastAPI REST Server (/decide, /audit/verify, /metrics)
├── Dockerfile                # Container Deployment Blueprint
├── pyproject.toml            # Package Metadata
├── requirements.txt          # Dependencies
└── README.md                 # System Documentation
```

---

## ⚡ Quick Start Guide

### 1. Installation & Virtual Environment
```bash
git clone https://github.com/PriyankaHichkad/FinOps-Security-Agent.git
cd FinOps-Security-Agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute PyTest Automated Suite
```bash
pytest tests/ -v
```

### 3. Run Reproducible DVC Pipeline
```bash
dvc repro
```

### 4. Launch FastAPI REST Microservice Server
```bash
uvicorn main:app --reload --port 8000
```
* Access Interactive OpenAPI / Swagger Documentation live at `http://localhost:8000/docs`.

---

## 📊 Model Performance Matrix (NeurIPS 2022 Out-of-Time Temporal Benchmark)

All candidate architectures were benchmarked under **Out-of-Time (OOT) Temporal Splitting** (Months 0–5 for Training, Months 6–7 for Out-of-Time Testing) across 26 MLflow experiment runs on the real Kaggle NeurIPS 2022 dataset (`Base.csv` — 1,000,000 rows):

| Sampling Strategy & Architecture | Recall @ 5% FPR ⭐ *(NeurIPS Metric)* | PR-AUC | ROC-AUC | Age Fairness FPR Ratio | Serving Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`[SMOTE_1to1]` LightGBM** | **`23.65%`** | 🏆 **`0.1063`** | **`0.7079`** | **`2.02x`** | **< 8ms** | 🏆 **Production Champion Model** |
| **`[Random_Undersample]` LightGBM** | `22.64%` | `0.1056` | `0.7247` | `1.60x` | < 5ms | Candidate |
| **`[Hybrid_1to3_Optimal]` LightGBM** | `24.32%` | `0.0978` | `0.7138` | `2.06x` | < 8ms | Candidate |
| **`[Baseline_Natural]` LightGBM** | `25.68%` | `0.0738` | `0.7141` | `2.17x` | < 8ms | Candidate |
| **`[Baseline_Natural]` Logistic Regression** | `22.64%` | `0.0664` | `0.6990` | `1.59x` | < 2ms | Candidate |
| **`[Baseline_Natural]` SVM** | `22.64%` | `0.0662` | `0.6999` | `1.61x` | < 2ms | Candidate |
| **`[Baseline_Natural]` Random Forest** | `20.27%` | `0.0572` | `0.7174` | `1.70x` | < 12ms | Candidate |
| **`[Baseline_Natural]` XGBoost** | `22.30%` | `0.0558` | `0.7187` | `1.70x` | < 10ms | Candidate |

---

## 🔬 Multi-Variant Cross-Domain Stress Test (Variants I – V)

The Production Champion Model was stress-tested across all 6 challenge dataset variants in `data/BAF_NeurIPS_2022_dataset/`:

| Dataset Variant | Challenge Type | Recall @ 5% FPR | PR-AUC | ROC-AUC | Generalization Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Base.csv`** | Representative baseline | **`24.39%`** | **`0.0874`** | **`0.7181`** | Baseline |
| **`Variant I.csv`** | Higher demographic group disparity | **`21.54%`** | **`0.0701`** | **`0.6980`** | Robust |
| **`Variant II.csv`** | Higher prevalence disparity | **`26.53%`** | **`0.1118`** | **`0.7264`** | Robust |
| **`Variant III.csv`** | High group separability | **`22.30%`** | **`0.0814`** | **`0.6943`** | Robust |
| **`Variant IV.csv`** | Train prevalence shift | **`26.56%`** | **`0.1150`** | **`0.7286`** | Robust |
| **`Variant V.csv`** | Train separability shift | **`20.55%`** | **`0.0888`** | **`0.7110`** | Robust |

---

## 🛠 Tools & Tech Stack
- [CI/CD Pipeline](https://github.com/PriyankaHichkad/FinOps-Security-Agent/actions)
- [Python 3.13+](https://www.python.org/)
- [MLflow Experimentation & Model Registry](https://mlflow.org/)
- [DVC (Data Version Control) Remote Storage](https://dagshub.com/PriyankaHichkad/FinOps-Security-Agent.dvc)
- [NeurIPS 2022 Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
