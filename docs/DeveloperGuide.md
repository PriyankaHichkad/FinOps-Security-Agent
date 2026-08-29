# FinOps-Security-Agent — Developer Guide

> **Enterprise Software Architecture, Component Design, Sequence Diagrams, and NFR Specifications.**

---

## 📋 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Component Architecture & Design](#-component-architecture--design)
   - [1. Data Ingestion & Sanitization Layer](#1-data-ingestion--sanitization-layer)
   - [2. NeurIPS 2022 ML & PCA Engine (`src/ml_engine.py`)](#2-neurips-2022-ml--pca-engine-srcml_enginepy)
   - [3. Financial Operations Agent (`src/finops_agent.py`)](#3-financial-operations-agent-srcfinops_agentpy)
   - [4. Security & Compliance Agent (`src/security_agent.py`)](#4-security--compliance-agent-srcsecurity_agentpy)
   - [5. Decision Orchestrator & Audit Ledger (`src/orchestrator.py`)](#5-decision-orchestrator--audit-ledger-srcorchestratorpy)
3. [Sequence Diagrams](#-sequence-diagrams)
   - [Transaction Event Decisioning Workflow](#transaction-event-decisioning-workflow)
   - [Dynamic Vendor Onboarding Workflow](#dynamic-vendor-onboarding-workflow)
4. [Design Decisions & Trade-off Rationales](#-design-decisions--trade-off-rationales)
5. [Non-Functional Requirements (NFRs)](#-non-functional-requirements-nfrs)
6. [Testing & Verification Guide](#-testing--verification-guide)

---

## 🏗 Architecture Overview

The system is designed as a **decoupled multi-agent micro-architecture** composed of specialized decision modules, an orchestration engine, and a cryptographic hash chain logger.

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & API Interface"]
        StreamlitUI["app.py: Streamlit Dashboard"]
        FastAPIEndpoint["main.py: REST Server (/decide)"]
    end

    subgraph CoreEngine ["Core Decisioning Pipeline (src/)"]
        Orchestrator["src/orchestrator.py: Multi-Agent Synthesizer"]
        MLEngine["src/ml_engine.py: NeurIPS LightGBM & PCA Engine"]
        FinOpsAgent["src/finops_agent.py: Policy Rules & PO Reconciliation"]
        SecOpsAgent["src/security_agent.py: UEBA & Prompt Injection Defense"]
    end

    subgraph DataStorage ["Data & Cryptographic Audit Layer"]
        VendorMaster["data/vendor_master.json: Reference Master DB"]
        AuditLedger["SHA-256 Hash Chain: H_i = SHA256(H_{i-1} || R_i)"]
        MLflowStore["mlruns/: MLflow Experiment Tracking"]
    end

    StreamlitUI & FastAPIEndpoint --> Orchestrator
    Orchestrator --> FinOpsAgent & MLEngine & SecOpsAgent
    FinOpsAgent --> VendorMaster
    MLEngine --> MLflowStore
    Orchestrator --> AuditLedger
```

---

## 🧩 Component Architecture & Design

### 1. NeurIPS 2022 ML & PCA Engine (`src/ml_engine.py`)
- **Covariance Eigen Decomposition**: $\Sigma = \frac{1}{n-1} X^T X$
- **Explained Variance Ratio**: $EVR_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$
- **95% Cumulative Variance Thresholding**: Dynamically compresses 24 numeric applicant features down to 19 principal components (retaining 96.21% cumulative variance).
- **MLflow Tracking**: Evaluates LightGBM, XGBoost, Random Forest, and Logistic Regression under experiment `"FinGuard_Fraud_ML_Benchmark"`.

### 2. Financial Operations Agent (`src/finops_agent.py`)
- **SequenceMatcher Ratio**: Calculates string similarity between applicant name and email username prefix dynamically.
- **Coercion Engine**: Sanitizes string inputs (`"$12,500.00"`) into floats (`12500.0`).
- **Policy Evaluator**: Evaluates $10k spending caps, vendor status, PO matching, and duplicate payment ledgers.

### 3. Security & Compliance Agent (`src/security_agent.py`)
- **UEBA Behavioral Anomaly Calculator**: Computes anomaly scores using velocity spikes and off-hours access patterns (1–4 AM).
- **Adversarial Prompt Injection Scanner**: Scans freeform note strings against regex and semantic prompt override signatures (`SAFE`/`UNSAFE`).

### 4. Decision Orchestrator & Audit Ledger (`src/orchestrator.py`)
Synthesizes multi-agent signals into a 3-way final verdict:
- **`AUTO_APPROVE`**: Low ML risk (<0.50), approved vendor, PO matched, safe security scan.
- **`ROUTE_TO_HUMAN_REVIEW`**: High dollar impact ($10k+ cap) or unapproved vendor requiring onboarding sign-off.
- **`AUTO_BLOCK`**: Prompt injection (`UNSAFE`), hard denylist rule, or critical ML fraud probability (>0.85).

---

## 🔄 Sequence Diagrams

### Transaction Event Decisioning Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / API Client
    participant API as FastAPI / Streamlit
    participant Orch as DecisionOrchestrator
    participant FinOps as FinOpsAgent
    participant ML as MLEngine
    participant Sec as SecurityAgent
    participant Ledger as SHA256 Audit Chain

    Client->>API: POST /decide (Transaction Event)
    API->>Orch: process_event(input_dict)
    Orch->>FinOps: evaluate_finops_policies(input_dict)
    FinOps-->>Orch: finops_evidence (PO match, vendor status, similarity)
    Orch->>ML: predict_fraud_risk(input_dict)
    ML-->>Orch: ml_evidence (fraud_probability, risk_tier)
    Orch->>Sec: evaluate_security_policies(input_dict)
    Sec-->>Orch: security_evidence (ueba_score, prompt_scan)
    Orch->>Orch: Synthesize 3-Way Verdict (AUTO_APPROVE / ROUTE_HUMAN / AUTO_BLOCK)
    Orch->>Ledger: tool_write_cryptographic_audit(event_id, verdict, evidence)
    Ledger-->>Orch: record + SHA-256 current_hash
    Orch-->>API: JSON Response (verdict, evidence, audit_hash)
    API-->>Client: 200 OK Response
```

---

## 💡 Design Decisions & Trade-off Rationales

### 1. Deterministic Code Rules vs. LLM Probabilistic Rule Execution
- **Decision**: Execute financial policy caps ($10k limit), PO matching, and duplicate payment detection in **pure deterministic Python code** rather than prompting an LLM.
- **Rationale**: Financial compliance requires **100% deterministic mathematical precision**. LLMs can hallucinate or fluctuate rule thresholds. Reserving ML for fraud risk scoring while using code for policy rules yields zero false policy rejections.

### 2. SHA-256 Hash Chaining vs. Traditional Database Storage
- **Decision**: Link each record with $H_i = \text{SHA256}(H_{i-1} \parallel R_i)$.
- **Rationale**: Traditional SQL databases allow retroactively editing past rows (`UPDATE transaction_log SET amount=...`). SHA-256 hash chaining ensures **tamper-evident auditability**—altering any past record invalidates every downstream hash, proving data tampering.

---

## ⚙️ Non-Functional Requirements (NFRs)

1. **Inference Latency SLA**: Complete multi-agent decisioning pipeline must execute in **< 50ms** end-to-end (sub-10ms for ML scoring).
2. **Audit Integrity**: 100% of decision records must be cryptographically signed with SHA-256 hash pointers.
3. **Portability**: Must run seamlessly on Linux, macOS, and Windows without external cloud service dependencies.
4. **Test Coverage**: Maintain 100% pass rate across automated PyTest unit test suite.
