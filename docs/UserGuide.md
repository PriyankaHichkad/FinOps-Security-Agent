# FinOps-Security-Agent — REST API & Operations Guide

> **Enterprise User & API Operational Manual for Autonomous Financial Decisioning and Security Compliance.**

Welcome to the **FinOps-Security-Agent** Operational Guide. This guide provides comprehensive instructions on interacting with the FastAPI REST API microservice, sending decision requests, verifying the cryptographic audit ledger, and managing vendor master rules.

---

## 📋 Table of Contents
1. [Quick Start Guide](#-quick-start-guide)
2. [REST API Endpoint Reference](#-rest-api-endpoint-reference)
   - [`POST /decide`](#post-decide)
   - [`GET /audit/verify`](#get-auditverify)
   - [`GET /metrics`](#get-metrics)
   - [`GET /health`](#get-health)
3. [Interactive OpenAPI / Swagger Documentation](#-interactive-openapi--swagger-documentation)
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

# 3. Launch FastAPI REST Server
uvicorn main:app --reload --port 8000
```
* Access Swagger API Documentation live at `http://localhost:8000/docs`.

---

## 🔌 REST API Endpoint Reference

### `POST /decide`
Processes an incoming transaction, application, or invoice event through the multi-agent decision core and appends the decision to the SHA-256 audit chain.

**Request Headers**: `Content-Type: application/json`

**Sample Request Body**:
```json
{
  "event_id": "EVT-2026-001",
  "applicant_name": "Alice Johnson",
  "email": "alicejohnson@gmail.com",
  "vendor_name": "Acme Corp",
  "invoice_amount": "$4,500.00",
  "po_number": "PO-1001",
  "income": 0.6,
  "credit_risk_score": 740,
  "velocity_6h": 1,
  "access_hour": 14,
  "notes": "Monthly software subscription fee",
  "actor_id": "USR-9042"
}
```

**Sample Response Body (`200 OK`)**:
```json
{
  "event_id": "EVT-2026-001",
  "final_verdict": "AUTO_APPROVE",
  "risk_level": "LOW_RISK",
  "audit_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "layer_breakdown": {
    "ml_engine": {
      "fraud_probability": 0.0421,
      "risk_tier": "LOW",
      "pca_components_used": 19
    },
    "finops_agent": {
      "sanitized_amount": 4500.0,
      "name_email_similarity": 0.9412,
      "extraction_confidence": 1.0,
      "policy_findings": [],
      "requires_human": false
    },
    "security_agent": {
      "injection_status": "SAFE",
      "ueba_score": 0.0,
      "security_flags": []
    }
  }
}
```

---

### `GET /audit/verify`
Validates the complete SHA-256 cryptographic hash chain on disk to prove zero retroactive tampering.

**Sample Response Body (`200 OK`)**:
```json
{
  "is_valid": true,
  "total_records": 12,
  "tip_hash": "a4f8e912bc34567890def1234567890abcde1234567890abcde1234567890abc",
  "status": "TAMPER_EVIDENT_VALIDATED"
}
```

---

### `GET /metrics`
Returns system performance SLAs, PCA variance statistics, and active model metadata.

---

### `GET /health`
Liveness check endpoint returning `{"status": "healthy", "service": "FinOps-Security-Agent"}`.

---

## 📚 Vendor Master Management

The vendor reference database is stored in `data/vendor_master.json`. 

- **Approved Vendors**: Pre-approved companies with custom dollar auto-approval caps (e.g. Acme Corp, TechData Inc).
- **Unapproved Vendors**: Any unknown vendor not in `vendor_master.json` automatically triggers `ROUTE_TO_HUMAN_REVIEW`.

---

## ❓ FAQ & Troubleshooting

**Q: How does the system handle string amounts like `"$12,500.00"`?**  
A: The FinOps agent automatically strips currency symbols, commas, and whitespace, coercing inputs into clean numerical floats (`12500.0`).

**Q: Where are audit logs persisted?**  
A: Every decision block is cryptographically linked and saved permanently to `data/audit_ledger.json`.
