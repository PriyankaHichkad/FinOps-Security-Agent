from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.orchestrator import orchestrator
from src.ml_engine import ml_engine
from src.logger import logger

app = FastAPI(
    title="FinOps & Security Compliance Agent API",
    description="REST Server for Autonomous Financial Decisioning, NeurIPS Fraud ML Scoring, and Cryptographic SHA-256 Audit Logging.",
    version="1.0.0"
)

class EventRequest(BaseModel):
    event_id: Optional[str] = Field(None, example="EVT-9042")
    applicant_name: Optional[str] = Field(None, example="John Smith")
    email: Optional[str] = Field(None, example="johnsmith@gmail.com")
    vendor_name: Optional[str] = Field(None, example="Acme Corp")
    invoice_amount: Optional[Any] = Field(12500.00, example=12500.00)
    po_number: Optional[str] = Field(None, example="PO-1001")
    income: Optional[float] = Field(0.5, example=0.5)
    name_email_similarity: Optional[float] = Field(None, example=0.82)
    velocity_6h: Optional[float] = Field(1.0, example=1.0)
    velocity_12h: Optional[float] = Field(None, example=2.0)
    credit_risk_score: Optional[float] = Field(720.0, example=720.0)
    access_hour: Optional[int] = Field(14, example=14)
    notes: Optional[str] = Field("Routine monthly invoice", example="Routine monthly invoice")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FinOps-Security-Agent", "version": "1.0.0"}

@app.post("/decide")
def decide_event(payload: EventRequest):
    """
    Submits a transaction/application event for autonomous decisioning.
    Synthesizes ML fraud risk, FinOps policy caps, and SecOps UEBA anomaly scores in <50ms.
    """
    try:
        input_dict = payload.model_dump()
        result = orchestrator.process_event(input_dict)
        return result
    except Exception as e:
        logger.error(f"Error processing decision in REST server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/verify")
def verify_audit_ledger():
    """
    Validates the integrity of the SHA-256 cryptographic audit chain.
    """
    return orchestrator.verify_audit_chain()

@app.get("/audit/ledger")
def get_full_audit_ledger():
    """
    Returns the complete cryptographic SHA-256 audit ledger history.
    """
    return orchestrator.get_audit_ledger()

@app.get("/metrics")
def get_system_metrics():
    """
    Returns ML model performance metrics (PR-AUC, ROC-AUC) and PCA Scree plot stats.
    """
    pca_metrics = ml_engine.get_pca_metrics()
    audit_status = orchestrator.verify_audit_chain()
    return {
        "model_champion": "LightGBM Classifier",
        "pca_variance": pca_metrics,
        "audit_ledger_status": audit_status
    }

@app.post("/decide/batch")
def decide_batch_events():
    """
    Executes PySpark Big Data Batch Processing on stored batch CSV datasets.
    """
    from src.pyspark_batch import PySparkBatchEngine
    engine = PySparkBatchEngine()
    summary = engine.run_batch_pipeline()
    return summary
