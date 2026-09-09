import os
import pytest
from src.llm_explainer import LLMFraudExplainer, llm_explainer
from src.orchestrator import orchestrator

def test_llm_explainer_fallback_mode():
    """Verifies deterministic XAI explanation generation in zero-cost offline mode."""
    explainer = LLMFraudExplainer()
    explainer.api_key = ""  # Force offline fallback mode

    mock_context = {
        "event_id": "EVT-TEST-001",
        "final_verdict": "ROUTE_TO_HUMAN_REVIEW",
        "risk_level": "HIGH_IMPACT_REVIEW",
        "layer_breakdown": {
            "ml_engine": {"fraud_probability": 0.72, "risk_tier": "HIGH"},
            "finops_agent": {
                "sanitized_amount": 12500.0,
                "name_email_similarity": 0.35,
                "policy_findings": ["Invoice amount $12,500.00 exceeds $10,000 auto-approval cap"]
            },
            "security_agent": {"injection_status": "SAFE", "ueba_score": 0.0, "security_flags": []}
        }
    }

    res = explainer.generate_explanation(mock_context)
    assert res is not None
    assert "llm_explanation" in res
    assert res["explainability_mode"] == "Grounded XAI (Zero-Cost Rule Engine)"
    assert "EVT-TEST-001" in res["llm_explanation"]
    assert "72.0%" in res["llm_explanation"] or "exceeds" in res["llm_explanation"]

def test_orchestrator_integration_with_xai():
    """Verifies orchestrator process_event includes llm_explanation in output payload."""
    sample_event = {
        "event_id": "EVT-2026-XAI-01",
        "applicant_name": "John Doe",
        "email": "johndoe123@gmail.com",
        "vendor_name": "Acme Corp",
        "invoice_amount": "$4,500.00",
        "po_number": "PO-1001",
        "income": 0.6,
        "credit_risk_score": 740,
        "velocity_6h": 1,
        "access_hour": 14
    }

    result = orchestrator.process_event(sample_event)
    assert "llm_explanation" in result
    assert "explainability_mode" in result
    assert len(result["llm_explanation"]) > 10
