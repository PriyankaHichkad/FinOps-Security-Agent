#!/usr/bin/env python3
"""
FinOps-Security-Agent — Predictive-Generative LLM Fraud Explainer Module (XAI)
Bridges Predictive ML risk scores (XGBoost + Focal Loss) and FinOps/SecOps policy evidence with Generative AI
to produce grounded, non-hallucinated natural language fraud decision explanations.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger import logger

class LLMFraudExplainer:
    """
    Predictive-Generative Hybrid XAI Engine.
    Translates mathematical XGBoost + Focal Loss fraud scores, SHAP risk drivers, and policy evidence
    into grounded natural language decision rationales.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def generate_explanation(self, decision_context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates a grounded natural language explanation for a decision event.
        Returns a dict containing 'llm_explanation' and 'explainability_mode'.
        """
        event_id = decision_context.get("event_id", "EVT-UNKNOWN")
        verdict = decision_context.get("final_verdict", "AUTO_APPROVE")
        risk_level = decision_context.get("risk_level", "LOW_RISK")
        
        ml_breakdown = decision_context.get("layer_breakdown", {}).get("ml_engine", {})
        finops_breakdown = decision_context.get("layer_breakdown", {}).get("finops_agent", {})
        sec_breakdown = decision_context.get("layer_breakdown", {}).get("security_agent", {})

        fraud_prob = ml_breakdown.get("fraud_probability", 0.0)
        risk_tier = ml_breakdown.get("risk_tier", "LOW")
        policy_findings = finops_breakdown.get("policy_findings", [])
        similarity = finops_breakdown.get("name_email_similarity", 1.0)
        sanitized_amount = finops_breakdown.get("sanitized_amount", 0.0)
        
        injection_status = sec_breakdown.get("injection_status", "SAFE")
        ueba_score = sec_breakdown.get("ueba_score", 0.0)
        security_flags = sec_breakdown.get("security_flags", [])

        # 1. Attempt Live Gemini API Call if GEMINI_API_KEY is configured
        if self.api_key:
            try:
                gemini_text = self._call_gemini_api(
                    event_id, verdict, fraud_prob, risk_level, policy_findings, 
                    similarity, sanitized_amount, injection_status, ueba_score, security_flags
                )
                if gemini_text:
                    return {
                        "llm_explanation": gemini_text,
                        "explainability_mode": "Grounded LLM (Google Gemini Free Tier)"
                    }
            except Exception as e:
                logger.warning(f"Gemini API call failed, falling back to deterministic XAI: {e}")

        # 2. Zero-Cost Deterministic Rule-Based XAI Template Fallback
        fallback_text = self._generate_deterministic_explanation(
            event_id, verdict, fraud_prob, risk_level, policy_findings, 
            similarity, sanitized_amount, injection_status, ueba_score, security_flags
        )
        return {
            "llm_explanation": fallback_text,
            "explainability_mode": "Grounded XAI (Zero-Cost Rule Engine)"
        }

    def _call_gemini_api(
        self, event_id: str, verdict: str, fraud_prob: float, risk_level: str,
        policy_findings: list, similarity: float, amount: float,
        injection_status: str, ueba_score: float, security_flags: list
    ) -> Optional[str]:
        """Calls Google Gemini REST API using urllib (zero external HTTP dependencies)."""
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        prompt = (
            f"You are an Enterprise Fraud Compliance Audit AI. Provide a concise, professional 2-sentence audit rationale "
            f"explaining the verdict for decision event {event_id}.\n\n"
            f"EVIDENCE (DO NOT INVENT EXTRA FACTS):\n"
            f"- Final Verdict: {verdict}\n"
            f"- Risk Tier: {risk_level} (Fraud Probability: {fraud_prob:.4f})\n"
            f"- Invoice Amount: ${amount:,.2f}\n"
            f"- Name-Email Match Similarity: {similarity*100:.1f}%\n"
            f"- Policy Findings: {policy_findings if policy_findings else 'None'}\n"
            f"- Security Injection Status: {injection_status}\n"
            f"- UEBA Anomaly Score: {ueba_score:.2f} (Flags: {security_flags if security_flags else 'None'})\n\n"
            f"Draft a direct, factual 2-sentence summary suitable for a SOX compliance audit."
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 150
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        return None

    def _generate_deterministic_explanation(
        self, event_id: str, verdict: str, fraud_prob: float, risk_level: str,
        policy_findings: list, similarity: float, amount: float,
        injection_status: str, ueba_score: float, security_flags: list
    ) -> str:
        """Generates a zero-cost, deterministic grounded natural language explanation."""
        drivers = []
        if fraud_prob > 0.50:
            drivers.append(f"high ML fraud probability of {fraud_prob*100:.1f}% (XGBoost + Focal Loss score)")
        elif fraud_prob > 0.20:
            drivers.append(f"moderate ML fraud score of {fraud_prob*100:.1f}%")
        
        if policy_findings:
            drivers.append(f"policy violations ({'; '.join(policy_findings)})")
        
        if similarity < 0.50:
            drivers.append(f"low applicant name-email similarity ({similarity*100:.1f}% match)")
            
        if injection_status == "UNSAFE":
            drivers.append("prompt injection security threat detected")
            
        if ueba_score > 0.50:
            drivers.append(f"elevated behavioral anomaly score ({ueba_score:.2f})")

        if verdict == "AUTO_APPROVE":
            return (
                f"Event {event_id} was AUTO_APPROVED under LOW_RISK classification. "
                f"ML model estimated a low fraud probability of {fraud_prob*100:.1f}%, "
                f"with valid vendor credentials, PO match, and clean security scan."
            )
        elif verdict == "AUTO_BLOCK":
            driver_str = ", ".join(drivers) if drivers else "critical security or policy violation"
            return (
                f"Event {event_id} was AUTO_BLOCKED due to {driver_str}. "
                f"The transaction presents critical financial risk requiring immediate rejection."
            )
        else:  # ROUTE_TO_HUMAN_REVIEW
            driver_str = ", ".join(drivers) if drivers else f"invoice amount of ${amount:,.2f} exceeding auto-approval limits"
            return (
                f"Event {event_id} was ROUTED TO HUMAN REVIEW due to {driver_str}. "
                f"The event requires manual verification by a compliance manager prior to disbursement."
            )

# Global singleton
llm_explainer = LLMFraudExplainer()
