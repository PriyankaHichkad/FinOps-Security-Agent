#!/usr/bin/env python3
"""
FinOps-Security-Agent — LangGraph Runnable Multi-Agent StateGraph Orchestrator
Consolidates Multi-Agent State Transitions into a clean LangGraph Workflow.
"""

import os
import json
import hashlib
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from src.logger import logger
from src.ml_engine import MLEngine
from src.finops_agent import FinOpsAgent
from src.security_agent import SecurityAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_LEDGER_PATH = os.path.join(BASE_DIR, "data", "audit_ledger.json")

class AgentState(TypedDict):
    event: Dict[str, Any]
    ml_evidence: Dict[str, Any]
    finops_evidence: Dict[str, Any]
    security_evidence: Dict[str, Any]
    final_verdict: str
    risk_level: str
    audit_hash: str

class LangGraphOrchestrator:
    """
    LangGraph StateGraph Multi-Agent Orchestrator.
    Executes sequential node transitions across ML, FinOps, and SecOps agents.
    """
    def __init__(self):
        self.ml_engine = MLEngine()
        self.finops_agent = FinOpsAgent()
        self.security_agent = SecurityAgent()
        self.audit_chain: List[Dict[str, Any]] = []
        self.load_audit_ledger()
        self.app = self._build_graph()

    def load_audit_ledger(self):
        """Loads SHA-256 audit ledger from disk."""
        if os.path.exists(AUDIT_LEDGER_PATH):
            try:
                with open(AUDIT_LEDGER_PATH, "r") as f:
                    self.audit_chain = json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading audit ledger: {e}")
                self.audit_chain = []
        else:
            genesis_entry = {
                "record_id": 0,
                "event_id": "GENESIS",
                "verdict": "GENESIS",
                "previous_hash": "0" * 64,
                "current_hash": hashlib.sha256(b"GENESIS_BLOCK_FINOPS_SECURITY_AGENT").hexdigest()
            }
            self.audit_chain = [genesis_entry]

    def save_audit_ledger(self):
        """Saves SHA-256 audit chain to disk."""
        os.makedirs(os.path.dirname(AUDIT_LEDGER_PATH), exist_ok=True)
        with open(AUDIT_LEDGER_PATH, "w") as f:
            json.dump(self.audit_chain, f, indent=2)

    def _build_graph(self):
        """Constructs LangGraph StateGraph Workflow."""
        builder = StateGraph(AgentState)

        # Define LangGraph Nodes
        builder.add_node("ml_scoring", self._ml_node)
        builder.add_node("finops_evaluation", self._finops_node)
        builder.add_node("secops_evaluation", self._secops_node)
        builder.add_node("verdict_synthesis", self._orchestrator_node)

        # Define Edges
        builder.add_edge(START, "ml_scoring")
        builder.add_edge("ml_scoring", "finops_evaluation")
        builder.add_edge("finops_evaluation", "secops_evaluation")
        builder.add_edge("secops_evaluation", "verdict_synthesis")
        builder.add_edge("verdict_synthesis", END)

        return builder.compile()

    def _ml_node(self, state: AgentState) -> Dict[str, Any]:
        ml_res = self.ml_engine.predict_fraud_risk(state["event"])
        return {"ml_evidence": ml_res}

    def _finops_node(self, state: AgentState) -> Dict[str, Any]:
        fin_res = self.finops_agent.evaluate_finops_policies(state["event"])
        return {"finops_evidence": fin_res}

    def _secops_node(self, state: AgentState) -> Dict[str, Any]:
        sec_res = self.security_agent.evaluate_security_policies(state["event"])
        return {"security_evidence": sec_res}

    def _orchestrator_node(self, state: AgentState) -> Dict[str, Any]:
        event = state["event"]
        ml_ev = state["ml_evidence"]
        fin_ev = state["finops_evidence"]
        sec_ev = state["security_evidence"]

        # Rule Synthesis Logic
        verdict = "AUTO_APPROVE"
        risk_level = "LOW_RISK"

        # 1. Hard Block Trigger
        if sec_ev.get("injection_status") == "UNSAFE" or ml_ev.get("fraud_probability", 0.0) >= 0.85:
            verdict = "AUTO_BLOCK"
            risk_level = "CRITICAL_RISK"
        # 2. Human Review Trigger
        elif fin_ev.get("requires_human", False) or ml_ev.get("fraud_probability", 0.0) >= 0.70 or sec_ev.get("ueba_score", 0.0) > 0.60:
            verdict = "ROUTE_TO_HUMAN_REVIEW"
            risk_level = "HIGH_IMPACT_REVIEW"

        # SHA-256 Hash Chain Record Creation
        prev_hash = self.audit_chain[-1]["current_hash"] if self.audit_chain else "0" * 64
        record_id = len(self.audit_chain)
        payload = f"{record_id}|{event.get('event_id', 'EVT-000')}|{verdict}|{prev_hash}"
        curr_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        audit_entry = {
            "record_id": record_id,
            "event_id": event.get("event_id", "EVT-UNKNOWN"),
            "verdict": verdict,
            "risk_level": risk_level,
            "prev_hash": prev_hash,
            "previous_hash": prev_hash,
            "current_hash": curr_hash
        }

        self.audit_chain.append(audit_entry)
        self.save_audit_ledger()

        return {
            "final_verdict": verdict,
            "risk_level": risk_level,
            "audit_hash": curr_hash
        }

    def process_event(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executes LangGraph workflow end-to-end for an incoming event."""
        initial_state: AgentState = {
            "event": input_dict,
            "ml_evidence": {},
            "finops_evidence": {},
            "security_evidence": {},
            "final_verdict": "",
            "risk_level": "",
            "audit_hash": ""
        }

        final_state = self.app.invoke(initial_state)

        return {
            "event_id": input_dict.get("event_id", "EVT-UNKNOWN"),
            "final_verdict": final_state["final_verdict"],
            "risk_level": final_state["risk_level"],
            "audit_hash": final_state["audit_hash"],
            "layer_breakdown": {
                "ml_engine": final_state["ml_evidence"],
                "finops_agent": final_state["finops_evidence"],
                "security_agent": final_state["security_evidence"]
            }
        }

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Validates SHA-256 cryptographic audit chain integrity."""
        if not self.audit_chain:
            return {"is_valid": True, "total_records": 0, "status": "EMPTY"}

        for i in range(1, len(self.audit_chain)):
            prev = self.audit_chain[i - 1]
            curr = self.audit_chain[i]
            prev_h = curr.get("prev_hash", curr.get("previous_hash", ""))
            if prev_h != prev.get("current_hash"):
                return {
                    "is_valid": False,
                    "tamper_detected_at_record": curr.get("record_id", i),
                    "status": "TAMPER_DETECTED"
                }

        return {
            "is_valid": True,
            "total_records": len(self.audit_chain),
            "status": "TAMPER_EVIDENT_VALIDATED"
        }
