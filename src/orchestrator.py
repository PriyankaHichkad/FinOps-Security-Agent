import hashlib
import json
from datetime import datetime
from src.logger import logger, FinGuardException
from src.ml_engine import ml_engine
from src.finops_agent import finops_agent
from src.security_agent import security_agent

import os

LEDGER_FILE_PATH = os.path.join("data", "audit_ledger.json")

class DecisionOrchestrator:
    """
    Multi-Agent Decision Orchestrator & Cryptographic Ledger Engine.
    Synthesizes signals from ML Engine, FinOps Agent, and Security Agent into a 3-way verdict
    and appends every decision to an immutable SHA-256 Cryptographic Hash Chain.
    """
    def __init__(self):
        self.audit_chain = []
        self._load_or_initialize_ledger()

    def _load_or_initialize_ledger(self):
        """Loads existing audit ledger from disk or initializes genesis block."""
        if os.path.exists(LEDGER_FILE_PATH):
            try:
                with open(LEDGER_FILE_PATH, "r") as f:
                    self.audit_chain = json.load(f)
                if self.audit_chain:
                    return
            except Exception as e:
                logger.warning(f"Could not load audit ledger from disk: {e}")
        
        self.audit_chain = []
        self._initialize_genesis_block()

    def _save_ledger_to_disk(self):
        """Saves current audit chain to disk."""
        try:
            os.makedirs(os.path.dirname(LEDGER_FILE_PATH), exist_ok=True)
            with open(LEDGER_FILE_PATH, "w") as f:
                json.dump(self.audit_chain, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist audit ledger to disk: {e}")

    def _initialize_genesis_block(self):
        """Initializes the tamper-evident cryptographic genesis block."""
        genesis_record = {
            "record_index": 0,
            "timestamp": "2026-08-28T00:00:00Z",
            "event_id": "GENESIS_BLOCK",
            "verdict": "INITIALIZED",
            "evidence": "FinOps-Security-Agent Cryptographic Audit Ledger Initialized",
            "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000"
        }
        genesis_record["current_hash"] = self._calculate_hash(genesis_record)
        self.audit_chain.append(genesis_record)

    @staticmethod
    def _calculate_hash(record_dict: dict) -> str:
        """Calculates SHA-256 cryptographic signature of a decision record."""
        # Make a shallow copy without current_hash to ensure canonical hashing
        data = {k: v for k, v in record_dict.items() if k != "current_hash"}
        canonical_json = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def tool_write_cryptographic_audit(self, event_id: str, verdict: str, inputs: dict, evidence: dict) -> dict:
        """
        Tool Call: Appends a decision record to the tamper-evident SHA-256 hash chain.
        """
        prev_hash = self.audit_chain[-1]["current_hash"] if self.audit_chain else "0000"
        record_index = len(self.audit_chain)

        record = {
            "record_index": record_index,
            "timestamp": datetime.now().isoformat(),
            "event_id": event_id,
            "verdict": verdict,
            "inputs": {
                "vendor_name": inputs.get("vendor_name", ""),
                "amount": inputs.get("sanitized_amount", 0.0),
                "applicant_name": inputs.get("applicant_name", "")
            },
            "evidence": evidence,
            "prev_hash": prev_hash
        }
        record["current_hash"] = self._calculate_hash(record)
        self.audit_chain.append(record)
        self._save_ledger_to_disk()
        return record

    def process_event(self, input_dict: dict) -> dict:
        """
        Main Decisioning Pipeline:
        1. Sanitize velocity & amounts.
        2. Evaluate ML Fraud Risk (NeurIPS LightGBM).
        3. Evaluate FinOps Policies & PO Reconciliation.
        4. Evaluate SecOps UEBA Anomaly & Injection Defense.
        5. Synthesize 3-Way Final Verdict.
        6. Append to SHA-256 Cryptographic Audit Ledger.
        """
        try:
            event_id = input_dict.get("event_id") or f"EVT-{len(self.audit_chain):04d}"

            # 1. FinOps Agent Evaluation
            finops_res = finops_agent.evaluate_finops_policies(input_dict)

            # Inject similarity & velocity into input dict for ML Engine
            input_dict["name_email_similarity"] = finops_res["name_email_similarity"]
            input_dict["velocity_6h"] = finops_agent.sanitize_velocity(input_dict)

            # 2. ML Engine Fraud Risk Evaluation (NeurIPS 2022)
            ml_res = ml_engine.predict_fraud_risk(input_dict)

            # 3. Security Agent Evaluation
            security_res = security_agent.evaluate_security_policies(input_dict)

            # 4. Multi-Agent Verdict Synthesis
            fraud_proba = ml_res["fraud_probability"]
            finops_hard_deny = finops_res["hard_deny"]
            sec_hard_block = security_res["hard_block"]

            finops_human = finops_res["requires_human"]
            sec_human = security_res["requires_review"]
            confidence = finops_res["extraction_confidence"]

            if fraud_proba >= 0.99 or finops_hard_deny or sec_hard_block:
                final_verdict = "AUTO_BLOCK"
                risk_level = "CRITICAL_RISK"
            elif finops_human or sec_human or fraud_proba >= 0.65 or confidence < 0.85:
                final_verdict = "ROUTE_TO_HUMAN_REVIEW"
                risk_level = "HIGH_IMPACT_REVIEW"
            else:
                final_verdict = "AUTO_APPROVE"
                risk_level = "LOW_RISK"
                # Record successful payment in FinOps ledger
                user_identifier = input_dict.get("actor_id") or input_dict.get("user_id") or input_dict.get("email") or input_dict.get("applicant_name") or "GLOBAL_USER"
                finops_agent.record_transaction_in_ledger(
                    finops_res["vendor_name"],
                    finops_res["sanitized_amount"],
                    finops_res.get("po_number", "N/A"),
                    user_identifier
                )

            # Compile evidence summary
            all_findings = []
            all_findings.extend(finops_res["policy_findings"])
            all_findings.extend(security_res["security_flags"])
            if fraud_proba >= 0.65:
                all_findings.append(f"NeurIPS LightGBM Fraud Probability elevated ({fraud_proba*100:.1f}%)")

            evidence = {
                "ml_fraud_score": fraud_proba,
                "ml_risk_tier": ml_res["risk_tier"],
                "extraction_confidence": confidence,
                "vendor_status": finops_res["vendor_info"].get("status"),
                "po_matched": finops_res["po_info"].get("po_matched"),
                "ueba_score": security_res["ueba_score"],
                "injection_status": security_res["injection_status"],
                "rationales": all_findings
            }

            # 5. Append to Cryptographic Ledger
            audit_record = self.tool_write_cryptographic_audit(event_id, final_verdict, finops_res, evidence)

            return {
                "event_id": event_id,
                "final_verdict": final_verdict,
                "risk_level": risk_level,
                "audit_hash": audit_record["current_hash"],
                "layer_breakdown": {
                    "ml_engine": ml_res,
                    "finops_agent": finops_res,
                    "security_agent": security_res
                },
                "evidence_summary": all_findings
            }

        except Exception as e:
            logger.error(f"Error in Decision Orchestrator: {e}")
            raise FinGuardException(e)

    def verify_audit_chain(self) -> dict:
        """
        Validates the integrity of the entire cryptographic hash chain.
        Returns is_valid (bool), total_records (int), and tamper_details (if any).
        """
        if not self.audit_chain:
            return {"is_valid": True, "total_records": 0, "status": "EMPTY"}

        for i in range(1, len(self.audit_chain)):
            curr = self.audit_chain[i]
            prev = self.audit_chain[i - 1]

            # 1. Verify link hash
            if curr["prev_hash"] != prev["current_hash"]:
                return {
                    "is_valid": False,
                    "total_records": len(self.audit_chain),
                    "tampered_index": i,
                    "reason": f"PrevHash mismatch at index {i}. Expected {prev['current_hash']}, got {curr['prev_hash']}"
                }

            # 2. Verify current hash calculation
            recalculated = self._calculate_hash(curr)
            if curr["current_hash"] != recalculated:
                return {
                    "is_valid": False,
                    "total_records": len(self.audit_chain),
                    "tampered_index": i,
                    "reason": f"Hash recalculation mismatch at index {i}. Record content was modified after creation!"
                }

        return {
            "is_valid": True,
            "total_records": len(self.audit_chain),
            "tip_hash": self.audit_chain[-1]["current_hash"],
            "status": "TAMPER_EVIDENT_VALIDATED"
        }

    @property
    def audit_ledger(self) -> list:
        """Property alias for self.audit_chain."""
        return self.audit_chain

    def get_audit_ledger(self) -> list:
        """Returns the complete cryptographic audit ledger."""
        return self.audit_chain

# Global Singleton Instance
orchestrator = DecisionOrchestrator()
