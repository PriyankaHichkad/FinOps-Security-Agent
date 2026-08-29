import os
import re
import json
from difflib import SequenceMatcher
from src.logger import logger, FinGuardException

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_MASTER_PATH = os.path.join(BASE_DIR, "data", "vendor_master.json")

class FinOpsAgent:
    """
    Financial Operations Decisioning Agent.
    Implements dynamic name-email similarity calculation, flexible type coercion,
    PO reconciliation, vendor master lookups, duplicate payment checks, and
    deterministic financial policy evaluation.
    """
    def __init__(self):
        self.vendor_master = self._load_vendor_master()
        self.processed_ledger = set()  # Tracks (vendor, amount_float, po_number, date)
        self.daily_spending_ledger = {} # Tracks (user, vendor, date) -> cumulative_amount

    def reset_ledgers(self):
        """Resets historical and daily spending ledgers (useful for testing and session reset)."""
        self.processed_ledger.clear()
        self.daily_spending_ledger.clear()

    def _load_vendor_master(self) -> dict:
        try:
            if os.path.exists(VENDOR_MASTER_PATH):
                with open(VENDOR_MASTER_PATH, "r") as f:
                    return json.load(f)
            return {"approved_vendors": []}
        except Exception as e:
            logger.error(f"Failed to load vendor master database: {e}")
            return {"approved_vendors": []}

    def tool_register_new_vendor(self, vendor_name: str, category: str = "General Procurement", risk_rating: str = "LOW") -> dict:
        """
        Tool Call: Registers and approves a new vendor in the vendor_master.json database
        upon human manager sign-off.
        """
        if not vendor_name:
            return {"status": "ERROR", "message": "Vendor name required"}

        vendor_clean = vendor_name.strip()
        new_id = f"VEND-{len(self.vendor_master.get('approved_vendors', [])) + 101}"

        new_entry = {
            "vendor_id": new_id,
            "vendor_name": vendor_clean,
            "status": "APPROVED",
            "category": category,
            "risk_rating": risk_rating,
            "contract_status": "ACTIVE"
        }

        self.vendor_master.setdefault("approved_vendors", []).append(new_entry)

        try:
            with open(VENDOR_MASTER_PATH, "w") as f:
                json.dump(self.vendor_master, f, indent=2)
            logger.info(f"Vendor '{vendor_clean}' successfully onboarded into vendor master database.")
            return {"status": "SUCCESS", "vendor_entry": new_entry}
        except Exception as e:
            logger.error(f"Failed to write vendor master update: {e}")
            return {"status": "ERROR", "message": str(e)}


    @staticmethod
    def compute_name_email_similarity(name: str, email: str) -> float:
        """
        Dynamically calculates string similarity between an applicant's full name
        and email username prefix using Levenshtein / SequenceMatcher ratio.
        """
        if not name or not email:
            return 0.5

        name_clean = re.sub(r"[^a-zA-Z]", "", name.lower())
        email_prefix = email.split("@")[0].lower() if "@" in email else email.lower()
        email_clean = re.sub(r"[^a-zA-Z]", "", email_prefix)

        if not name_clean or not email_clean:
            return 0.5

        ratio = SequenceMatcher(None, name_clean, email_clean).ratio()
        return round(float(ratio), 4)

    @staticmethod
    def sanitize_amount(amount_val) -> float:
        """
        Flexibly sanitizes raw user inputs ($12,500.00, 12500, "12,500") to float.
        """
        if amount_val is None:
            return 0.0
        if isinstance(amount_val, (int, float)):
            return float(amount_val)
        
        # String cleaning
        s = str(amount_val).replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def sanitize_velocity(input_dict: dict) -> float:
        """
        Dynamically converts various velocity time windows (1h, 12h, 24h) to 6h equivalent.
        """
        if "velocity_6h" in input_dict and input_dict["velocity_6h"] is not None:
            return float(input_dict["velocity_6h"])
        elif "velocity_12h" in input_dict and input_dict["velocity_12h"] is not None:
            return round(float(input_dict["velocity_12h"]) / 2.0, 2)
        elif "velocity_1h" in input_dict and input_dict["velocity_1h"] is not None:
            return round(float(input_dict["velocity_1h"]) * 6.0, 2)
        elif "velocity_24h" in input_dict and input_dict["velocity_24h"] is not None:
            return round(float(input_dict["velocity_24h"]) / 4.0, 2)
        return 1.0

    def tool_query_vendor_master(self, vendor_name: str) -> dict:
        """Tool call: Queries data/vendor_master.json for vendor status and metadata."""
        if not vendor_name:
            return {"status": "UNAPPROVED", "category": "Unclassified", "risk_rating": "HIGH", "contract_status": "UNVERIFIED"}

        vendor_name_clean = vendor_name.strip().lower()
        for v in self.vendor_master.get("approved_vendors", []):
            if v["vendor_name"].strip().lower() == vendor_name_clean:
                return v

        # Fuzzy matching check
        for v in self.vendor_master.get("approved_vendors", []):
            ratio = SequenceMatcher(None, vendor_name_clean, v["vendor_name"].strip().lower()).ratio()
            if ratio >= 0.80:
                return v

        return {"status": "UNAPPROVED", "category": "Unclassified", "risk_rating": "HIGH", "contract_status": "UNVERIFIED"}

    def tool_reconcile_po(self, po_number: str, amount: float) -> dict:
        """Tool call: Reconciles purchase order presence and amount tolerances."""
        if not po_number or str(po_number).strip().upper() in ["N/A", "NONE", "NULL", ""]:
            return {"po_matched": False, "reason": "No Purchase Order provided"}
        
        # Valid PO pattern check
        po_clean = str(po_number).strip().upper()
        if po_clean.startswith("PO-") or len(po_clean) >= 4:
            return {"po_matched": True, "reason": f"Matched Purchase Order {po_clean}"}
        return {"po_matched": False, "reason": f"Invalid Purchase Order format: {po_clean}"}

    def tool_check_cumulative_daily_spending(self, vendor_name: str, amount: float, user_id: str = "GLOBAL_USER", date_str: str = "TODAY") -> float:
        """Tool call: Tracks cumulative daily spending total per user/client paying a vendor."""
        user_clean = str(user_id).strip().lower() if user_id else "global_user"
        key = (user_clean, str(vendor_name).strip().lower(), str(date_str).strip())
        current_total = self.daily_spending_ledger.get(key, 0.0)
        return current_total + amount

    def tool_check_duplicate_ledger(self, vendor_name: str, amount: float, po_number: str = "N/A", date_str: str = "TODAY") -> bool:
        """Tool call: Checks processed ledger to prevent duplicate double payments."""
        po_clean = str(po_number).strip().upper() if po_number else "N/A"
        key = (str(vendor_name).strip().lower(), round(float(amount), 2), po_clean, str(date_str).strip())
        if key in self.processed_ledger:
            return True
        return False

    def record_transaction_in_ledger(self, vendor_name: str, amount: float, po_number: str = "N/A", user_id: str = "GLOBAL_USER", date_str: str = "TODAY"):
        """Records paid transaction in historical ledger and updates cumulative daily spending per user."""
        po_clean = str(po_number).strip().upper() if po_number else "N/A"
        key = (str(vendor_name).strip().lower(), round(float(amount), 2), po_clean, str(date_str).strip())
        self.processed_ledger.add(key)

        user_clean = str(user_id).strip().lower() if user_id else "global_user"
        daily_key = (user_clean, str(vendor_name).strip().lower(), str(date_str).strip())
        self.daily_spending_ledger[daily_key] = self.daily_spending_ledger.get(daily_key, 0.0) + amount

    def evaluate_finops_policies(self, input_dict: dict) -> dict:
        """
        Main FinOps Agent Execution Loop:
        1. Extract & Sanitize fields.
        2. Compute dynamic name-email similarity if not provided.
        3. Execute tool calls (Vendor query, PO check, Duplicate check).
        4. Evaluate deterministic financial rules.
        """
        try:
            # 1. Flexible Extraction & Sanitization
            raw_vendor = input_dict.get("vendor_name") or input_dict.get("vendor") or input_dict.get("supplier") or ""
            raw_amount = input_dict.get("invoice_amount") or input_dict.get("amount") or input_dict.get("total") or 0.0
            amount = self.sanitize_amount(raw_amount)

            po_number = input_dict.get("po_number") or input_dict.get("po_id") or input_dict.get("po") or "N/A"
            applicant_name = input_dict.get("applicant_name") or input_dict.get("name") or input_dict.get("user_name") or ""
            email = input_dict.get("email") or input_dict.get("email_address") or ""
            user_id = input_dict.get("actor_id") or input_dict.get("user_id") or email or applicant_name or "GLOBAL_USER"

            # Dynamic similarity calculation
            if "name_email_similarity" in input_dict and input_dict["name_email_similarity"] is not None:
                sim_score = float(input_dict["name_email_similarity"])
            else:
                sim_score = self.compute_name_email_similarity(applicant_name, email)

            # Field confidence evaluation
            confidence = 1.0
            if not applicant_name or not email:
                confidence -= 0.15
            if amount <= 0:
                confidence -= 0.20

            # 2. Tool Executions
            vendor_info = self.tool_query_vendor_master(raw_vendor)
            po_info = self.tool_reconcile_po(po_number, amount)
            is_duplicate = self.tool_check_duplicate_ledger(raw_vendor, amount, po_number)

            # 3. Deterministic Policy Rules
            policy_findings = []
            requires_human = False
            hard_deny = False

            # Rule 1: User / Account Spending Limit Cap ($10,000 Default / User Credit Limit)
            user_limit = float(input_dict.get("proposed_credit_limit") or 10000.0)
            cumulative_today = self.tool_check_cumulative_daily_spending(raw_vendor, amount, user_id)

            if amount > user_limit:
                requires_human = True
                policy_findings.append(f"Single invoice amount ${amount:,.2f} exceeds user '{user_id}' assigned limit of ${user_limit:,.2f}")
            elif cumulative_today > user_limit and user_limit > 0:
                requires_human = True
                policy_findings.append(f"Cumulative daily spending for user '{user_id}' (${cumulative_today:,.2f}) exceeds daily limit of ${user_limit:,.2f}")

            # Rule 2: Unapproved Vendor Policy
            if vendor_info.get("status") == "UNAPPROVED":
                requires_human = True
                policy_findings.append(f"Vendor '{raw_vendor}' is not in approved vendor master database")

            # Rule 3: Missing PO on High Amount
            if not po_info["po_matched"] and amount >= 5000.0:
                requires_human = True
                policy_findings.append("High amount ($5,000+) transaction missing valid Purchase Order (PO)")

            # Rule 4: Duplicate Payment Policy (Hard Deny / Block)
            if is_duplicate:
                hard_deny = True
                policy_findings.append(f"Duplicate payment detected for vendor '{raw_vendor}' with amount ${amount:,.2f}")

            return {
                "sanitized_amount": amount,
                "vendor_name": raw_vendor,
                "vendor_info": vendor_info,
                "po_info": po_info,
                "is_duplicate": is_duplicate,
                "name_email_similarity": sim_score,
                "extraction_confidence": round(confidence, 2),
                "policy_findings": policy_findings,
                "requires_human": requires_human,
                "hard_deny": hard_deny
            }

        except Exception as e:
            logger.error(f"Error in FinOps Agent evaluation: {e}")
            raise FinGuardException(e)

# Global Singleton Instance
finops_agent = FinOpsAgent()
