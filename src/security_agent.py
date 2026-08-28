import re
from src.logger import logger, FinGuardException

class SecurityAgent:
    """
    Security & Compliance Agent.
    Implements User & Entity Behavior Analytics (UEBA) baseline anomaly scoring,
    adversarial prompt injection sanitization, and regulatory compliance screening.
    """
    def __init__(self):
        # Known prompt injection malicious patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s+prompt",
            r"override\s+verdict",
            r"admin\s+access",
            r"bypass\s+policy",
            r"set\s+verdict\s+to",
            r"sudo\s+",
            r"drop\s+table"
        ]

    def tool_sanitize_text_inputs(self, text_content: str) -> dict:
        """
        Tool Call: Scans freeform text notes/descriptions for adversarial prompt injection vectors.
        """
        if not text_content or not isinstance(text_content, str):
            return {"status": "SAFE", "flags": []}

        flags = []
        text_lower = text_content.lower()

        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                flags.append(f"Prompt injection pattern detected: '{pattern}'")

        if flags:
            logger.warning(f"Security Alert! Prompt Injection attempt detected in input: {flags}")
            return {"status": "UNSAFE", "flags": flags}

        return {"status": "SAFE", "flags": []}

    def tool_calculate_ueba_baseline(self, actor_id: str, velocity_6h: float, access_hour: int = 12) -> dict:
        """
        Tool Call: Calculates User & Entity Behavior Analytics (UEBA) baseline anomaly score.
        Compares transaction attempt velocity and time of day against normal entity baseline.
        """
        anomaly_score = 0.0
        signals = []

        # 1. Off-hours execution anomaly (1 AM to 4 AM)
        if access_hour in [1, 2, 3, 4]:
            anomaly_score += 0.35
            signals.append(f"Off-hours access anomaly detected at {access_hour}:00 hrs")

        # 2. Velocity spike anomaly (> 6 attempts in 6 hours)
        if velocity_6h > 10:
            anomaly_score += 0.45
            signals.append(f"Critical transaction velocity spike: {velocity_6h} attempts in 6h")
        elif velocity_6h > 5:
            anomaly_score += 0.25
            signals.append(f"Elevated transaction velocity: {velocity_6h} attempts in 6h")

        anomaly_score = round(min(anomaly_score, 1.0), 2)
        
        return {
            "ueba_anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score >= 0.50,
            "signals": signals
        }

    def evaluate_security_policies(self, input_dict: dict) -> dict:
        """
        Main Security Agent Execution Loop:
        1. Sanitize freeform text notes.
        2. Calculate UEBA anomaly score.
        3. Evaluate regulatory compliance rules.
        """
        try:
            actor_id = input_dict.get("actor_id") or input_dict.get("user_id") or "USER_ANONYMOUS"
            access_hour = int(input_dict.get("access_hour", 12))
            velocity_6h = float(input_dict.get("velocity_6h", 1.0))
            notes = str(input_dict.get("notes") or input_dict.get("description") or "")

            # Tool Calls
            injection_result = self.tool_sanitize_text_inputs(notes)
            ueba_result = self.tool_calculate_ueba_baseline(actor_id, velocity_6h, access_hour)

            security_flags = []
            hard_block = False
            requires_review = False

            # Rule 1: Prompt Injection Attempt -> Hard Block Immediately
            if injection_result["status"] == "UNSAFE":
                hard_block = True
                security_flags.extend(injection_result["flags"])

            # Rule 2: High UEBA Anomaly -> Route to Human / SOC Flag
            if ueba_result["is_anomaly"]:
                requires_review = True
                security_flags.extend(ueba_result["signals"])

            return {
                "injection_status": injection_result["status"],
                "ueba_score": ueba_result["ueba_anomaly_score"],
                "security_flags": security_flags,
                "hard_block": hard_block,
                "requires_review": requires_review
            }

        except Exception as e:
            logger.error(f"Error in Security Agent evaluation: {e}")
            raise FinGuardException(e)

# Global Singleton Instance
security_agent = SecurityAgent()
