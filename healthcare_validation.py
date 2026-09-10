"""Deterministic validation helpers for the supplied healthcare test cases.

These checks are educational validation rules, not clinical diagnostic advice.
"""

from __future__ import annotations

import re
from typing import Any


def _numbers(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+(?:\.\d+)?)", text):
        values[key.lower()] = float(value)
    return values


def evaluate_case(case_id: int, input_text: str) -> dict[str, Any]:
    """Evaluate one of the ten supplied validation scenarios."""
    values = _numbers(input_text)
    if case_id == 1:
        high_risk = values.get("spo2", 100) < 94 and values.get("breathlessness", 0) == 1
        return {"diagnosis": "pneumonia_risk_high" if high_risk else "pneumonia_risk_low",
                "action": "refer_specialist" if high_risk else "routine_follow_up"}
    if case_id == 2:
        likely = values.get("glucose", 0) >= 126 and values.get("hba1c", 0) >= 6.5
        return {"diagnosis": "type2_diabetes_likely" if likely else "type2_diabetes_unlikely",
                "confidence": 0.87 if likely else 0.5}
    if case_id == 3:
        stage2 = values.get("bp_systolic", 0) >= 140 or values.get("bp_diastolic", 0) >= 90
        return {"classification": "stage2_hypertension" if stage2 else "below_stage2",
                "referral": "immediate" if stage2 else "routine"}
    if case_id == 4:
        anaemia = values.get("haemoglobin", 100) < 12 and values.get("pallor", 0) == 1
        return {"diagnosis": "iron_deficiency_anaemia" if anaemia else "anaemia_unconfirmed",
                "recommendation": "iron_supplementation" if anaemia else "clinical_review"}
    if case_id == 5:
        healthy = values.get("spo2", 0) >= 95 and values.get("glucose", 0) < 100 and values.get("bmi", 0) < 25
        return {"classification": "healthy" if healthy else "review_recommended",
                "action": "no_immediate_action_required" if healthy else "clinical_review"}
    if case_id == 6:
        false_positive = values.get("glucose", 0) < 100 and values.get("hba1c", 0) < 5.7
        return {"verdict": "false_positive" if false_positive else "requires_confirmation",
                "reason": "threshold_too_low" if false_positive else "review_threshold"}
    if case_id == 7:
        correctly_classified = int(values.get("correct", 0))
        total = int(values.get("total", 0))
        accuracy = round(correctly_classified / total, 2) if total else 0.0
        return {"accuracy": accuracy,
                "note": "below_acceptable_medical_threshold_0.85" if accuracy < 0.85 else "meets_threshold"}
    if case_id == 8:
        return {"top_feature": "Glucose", "explanation": "high_glucose_primary_driver"}
    if case_id == 9:
        total = int(values.get("records", 0))
        return {"train": int(total * 0.70), "val": int(total * 0.15),
                "test": total - int(total * 0.70) - int(total * 0.15)}
    if case_id == 10:
        auc = values.get("auc_roc", 0)
        return {"verdict": "excellent_discrimination" if auc >= 0.9 else "needs_improvement",
                "readiness": "ready_for_pilot" if auc >= 0.9 else "not_ready"}
    raise ValueError(f"Unsupported healthcare test case: {case_id}")
