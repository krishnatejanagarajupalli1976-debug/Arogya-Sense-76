import pytest

from healthcare_validation import evaluate_case


@pytest.mark.parametrize(
    ("case_id", "input_text", "expected"),
    [
        (1, "Age:45 Fever:1 Cough:1 Breathlessness:1 SpO2:93 BP:128/82",
         {"diagnosis": "pneumonia_risk_high", "action": "refer_specialist"}),
        (2, "BMI:31.4, Glucose:148, HbA1c:7.2, Fatigue:1, Polyuria:1",
         {"diagnosis": "type2_diabetes_likely", "confidence": 0.87}),
        (3, "BP_systolic:158, BP_diastolic:98, Age:55, Headache:1",
         {"classification": "stage2_hypertension", "referral": "immediate"}),
        (4, "Haemoglobin:9.2 g/dL, Fatigue:1, Pallor:1, Age:28 Gender:F",
         {"diagnosis": "iron_deficiency_anaemia", "recommendation": "iron_supplementation"}),
        (5, "SpO2:97, No fever, No cough, Glucose:85, BMI:22.4, Age:30",
         {"classification": "healthy", "action": "no_immediate_action_required"}),
        (6, "Glucose:78, HbA1c:5.1", {"verdict": "false_positive", "reason": "threshold_too_low"}),
        (7, "Correct:68 Total:100", {"accuracy": 0.68, "note": "below_acceptable_medical_threshold_0.85"}),
        (8, "Glucose:0.42 BMI:0.31 Age:0.18 Gender:-0.04",
         {"top_feature": "Glucose", "explanation": "high_glucose_primary_driver"}),
        (9, "Records:800", {"train": 560, "val": 120, "test": 120}),
        (10, "AUC_ROC:0.91", {"verdict": "excellent_discrimination", "readiness": "ready_for_pilot"}),
    ],
)
def test_supplied_healthcare_cases(case_id, input_text, expected):
    assert evaluate_case(case_id, input_text) == expected
