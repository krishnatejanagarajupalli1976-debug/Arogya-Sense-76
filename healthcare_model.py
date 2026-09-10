"""Shared disease prediction model and explainability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:
    shap = None

DISCLAIMER = "This is an educational screening tool, not a diagnosis. Seek qualified medical care for symptoms or emergencies."
DISEASE_SYMPTOMS: dict[str, list[str]] = {
    "Common Cold": ["sneezing", "runny_nose", "sore_throat", "cough"], "Influenza": ["fever", "chills", "body_ache", "fatigue", "cough"], "COVID-19": ["fever", "dry_cough", "loss_of_taste", "loss_of_smell", "fatigue"],
    "Pneumonia": ["high_fever", "productive_cough", "chest_pain", "shortness_of_breath"], "Bronchitis": ["cough", "mucus", "chest_discomfort", "fatigue"], "Asthma": ["wheezing", "shortness_of_breath", "chest_tightness", "cough"],
    "Allergic Rhinitis": ["sneezing", "itchy_eyes", "runny_nose", "nasal_congestion"], "Sinusitis": ["facial_pain", "nasal_congestion", "headache", "thick_nasal_discharge"], "Migraine": ["severe_headache", "nausea", "light_sensitivity", "visual_aura"],
    "Tension Headache": ["headache", "neck_pain", "scalp_tenderness", "stress"], "Gastroenteritis": ["vomiting", "diarrhea", "stomach_cramps", "fever"], "Food Poisoning": ["vomiting", "diarrhea", "stomach_cramps", "nausea"],
    "GERD": ["heartburn", "acid_regurgitation", "chest_discomfort", "difficulty_swallowing"], "Irritable Bowel Syndrome": ["stomach_cramps", "bloating", "diarrhea", "constipation"], "Urinary Tract Infection": ["painful_urination", "frequent_urination", "pelvic_pain", "cloudy_urine"],
    "Kidney Stones": ["flank_pain", "blood_in_urine", "painful_urination", "nausea"], "Hypertension": ["headache", "dizziness", "blurred_vision", "chest_pain"], "Type 2 Diabetes": ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_sensitivity", "dry_skin"], "Hyperthyroidism": ["weight_loss", "rapid_heartbeat", "sweating", "anxiety"], "Anemia": ["fatigue", "pale_skin", "dizziness", "shortness_of_breath"],
    "Dengue Fever": ["high_fever", "severe_headache", "joint_pain", "skin_rash"], "Malaria": ["fever", "chills", "sweating", "headache"], "Tuberculosis": ["persistent_cough", "night_sweats", "weight_loss", "blood_in_sputum"],
    "Chickenpox": ["skin_rash", "itching", "fever", "fatigue"], "Measles": ["skin_rash", "fever", "cough", "red_eyes"], "Hepatitis A": ["jaundice", "nausea", "abdominal_pain", "dark_urine"],
    "Dermatitis": ["skin_rash", "itching", "dry_skin", "redness"], "Conjunctivitis": ["red_eyes", "itchy_eyes", "eye_discharge", "eye_pain"], "Depression": ["low_mood", "sleep_changes", "fatigue", "loss_of_interest"],
}


def _all_symptoms() -> list[str]:
    return sorted({symptom for symptoms in DISEASE_SYMPTOMS.values() for symptom in symptoms})


def _training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    symptoms = _all_symptoms()
    rows, labels = [], []
    rng = np.random.default_rng(42)
    for disease, signature in DISEASE_SYMPTOMS.items():
        for _ in range(18):
            row = [int(symptom in signature) for symptom in symptoms]
            for index in rng.choice(len(symptoms), size=2, replace=False):
                row[index] = 1 - row[index]
            rows.append(row)
            labels.append(disease)
    return np.asarray(rows), np.asarray(labels), symptoms


class DiseaseModel:
    def __init__(self) -> None:
        x, y, self.features = _training_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        self.model = RandomForestClassifier(n_estimators=240, random_state=42, class_weight="balanced")
        self.model.fit(x_train, y_train)
        self.accuracy = float(accuracy_score(y_test, self.model.predict(x_test)))
        self.explainer = shap.TreeExplainer(self.model) if shap is not None else None

    def predict(self, selected_symptoms: list[str]) -> dict[str, Any]:
        unknown = sorted(set(selected_symptoms) - set(self.features))
        if unknown:
            raise ValueError(f"Unknown symptoms: {', '.join(unknown)}")
        vector = np.asarray([[int(symptom in selected_symptoms) for symptom in self.features]])
        probabilities = self.model.predict_proba(vector)[0]
        order = np.argsort(probabilities)[::-1][:3]
        contribution = vector[0] * self.model.feature_importances_
        method = "Tree feature contribution for selected symptoms"
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(vector)
            if isinstance(shap_values, list):
                contribution = np.max(np.abs(np.asarray(shap_values)[:, 0, :]), axis=0) * vector[0]
            else:
                raw_values = np.asarray(shap_values)
                if raw_values.ndim == 3:
                    raw_values = raw_values[0]
                contribution = (np.max(np.abs(raw_values), axis=-1) if raw_values.ndim == 2 else raw_values.reshape(-1)) * vector[0]
            method = "SHAP TreeExplainer contribution for selected symptoms"
        explanation = [{"symptom": self.features[index].replace("_", " ").title(), "impact": round(float(contribution[index]), 4)} for index in np.argsort(contribution)[::-1] if vector[0][index] and contribution[index] > 0][:5]
        return {"disease": str(self.model.classes_[order[0]]), "confidence": round(float(probabilities[order[0]]), 4), "top_predictions": [{"disease": str(self.model.classes_[index]), "confidence": round(float(probabilities[index]), 4)} for index in order], "shap_explanation": explanation, "explanation_method": method, "disclaimer": DISCLAIMER}


_model: DiseaseModel | None = None


def get_model() -> DiseaseModel:
    global _model
    if _model is None:
        _model = DiseaseModel()
    return _model


def model_metadata() -> dict[str, Any]:
    model = get_model()
    return {"disease_count": len(DISEASE_SYMPTOMS), "symptom_count": len(model.features), "validation_accuracy": round(model.accuracy, 4)}
