from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from healthcare_model import DISCLAIMER, DISEASE_SYMPTOMS, get_model, model_metadata
from healthcare_validation import evaluate_case

app = FastAPI(title="ArogyaSense Disease Prediction API", version="1.0.0", description="Symptom screening API for the Nagpur hospital chain.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1, max_length=20, description="Symptom identifiers from GET /symptoms")


class HealthcareCaseRequest(BaseModel):
    case_id: int = Field(ge=1, le=10)
    input_text: str = Field(min_length=1)


@app.get("/")
def root() -> dict:
    return {
        "service": "arogyasense",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "arogyasense", **model_metadata()}


@app.get("/symptoms")
def symptoms() -> dict:
    values = get_model().features
    return {"symptoms": values, "count": len(values)}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return get_model().predict(request.symptoms)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/healthcare-case")
def healthcare_case(request: HealthcareCaseRequest) -> dict:
    try:
        return {"case_id": request.case_id, "result": evaluate_case(request.case_id, request.input_text),
                "disclaimer": DISCLAIMER}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/diseases")
def diseases() -> dict:
    return {"diseases": sorted(DISEASE_SYMPTOMS), "count": len(DISEASE_SYMPTOMS), "disclaimer": DISCLAIMER}
