# ArogyaSense 🏥

Disease prediction and symptom screening for an 8-hospital healthcare network in Nagpur. The project includes a FastAPI backend, Streamlit UI, 30-disease model, and per-prediction feature explanations.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
uvicorn api:app --reload
```

Open the documented API at http://127.0.0.1:8000/docs. In another terminal, run the UI with `streamlit run streamlit_app.py`.

## API example

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"symptoms":["fever","chills","body_ache","fatigue","cough"]}'
```

The supplied download URLs currently respond with candidate portal HTML instead of the advertised CSV/TXT files. This repo uses a deterministic 30-disease fallback training pack until the real data is available. This is an educational screening tool, not a diagnosis.

## Healthcare validation cases

The ten supplied healthcare validation scenarios are available through `POST /healthcare-case`:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthcare-case -Method Post -ContentType 'application/json' -Body '{"case_id":1,"input_text":"Age:45 Fever:1 Cough:1 Breathlessness:1 SpO2:93 BP:128/82"}'
```

The endpoint returns deterministic validation results for triage, screening, classification, explainability, data splitting, and model evaluation scenarios. These rules are educational checks and do not replace clinical assessment.

Run tests with `pytest -q`.
