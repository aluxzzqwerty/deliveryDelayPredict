from fastapi import FastAPI, HTTPException
import pandas as pd

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import get_model

app = FastAPI(
    title="Late Delivery Risk Predictor",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = get_model()

        # Convert dict → DataFrame
        X = pd.DataFrame([request.features])

        # Predict
        proba = model.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)

        return PredictionResponse(
            prediction=pred,
            probability=float(proba)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
