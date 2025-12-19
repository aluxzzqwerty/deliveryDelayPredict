import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model
