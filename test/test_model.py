import pandas as pd
from src.config import ruta_model,VAL_SEASON,ruta_processed,MODEL_FEATURES,TEST_SEASON
import joblib


def test_model_existe():
    """Verifica que el archivo lgbm_balanced.pkl existe."""
    archivo = ruta_model / "lgbm_balanced.pkl"
    assert archivo.exists(), "lgbm_balanced.pkl no existe"
    

def test_model_carga():
    """Verifica que el modelo se puede cargar correctamente."""
    model = joblib.load(ruta_model / "lgbm_balanced.pkl")
    assert model is not None, "El modelo cargado es None"

def test_model_predict():
    """Verifica que el modelo predice 3 clases."""
    model = joblib.load(ruta_model / "lgbm_balanced.pkl")
    df = pd.read_csv(ruta_processed / "data_modeling.csv")
    X = df[df["temporada"] == TEST_SEASON][MODEL_FEATURES]
    proba = model.predict_proba(X)
    assert proba.shape[1] == 3, f"Se esperaban 3 clases, tiene {proba.shape[1]}"

def test_pred_100():
    """Verifica que las probabilidades suman ~1."""
    model = joblib.load(ruta_model / "lgbm_balanced.pkl")
    df = pd.read_csv(ruta_processed / "data_modeling.csv")
    X = df[df["temporada"] == VAL_SEASON][MODEL_FEATURES]
    proba = model.predict_proba(X)
    sumas = proba.sum(axis=1)
    assert all(s > 0.99 and s < 1.01 for s in sumas), "Las probabilidades no suman ~1"