from fastapi import FastAPI
import joblib
from pathlib import Path
from pydantic import BaseModel
from src.config import MAP_INVERSO

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "lgbm_balanced.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Football Prediction API")

@app.get("/")
def home():
    return {"mensaje": "API de predicción de fútbol funcionando"}

class MatchFeatures(BaseModel):
    Goles_avg5_local: float
    goles_rival_avg5_local: float
    tiros_avg5_local: float
    tiros_al_arco_avg5_local: float
    corners_avg5_local: float
    faltas_avg5_local: float
    amarillas_avg5_local: float
    rojas_avg5_local: float
    goles_ht_avg5_local: float
    goles_ht_rival_avg5_local: float
    resultado_avg5_local: float
    Goles_avg5_visitante: float
    goles_rival_avg5_visitante: float
    tiros_avg5_visitante: float
    tiros_al_arco_avg5_visitante: float
    corners_avg5_visitante: float
    faltas_avg5_visitante: float
    amarillas_avg5_visitante: float
    rojas_avg5_visitante: float
    goles_ht_avg5_visitante: float
    goles_ht_rival_avg5_visitante: float
    resultado_avg5_visitante: float
    B365H: float
    B365D: float
    B365A: float


@app.post("/predict")
def predict(match: MatchFeatures):
    features = [[
        match.Goles_avg5_local,
        match.goles_rival_avg5_local,
        match.tiros_avg5_local,
        match.tiros_al_arco_avg5_local,
        match.corners_avg5_local,
        match.faltas_avg5_local,
        match.amarillas_avg5_local,
        match.rojas_avg5_local,
        match.goles_ht_avg5_local,
        match.goles_ht_rival_avg5_local,
        match.resultado_avg5_local,
        match.Goles_avg5_visitante,
        match.goles_rival_avg5_visitante,
        match.tiros_avg5_visitante,
        match.tiros_al_arco_avg5_visitante,
        match.corners_avg5_visitante,
        match.faltas_avg5_visitante,
        match.amarillas_avg5_visitante,
        match.rojas_avg5_visitante,
        match.goles_ht_avg5_visitante,
        match.goles_ht_rival_avg5_visitante,
        match.resultado_avg5_visitante,
        match.B365H,
        match.B365D,
        match.B365A
    ]]
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    return {
        "prediccion": MAP_INVERSO[pred],
        "probabilidades": {
            "Home": round(float(proba[0]), 4),
            "Draw": round(float(proba[1]), 4),
            "Away": round(float(proba[2]), 4),
        }
    }
