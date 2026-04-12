import pandas as pd
import os
import joblib
from src.config import (
    ruta_processed,ruta_model,
    MAP_INVERSO,ruta_scores,VAL_SEASON,MODEL_FEATURES
)

SCORE_SEASON = VAL_SEASON


# -----------Cargar modelo y datos-----------

def load_data(filename):
    """
    Carga el dataset procesado desde la carpeta processed.
    """
    df = pd.read_csv(ruta_processed / filename)
    print(f"Datos cargados: {df.shape}")
    return df


def load_model(filename):
    """
    Carga el modelo entrenado desde la carpeta model.
    """
    model = joblib.load(ruta_model / filename)
    print(f"Modelo cargado: {filename}")
    return model


#----- Prediccion -------
def get_pred(model, df, season):
    df_score = df[df['temporada'] == season].copy()
    X = df_score[MODEL_FEATURES]
    print(f"Generando predicciones para la temporada {season} con {X.shape[0]} muestras.")
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    df_score['Prediccion'] = pd.Series(y_pred, index=X.index).map(MAP_INVERSO)
    df_score['Prob_H'] = y_prob[:, 0]
    df_score['Prob_D'] = y_prob[:, 1]
    df_score['Prob_A'] = y_prob[:, 2]

    cols_output = [
        "id_partido", "Date", "Equipo_local", "Equipo_visitante",
        "FTR", "Prediccion", "Prob_H", "Prob_D", "Prob_A",
    ]
    cols_disponibles = [c for c in cols_output if c in df_score.columns]
    df_result = df_score[cols_disponibles]

    print(f"Predicciones generadas: {len(df_result)}")
    return df_result


#--------- Exportar resultados -----------
def export_results(df, filename):
    """
    Exporta el dataframe con predicciones a un archivo CSV en la carpeta processed.
    """
    os.makedirs(ruta_scores, exist_ok=True)
    filepath = ruta_scores / filename
    
    #verificar si archivo existe
    archivo_existe = os.path.isfile(filepath)
    df.to_csv(
        filepath,
        mode = 'a' if archivo_existe else 'w',
        index=False,
        header=not archivo_existe)
    print(f"Resultados exportados a: {filepath}")
   
# --------- Main -----------
def main():
    print("="*50)
    print("Cargando los datos y modelo...")
    print("="*50)
    
    df = load_data("data_modeling.csv")
    model = load_model("lgbm_balanced.pkl")
    print("="*50)
    print("Realizando predicciones...",end="\n")
    print("Predicciones para la temporada:", SCORE_SEASON)
    print("="*50)
    pred = get_pred(model, df, SCORE_SEASON)
    
    print("="*50)
    print("Exportando resultados...")
    print("="*50)
    export_results(pred, f"predicciones_{SCORE_SEASON}.csv")
    
    print("\n── Primeras predicciones ──")
    print(pred.head(10).to_string(index=False))
    
if __name__ == "__main__":
    main()