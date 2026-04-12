import pandas as pd
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier
from src.config import (
    ruta_processed,ruta_model,
    MAP_RESULTADO,TEST_SEASON,VAL_SEASON,MODEL_FEATURES
)

#------ Parámetros para GridSearch (pueden ajustarse según recursos y tiempo) ------
PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 5, 7],
    "num_leaves": [15, 31, 50],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [5, 10, 20],
}


# ---cargar los datos-----
def read_data(filename):
    """
    Lee el dataset procesado desde la carpeta processed.
    """
    df = pd.read_csv(ruta_processed / filename)
    print(f"Datos cargados: {df.shape}")
    return df



# ---- Dividir los datos ----
def split_data(df):
    """
    Divide los datos respetando el orden temporal
    """
    exclude_seasons = [TEST_SEASON, VAL_SEASON]
    train_x = df[~df["temporada"].isin(exclude_seasons)][MODEL_FEATURES]
    train_y =df[~df["temporada"].isin(exclude_seasons)]['FTR'].map(MAP_RESULTADO)

    
    print(f"Train: {train_x.shape}")
    return train_x, train_y

#--- Entrenar el modelo ----
def train_model(train_x, train_y):
    """
    Entrenar el modelo LightGBM balanceado con GridSearCV y TimeSeriesSplit
    """
    ts = TimeSeriesSplit(n_splits=5)
    
    lgbm = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced")
    
    gsearch = GridSearchCV(
        estimator=lgbm,
        param_grid=PARAM_GRID,
        cv=ts,
        n_jobs=-1
    )
    
    print("Iniciando entrenamiento LightGBM con GridSearchCV...")
    gsearch.fit(train_x, train_y)
    
    print(f"Mejores parámetros: {gsearch.best_params_}")
    print(f"Mejor score CV: {gsearch.best_score_:.4f}")
    return gsearch.best_estimator_

#--- Guardando el modelo ----
def save_model(model, filename):
    """
    Guarda el modelo entrenado en la carpeta model.
    """
    os.makedirs(ruta_model, exist_ok=True)
    filepath = ruta_model / filename
    joblib.dump(model, filepath)
    print(f"Modelo guardado: {filepath}")


#---generar main----

def main():
    print("=" * 50)
    print("PASO 1: Carga de datos")
    print("=" * 50)
    df = read_data("data_modeling.csv")    
    
    
    print("=" * 50)
    print("PASO 2: División de datos")
    print("=" * 50)
    train_x,train_y = split_data(df)
    
    print("=" * 50)
    print("PASO 3: Entrenamiento del modelo")
    print("=" * 50)
    model = train_model(train_x, train_y)
    
    print("=" * 50)
    print("PASO 4: Guardar el modelo")
    print("=" * 50)
    save_model(model, "lgbm_balanced.pkl")

if __name__ == "__main__":
    main()