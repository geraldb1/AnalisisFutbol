import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import (
    ruta_processed,ruta_model,
    MAP_RESULTADO,TEST_SEASON,VAL_SEASON,MODEL_FEATURES
)

LABEL_NAMES = ["H (Home)", "D (Draw)", "A (Away)"]


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

#-------dividir los datos para evaluación-------
def split_data(df):
    """
    Divide los datos respetando el orden temporal
    """
    test_x = df[df["temporada"] == TEST_SEASON][MODEL_FEATURES]
    test_y = df[df["temporada"] == TEST_SEASON]['FTR'].map(MAP_RESULTADO)
    
    val_x = df[df["temporada"] == VAL_SEASON][MODEL_FEATURES]
    val_y = df[df["temporada"] == VAL_SEASON]['FTR'].map(MAP_RESULTADO)
    
    print(f"Test: {test_x.shape}, Validation: {val_x.shape}")
    return test_x, test_y, val_x, val_y

#-------evaluar el modelo-------

def evaluate_baseline(df, season,y_true):
    """
    Evalua el baseline de Bet365 (menor numero = mayor probabilidad)
    """
    baseline_preds = df[df["temporada"] == season][['B365H', 'B365D', 'B365A']].idxmin(axis=1,skipna=True).str[-1]
    baseline_map = baseline_preds.map(MAP_RESULTADO)
    acc = accuracy_score(y_true, baseline_map)
    print(f"Baseline Accuracy for {season}: {acc:.4f}")
    print("Baseline Classification Report:")
    print(classification_report(y_true, baseline_map, target_names=LABEL_NAMES))
    print("Baseline Confusion Matrix:")
    print(confusion_matrix(y_true, baseline_map))
    return acc



def evaluate_model(model, test_x, test_y, val_x, val_y):
    """
    Evalúa el modelo en los conjuntos de prueba y validación, mostrando métricas clave.
    """
    for name, x, y in [("Test", test_x, test_y), ("Validation", val_x, val_y)]:
        print(f"\nEvaluando en {name} set:")
        preds = model.predict(x)
        acc = accuracy_score(y, preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y, preds, target_names=LABEL_NAMES))
        print("Confusion Matrix:")
        print(confusion_matrix(y, preds))



#----main----
def main():
    print("=" * 50)
    print("PASO 1: Carga de datos y modelo")
    df = load_data("data_modeling.csv")
    model = load_model("lgbm_balanced.pkl")

    print("\nPASO 2: Dividir los datos para evaluación")
    test_x, test_y, val_x, val_y = split_data(df)

    print("\nPASO 3: Evaluar el baseline de Bet365")
    evaluate_baseline(df, TEST_SEASON, test_y)
    evaluate_baseline(df, VAL_SEASON, val_y)

    print("\nPASO 4: Evaluar el modelo entrenado")
    evaluate_model(model, test_x, test_y, val_x, val_y)

if __name__ == "__main__":
    main()
        
