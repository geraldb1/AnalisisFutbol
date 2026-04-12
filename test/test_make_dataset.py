import pandas as pd
from src.config import ruta_processed

def test_data_modeling_existe():
    """Verifica que el archivo data_modeling.csv existe."""
    archivo = ruta_processed / "data_modeling.csv"
    assert archivo.exists(), "data_modeling.csv no existe"

def test_data_modeling_columnas():
    """Verifica que el dataset tiene 31 columnas."""
    df = pd.read_csv(ruta_processed / "data_modeling.csv")
    assert df.shape[1] == 31, f"Se esperaban 31 columnas, tiene {df.shape[1]}"

def test_data_modeling_sin_nulos():
    """Verifica que no hay valores nulos en las features."""
    df = pd.read_csv(ruta_processed / "data_modeling.csv")
    nulos = df.isnull().sum().sum()
    assert nulos == 0, f"Hay {nulos} valores nulos"

def test_data_modeling_target_valido():
    """Verifica que FTR solo contiene H, D, A."""
    df = pd.read_csv(ruta_processed / "data_modeling.csv")
    valores = set(df["FTR"].unique())
    assert valores == {"H", "D", "A"}, f"FTR tiene valores inesperados: {valores}"