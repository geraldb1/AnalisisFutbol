import pandas as pd
import numpy as np
import requests
import os
from io import StringIO
from src.config import (
    ruta_raw, ruta_processed,
    TEMPORADAS, FILTRO_TEMPORADAS
)

# ─────────────────────── Descarga de las temporadas de la pagina football-data.co.uk ──────────────────────────
def download_data():
    """
    Descarga los datos de fútbol desde football-data.co.uk y los guarda en la carpeta raw.
    """
    os.makedirs(ruta_raw, exist_ok=True)
    for temporada in TEMPORADAS:
        archivo = ruta_raw / f"SP1_{temporada}.csv"
        csv = f"SP1_{temporada}.csv"
        if archivo.exists() and temporada != TEMPORADAS[-1]:
            print(f"Archivo {csv} ya existe, saltando descarga.")
            continue
        url = f"http://www.football-data.co.uk/mmz4281/{temporada}/SP1.csv"
        print(f"Descargando datos de la temporada {temporada}...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(archivo, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"Datos de la temporada {temporada} descargados y guardados en {archivo}.")

        except requests.RequestException as e:
            print(f"Error al descargar datos de la temporada {temporada}: {e}")

    print("Descarga de datos completa.")
  
# ─────────────────────── Consolidación de datos ──────────────────────────    
def cons_data():
    """
    Lee los archivos CSV descargados, los concatena en un solo DataFrame y lo guarda en la carpeta reprocess.
    """
    os.makedirs(ruta_processed, exist_ok=True)
    archivos = sorted(ruta_raw.glob("SP1_*.csv"))
    df_all = pd.DataFrame()
    print(f"Consolidando datos de {len(archivos)} temporadas...")
    
    for archivo in archivos:
        print(f"Procesando temporada: {archivo.name}")
        campaña = archivo.stem.split("_")[1]
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                contenido = f.read()
        except UnicodeDecodeError:
            with open(archivo, "r", encoding="latin-1") as f:
                contenido = f.read()
        lineas = contenido.splitlines()
        lineas_limpias = [linea.rstrip(',') for linea in lineas]
        texto_limpio = "\n".join(lineas_limpias)
        df_temp = pd.read_csv(StringIO(texto_limpio),sep = ",")
        df_temp["temporada"] = campaña
        df_temp.drop(columns=['Div'], inplace=True, errors='ignore')  # luego elimina Div
        df_all = pd.concat([df_all, df_temp], ignore_index=True)  # finalmente concatena
        print(f"Total de datos cargados {campaña}: {len(df_temp)}")
        
    df_all.to_csv(ruta_processed / "data_consolidated.csv", index=False)
    return df_all

# ─────────────────────── Limpieza de datos ──────────────────────────    
def clean_data(df):
    """
    Limpia el DataFrame eliminando filas con valores faltantes y filtrando por temporadas.
    """
    #Elimina columnas completamente vacías
    unnamed = [col for col in df.columns if 'Unnamed' in col]
    df.drop(columns=unnamed, inplace=True)
    
    #Cambiar el formato de la columna Date a datetime, con dayfirst=True para interpretar correctamente el formato día/mes/año
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.time
    
    #Generar el ID_PARTIDO
    df = df.reset_index(drop=True)  # Reiniciar el índice para asegurar que sea secuencial
    df['id_partido'] = df.index + 1  # Crear ID_PARTIDO comenzando desde 1
    
    #Seleccionar las columnas relevantes para el modelo
    columnas_claves = [
       "id_partido", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF",
        "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A", "temporada"]
    cols_disponibles = [c for c in columnas_claves if c in df.columns]
    df = df[cols_disponibles]
    print(f"✅ Limpieza completa - Columnas: {df.shape[1]}, Filas: {df.shape[0]}")
    df.to_csv(ruta_processed / "data_cleaned.csv", index=False)
    return df

def data_engineering(df):
    """
    Realiza ingeniería de características creando nuevas columnas basadas en las existentes.
    """
    #Se retira las temporadas que no tienen datos completos
    partidos = df[~df['temporada'].isin(FILTRO_TEMPORADAS)].copy()
    
    #Datos de los equipos Locales
    data_local = partidos[['temporada','Date','id_partido','HomeTeam',
                        'FTHG','FTAG','HS','HST','HC','HF','HY','HR','HTHG','HTAG','FTR']]\
                .rename(columns={
                    'HomeTeam':'Equipo','FTHG':'Goles','FTAG':'goles_rival','HS':'tiros',
                    'HST':'tiros_al_arco','HC':'corners','HF':'faltas','HY':'amarillas',
                    'HR':'rojas','HTHG':'goles_ht','HTAG':'goles_ht_rival','FTR':'resultado'})
    data_local['Condicion'] = 'Local'

    #Datos de los equipos Visitantes
    data_visitante = partidos[['temporada','Date','id_partido','AwayTeam',
                        'FTAG','FTHG','AS','AST','AC','AF','AY','AR','HTAG','HTHG','FTR']]\
                .rename(columns={
                    'AwayTeam':'Equipo','FTAG':'Goles','FTHG':'goles_rival','AS':'tiros',
                    'AST':'tiros_al_arco','AC':'corners','AF':'faltas','AY':'amarillas',
                    'AR':'rojas','HTAG':'goles_ht','HTHG':'goles_ht_rival','FTR':'resultado'})
    data_visitante['Condicion'] = 'Visitantes'

    #Mapeo del resultado a puntos para el equipo local y visitante
    res_local = {"H": 3, "D": 1, "A": 0}
    res_visitante = {"A": 3, "D": 1, "H": 0}        
    data_local['resultado'] = data_local['resultado'].map(res_local)
    data_visitante['resultado'] = data_visitante['resultado'].map(res_visitante)
    
    #Concatenar los datos de locales y visitantes para calcular las estadísticas por equipo
    df_partidos = pd.concat([data_local, data_visitante], ignore_index=True)
    
    #Ordenar por equipo y fecha para calcular correctamente las estadísticas móviles
    df_partidos = df_partidos.sort_values(['Equipo', 'Date'])
    
    #Se calcula el promedio móvil de las últimas 5 jornadas para cada equipo
    col = ['Goles', 'goles_rival',
       'tiros', 'tiros_al_arco', 'corners', 'faltas', 'amarillas', 'rojas',
       'goles_ht', 'goles_ht_rival', 'resultado']
    for c in col:
        df_partidos[f'{c}_avg5'] = (
        df_partidos.groupby('Equipo')[c]
        .transform(lambda x: x.rolling(5).mean().shift(1))
    )
    
    #Se Re-pivoteara para darle formato partido
    df_local = df_partidos[df_partidos['Condicion']== 'Local']
    df_visitante = df_partidos[df_partidos['Condicion']== 'Visitantes'] 
    
    col_avg = [
       'temporada', 'Date', 'id_partido', 'Equipo',
       'Goles_avg5', 'goles_rival_avg5',
       'tiros_avg5', 'tiros_al_arco_avg5',
       'corners_avg5', 'faltas_avg5',
       'amarillas_avg5', 'rojas_avg5',
       'goles_ht_avg5', 'goles_ht_rival_avg5',
       'resultado_avg5']
    df_local = df_local[col_avg]
    df_visitante = df_visitante[col_avg]
    
    df_avg = df_local.merge(df_visitante,on='id_partido',suffixes=('_local', '_visitante'))
    
    #Unir los datos de las apuestas B365 y el resultado final del partido
    df_odds = partidos[['id_partido','B365H','B365D','B365A','FTR']]
    df_avg = df_avg.merge(df_odds,on='id_partido',how = 'right')
    
    # Renombrar y limpiar
    df_avg = df_avg.drop(columns=["temporada_local", "Date_local"]).rename(
        columns={"temporada_visitante": "temporada", "Date_visitante": "Date"}
    )
    
    # Eliminar filas con NaN (primeros partidos sin datos previos)
    df_avg = df_avg.dropna()
    
    print(f"✅ Ingeniería de características completa - Columnas: {df_avg.shape[1]}, Filas: {df_avg.shape[0]}")
    return df_avg

def export_data(df,filename):
    """
    Exporta el DataFrame final a un archivo CSV en la carpeta processed.
    """
    os.makedirs(ruta_processed, exist_ok=True)
    filepaht = ruta_processed / filename
    df.to_csv(filepaht, index=False)
    print(f"✅ Datos exportados {filepaht} a {ruta_processed}")


# ------ MAIN ------
def main():
    print('='*50)
    print("PASO 1: Descarga de los datos:")
    print('='*50)
    download_data()
    
    print('='*50)
    print("PASO 2: Consolidar los datos:")
    print('='*50)
    df = cons_data()
    df = clean_data(df)
    
    print('='*50)
    print("PASO 3: Ingeniería de características:")
    print('='*50)
    df_modelado = data_engineering(df)
    
    print('='*50)
    print("PASO 4: Exportar datos para el modelado:")
    print('='*50)
    export_data(df_modelado,"data_modeling.csv")

if __name__ == "__main__":
    main()
