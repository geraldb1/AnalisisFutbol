from pathlib import Path
import datetime
# ── Rutas del Proyecto ─────────────────────────────────────
# Path(__file__) = ubicación de este archivo (src/Config.py)
# .resolve().parent.parent = sube 2 niveles → raíz del proyecto
# Funciona en cualquier máquina sin cambiar nada

ruta_base = Path(__file__).resolve().parent.parent
ruta_raw = ruta_base / "data" / "raw"
ruta_processed = ruta_base / "data" / "processed"
ruta_scores = ruta_base / "data" / "scores"
ruta_model = ruta_base / "model"



# ─────────────────── Temporadas ────────────────────────────────
year = datetime.datetime.now().year
TEMPORADAS = []
for anio in range(2000, year):
    inicio = str(anio)[2:]
    fin = str(anio + 1)[2:]
    TEMPORADAS.append(inicio + fin)

# Temporadas sin stats completas (a filtrar)
FILTRO_TEMPORADAS = ["0001", "0102", "0203", "0304", "0405"]


# ── Split temporal ─────────────────────────────────────────
TEST_SEASON = int(TEMPORADAS[-2])  # Penúltima temporada completa para test
VAL_SEASON = int(TEMPORADAS[-1])   # Última temporada para validacion (aunque no esté completa, se usará para validar el modelo en datos recientes)



# ── Mapeo de resultado ─────────────────────────────────────
MAP_RESULTADO = {"H": 0, "D": 1, "A": 2}
MAP_INVERSO = {0: "H", 1: "D", 2: "A"}

# ── Features del modelo ───────────────────────────────────
MODEL_FEATURES = [
    "Goles_avg5_local", "goles_rival_avg5_local",
    "tiros_avg5_local", "tiros_al_arco_avg5_local",
    "corners_avg5_local", "faltas_avg5_local",
    "amarillas_avg5_local", "rojas_avg5_local",
    "goles_ht_avg5_local", "goles_ht_rival_avg5_local",
    "resultado_avg5_local",
    "Goles_avg5_visitante", "goles_rival_avg5_visitante",
    "tiros_avg5_visitante", "tiros_al_arco_avg5_visitante",
    "corners_avg5_visitante", "faltas_avg5_visitante",
    "amarillas_avg5_visitante", "rojas_avg5_visitante",
    "goles_ht_avg5_visitante", "goles_ht_rival_avg5_visitante",
    "resultado_avg5_visitante",
    "B365H", "B365D", "B365A",
]