import requests
from src.config import ruta_base
url = "https://www.football-data.co.uk/notes.txt"

ruta = ruta_base / "Diccionario_Data_Partidos.txt"
response = requests.get(url)

with open(ruta, "w", encoding="utf-8") as f:
    f.write(response.text)

print("✅ notes.txt guardado")