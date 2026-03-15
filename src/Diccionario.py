import requests

url = "https://www.football-data.co.uk/notes.txt"
ruta = r"c:\Users\brger\OneDrive\Desktop\Project\Football_Prediction\Diccionario_Data_Partidos.txt"

response = requests.get(url)

with open(ruta, "w", encoding="utf-8") as f:
    f.write(response.text)

print("✅ notes.txt guardado")