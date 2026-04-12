# Análisis Fútbol - Predicción La Liga

Proyecto end-to-end de Machine Learning para predecir resultados de partidos de La Liga española (Home / Draw / Away), abarcando desde la ingesta de datos hasta el despliegue del modelo.

## Descripción del Problema

Predecir el resultado de un partido de fútbol es un problema de clasificación multiclase con un desafío conocido en la industria:  **los empates son casi impredecibles** . Incluso las casas de apuestas como Bet365, con modelos sofisticados y millones en datos, obtienen 0% de recall en empates — simplemente nunca los predicen.

Este proyecto aborda ese desafío utilizando ~25 temporadas de datos históricos de La Liga (2000-2026), con un enfoque especial en lograr predicciones balanceadas entre las tres clases.

## Arquitectura del Proyecto

```
Football_Prediction/
├── Data/
│   ├── raw/                  ← 26 CSVs de football-data.co.uk
│   └── processed/            ← data_modeling.csv (7,717 partidos × 25 features)
├── SQL/
│   └── 01_creacion_tablas.sql ← Star schema (dim_equipo, dim_fecha, fact_partido)
├── Script/
│   ├── 01_Scrapping.ipynb     ← Descarga de datos
│   ├── 02_Extract.ipynb       ← ETL: limpieza + carga a SQL Server
│   ├── 03_EDA.ipynb           ← Análisis exploratorio
│   ├── 04_FeatureEngineering.ipynb ← Rolling averages + unpivot
│   └── 05_Modeling.ipynb      ← Entrenamiento y evaluación
├── model/
│   └── lgbm_balanced.pkl      ← Modelo final exportado
├── src/                       ← Pipeline modularizado (producción)
└── README.md
```

## Datos

**Fuente:** [football-data.co.uk](https://www.football-data.co.uk/spainm.php) — 26 archivos CSV con estadísticas de partidos de La Liga desde la temporada 2000/01 hasta 2025/26.

**Base de datos:** SQL Server Express con modelo estrella:

* `dim_equipo` — 52 equipos históricos
* `dim_fecha` — fechas con atributos temporales
* `fact_partido` — partidos con estadísticas y cuotas

**Dataset final:** 7,717 partidos con 25 features (temporadas 2005/06 a 2025/26, filtrando las primeras 5 por falta de estadísticas completas).

## Feature Engineering

El enfoque central es capturar el **momentum reciente** de cada equipo, no su identidad:

1. **Unpivot** de `fact_partido` en vista por equipo (local + visitante en una sola tabla cronológica)
2. **Rolling averages** con ventana de 5 partidos y `shift(1)` para evitar data leakage
3. **Re-pivot** al formato partido con sufijos `_local` / `_visitante`

**25 features finales:**

| Tipo                  | Features                                                                                                      | Cantidad |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | -------- |
| Rolling avg local     | Goles, goles rival, tiros, tiros al arco, corners, faltas, amarillas, rojas, goles HT, goles HT rival, puntos | 11       |
| Rolling avg visitante | Las mismas 11 estadísticas                                                                                   | 11       |
| Cuotas Bet365         | B365H, B365D, B365A                                                                                           | 3        |

**Decisiones de diseño:**

* Los rolling averages son **continuos entre temporadas** (no se resetean) para preservar la forma del equipo
* Los nombres de equipo se **excluyen como features** — los equipos que ascienden/descienden no tendrían historial, y el rolling average ya codifica la forma
* Solo se retienen las cuotas de **B365** (las demás casas tienen 40-93% de valores nulos)

## Metodología

**Split temporal estricto** (sin data leakage):

* **Train:** Temporadas 2005/06 a 2023/24
* **Test:** Temporada 2024/25 (380 partidos)
* **Validation:** Temporada 2025/26 (265 partidos, temporada en curso)

**Cross-validation:** TimeSeriesSplit con 5 folds sobre el set de entrenamiento.

**Búsqueda de hiperparámetros:** GridSearchCV para todos los modelos.

## Resultados

### Comparación de Modelos (Test — Temporada 2024/25)

| Modelo                        | Accuracy         | Recall (H)     | Recall (D)     | Recall (A)     |
| ----------------------------- | ---------------- | -------------- | -------------- | -------------- |
| B365 Baseline                 | 54.47%           | 0.87           | 0.00           | 0.58           |
| Decision Tree (tuneado)       | 56.05%           | 0.86           | 0.00           | 0.57           |
| Random Forest (tuneado)       | 55.26%           | 0.85           | 0.00           | 0.58           |
| LightGBM                      | 53.95%           | 0.93           | 0.00           | 0.42           |
| **LightGBM (balanced)** | **52.37%** | **0.57** | **0.39** | **0.57** |

### Modelo Elegido: LightGBM Balanced

Se prioriza el **equilibrio entre clases** sobre el accuracy global. Es el único modelo capaz de predecir empates.

**Validación (temporada 2025/26):**

* Accuracy: 50.6%
* Draw recall: 0.44 (mejoró vs test)
* Sin overfitting — generalización consistente

### Distribución de clases

* Home (H): ~47%
* Away (A): ~28%
* Draw (D): ~25%

## Hallazgos Clave

1. **Techo natural de predicción:** Todos los modelos sin balanceo orbitan entre 54-56%, apenas superando a B365. Las features actuales tienen un límite natural.
2. **El Draw es un problema del dominio, no del modelo:** Ni B365 ni ningún modelo estándar predice empates. Se requiere penalización explícita (`class_weight='balanced'`) para lograrlo, a costa de accuracy global.
3. **El accuracy no es la métrica correcta:** Un modelo con 56% de accuracy que solo predice H y A es menos útil que uno con 50% que distribuye predicciones entre las tres clases.
4. **Promediar porcentajes entre temporadas es un error estadístico:** Las proporciones deben calcularse como proporción ponderada, no como promedio de promedios (relacionado con la Paradoja de Simpson).
5. **Los H2H (head-to-head) no aportan:** Con solo ~2 enfrentamientos por temporada por pareja de equipos, los datos son demasiado escasos. El rolling form general es más robusto.

## Stack Tecnológico

* **Python** — pandas, numpy, scikit-learn, LightGBM, matplotlib, seaborn
* **SQL Server Express** — modelo estrella para almacenamiento y EDA
* **Jupyter Notebooks** — desarrollo y exploración
* **Git/GitHub** — control de versiones

## Roadmap

* [X] Ingesta de datos (scraping football-data.co.uk)
* [X] ETL y carga a SQL Server (star schema)
* [X] EDA
* [X] Feature Engineering (rolling averages)
* [X] Modelamiento (DT → RF → LightGBM → LightGBM balanced)
* [X] Exportación del modelo (.pkl)
* [X] Modularización del pipeline (`src/`)
* [X] Tests (pytest)
* [ ] API de predicción (FastAPI)
* [ ] Containerización (Docker)
* [ ] CI/CD (GitHub Actions)

## 📄 Licencia

MIT

## 👤 Autor

**Brandon Gerald** — [GitHub](https://github.com/geraldb1)
