# FlightOnTime – Data Science

## Descripción
Este repositorio contiene el trabajo de análisis de datos, limpieza,
feature engineering, entrenamiento y evaluación de modelos de Machine Learning
para el proyecto FlightOnTime.

El objetivo es predecir si un vuelo llegará a tiempo o con retraso a partir
de variables operativas y temporales.


## Estructura del proyecto

FlightOnTime – Data Science
Descripción del Proyecto

nocountry-data-science
├── QA_requirements.md
├── README.md
├── docs/
│   └── data/
│       ├── README.md
│       ├── archivo_limpio.csv
│       ├── dataset.csv
│   ├── Evaluación_de_modelos.iynb
│   ├── Evaluación_modelos_jesus.iynb
│   ├── modelo-flighOnTie.iynb
│   ├── limpieza_datos-flighOnTie.iynb


FlightOnTime – Data Science & ML es el componente encargado del análisis de datos, feature engineering, entrenamiento, evaluación y despliegue del modelo de Machine Learning utilizado para predecir si un vuelo llegará a tiempo o con retraso.


flight-delay-ml-api/
├── main.py
├── requirements.txt
├── README.md
├── model/
│   └── modelo_flight_on_time.pkl
├── artifacts/
│   ├── aerolinea_delay_rate.pkl
│   ├── origen_delay_rate.pkl
│   ├── destino_delay_rate.pkl
│   └── global_delay_rate.pkl



Este proyecto expone una API de Machine Learning que recibe información estructurada de un vuelo y devuelve una predicción probabilística, la cual es consumida por el Backend en Java (Spring Boot) del sistema FlightOnTime.


Objetivo del Modelo

Predecir la probabilidad de retraso de un vuelo utilizando variables operativas y temporales, priorizando:

Recall de la clase “Retrasado” (detectar la mayor cantidad de retrasos posibles)

Buen balance entre precisión y estabilidad del modelo

Interpretabilidad y robustez para un entorno de producción

Enfoque de Machine Learning

Tipo de problema: Clasificación binaria

Variable objetivo:

delay

0 → A tiempo

1 → Retrasado

Modelo final: Clasificador entrenado y serializado (.pkl)

Threshold de decisión: 0.5



## Tecnologías Utilizadas

Python 3.10+

Pandas – Manipulación de datos

NumPy – Cálculo numérico

Scikit-learn – Modelado y evaluación

Joblib – Serialización del modelo

FastAPI – Exposición del modelo como API

Pydantic – Validación de datos de entrada

Matplotlib / Seaborn – Visualización (EDA)

Jupyter Notebook – Exploración y entrenamiento

## Dataset
El dataset original y el dataset limpio se encuentran en:
docs/data/


Dataset histórico de vuelos con información operativa y temporal.

Variables Relevantes
Variable	Descripción
aerolinea	Código IATA de la aerolínea
origen	Aeropuerto de origen
destino	Aeropuerto de destino
fecha_partida	Fecha y hora de salida
distancia_km	Distancia del vuelo
delay	Variable objetivo (0 / 1)
Feature Engineering

Se crearon variables adicionales para mejorar el desempeño del modelo:

1) Delay Rates Históricos

aerolinea_delay_rate

origen_delay_rate

destino_delay_rate

global_delay_rate (fallback)

2) Variables Temporales

Bloque horario

madrugada

mañana

tarde

noche

Riesgo horario (mapa de riesgo por bloque)

Fin de semana + noche (feature binaria)

3) Features Finales del Modelo

El modelo recibe exactamente estas variables (en este orden):

[
  aerolinea_delay_rate,
  fin_semana_y_noche,
  bloque_risk,
  distancia_km,
  origen_delay_rate,
  destino_delay_rate
]

Evaluación del Modelo

Las métricas fueron evaluadas tanto en test como en simulación de producción.

Métrica	Test	Producción
Precision (delay)	0.820	0.821
Recall (delay)	0.747	0.747
F1-score	0.782	0.782

Métrica prioritaria: Recall (delay)

API de Machine Learning

El modelo se expone mediante una API construida con FastAPI, consumida por el backend.

Endpoint Principal

POST /predict

Request

{
  "aerolinea": "AA",
  "origen": "JFK",
  "destino": "LAX",
  "fecha_partida": "2024-01-15T14:30:00",
  "distancia_km": 350.0
}


Response

{
  "prevision": "A tiempo",
  "probabilidad": 0.85
}

Ejecución Local
1) Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

2) Instalar dependencias
pip install -r requirements.txt

3) Ejecutar API
uvicorn main:app --reload


API disponible en:

http://localhost:8000


## ML y API del proyecto
Integración con Backend

Relación entre Data Science y API

El entrenamiento y evaluación del modelo se realiza en el repositorio nocountry-data-science.

El modelo final y los artifacts generados son exportados a flight-delay-ml-api.

La API aplica exactamente el mismo orden y definición de features utilizados durante el entrenamiento.

El backend Java consume esta API para exponer la predicción al usuario final.

## Descripción del Flujo

El proceso inicia con el dataset original de vuelos.

Se realiza limpieza y preprocesamiento para generar un dataset confiable.

Se crean variables derivadas mediante feature engineering.

Se entrenan y comparan múltiples modelos.

El modelo final se evalúa priorizando el recall de la clase “Retrasado”.

El modelo y los artifacts se serializan en archivos .pkl.

La API de Machine Learning (FastAPI) carga el modelo y expone el endpoint /predict.

El backend Java consume la API, valida los datos y expone el servicio final.

El usuario final recibe la predicción junto con su probabilidad.
┌───────────────────────────────┐
│        Dataset Original       │
│        (dataset.csv)          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Limpieza y Preprocesamiento  │
│  - Manejo de nulos            │
│  - Corrección de formatos     │
│  - Generación dataset limpio  │
│  (archivo_limpio.csv)         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Feature Engineering       │
│  - Variables temporales       │
│  - Delay rates históricos     │
│  - Selección de features      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Entrenamiento de Modelos    │
│  - Modelos baseline           │
│  - Ajuste de hiperparámetros  │
│  - Selección del modelo final │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    Evaluación del Modelo      │
│  - Precision                 │
│  - Recall (prioritaria)      │
│  - F1-score                  │
│  - Confusion Matrix          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Exportación de Artefactos   │
│  - modelo_flight_on_time.pkl  │
│  - delay rates (.pkl)         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   API de Machine Learning     │
│       (FastAPI)               │
│  - Carga modelo y artifacts  │
│  - Transformación features   │
│  - Endpoint POST /predict    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Backend Java             │
│     (Spring Boot)             │
│  - Validaciones de entrada    │
│  - Manejo de errores          │
│  - Persistencia               │
│  - Exposición del endpoint    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Usuario Final          │
│  - Solicitud de predicción   │
│  - Resultado y probabilidad  │
└───────────────────────────────┘


Descripción de Componentes

main.py
Archivo principal de la aplicación FastAPI.
Contiene:

Definición del esquema de entrada

Carga del modelo entrenado

Transformación de features

Endpoint /predict

requirements.txt
Lista de dependencias necesarias para ejecutar la API.

model/
Contiene el modelo final entrenado:

modelo_flight_on_time.pkl: modelo de Machine Learning serializado y listo para producción.

artifacts/
Archivos auxiliares utilizados durante la inferencia:

Delay rates históricos por aerolínea, aeropuerto de origen y destino

Delay rate global utilizado como fallback para valores no vistos
Esta API es consumida por el backend Java mediante:

https://flightdelaypredictor-api.onrender.com


El backend se encarga de:

Validaciones

Manejo de errores

Persistencia

Exposición pública del endpoint final

signación de Responsabilidades

Brandon Calderón – Data Science Lead

Coordinación general del proyecto de Data Science

Definición de métricas y criterios de evaluación

Feature engineering

Selección del modelo final

Preparación del modelo para producción


Jesús Sanchez Farro – Exploración y Modelos Base

Análisis exploratorio de datos (EDA)

Visualización de variables

Entrenamiento de modelos baseline

Evaluación inicial de métricas


Leonard Espejo Mojica – Limpieza y Preprocesamiento

Limpieza del dataset original

Tratamiento de valores faltantes

Transformación de variables

Generación del dataset limpio


Damian Yaccuzzi  – Evaluación y Validación

Evaluación comparativa de modelos

Análisis de métricas (precision, recall, F1-score)

Confusion matrix

Validación de resultados


Luis Rello Mayor – Documentación y QA

Documentación técnica del proceso

Organización de notebooks

Revisión de consistencia
