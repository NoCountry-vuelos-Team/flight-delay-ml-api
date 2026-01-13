
# Importación de bibliotecas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import os
import numpy as np

app = FastAPI(title="Flight Delay Predictor API")

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Cargar modelo (UNA sola vez)
modelo = joblib.load(
    os.path.join(MODEL_DIR, "modelo_flight_on_time.pkl")
)

#Cargar .pkl
aerolinea_delay_rate = joblib.load(
    os.path.join(ARTIFACTS_DIR, "aerolinea_delay_rate.pkl")
)

origen_delay_rate = joblib.load(
    os.path.join(ARTIFACTS_DIR, "origen_delay_rate.pkl")
)

destino_delay_rate = joblib.load(
    os.path.join(ARTIFACTS_DIR, "destino_delay_rate.pkl")
)

global_delay_rate = joblib.load(
    os.path.join(ARTIFACTS_DIR, "global_delay_rate.pkl")
)


# Transformacion de features

# Estación del año
def obtener_bloque_horario(fecha: datetime) -> str:
    hora = fecha.hour
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    else:
        return "noche"


# Riesgo horario
BLOQUE_RISK_MAP = {
    "madrugada": 2,
    "mañana": 4,
    "tarde": 1,
    "noche": 3
}


# Horario
def es_fin_semana_y_noche(fecha: datetime) -> int:
    es_fin_semana = fecha.weekday() >= 5
    es_noche = fecha.hour >= 20 or fecha.hour < 6
    return int(es_fin_semana and es_noche)






class FlightInput(BaseModel):
    aerolinea: str
    origen: str
    destino: str
    fecha_partida: datetime
    distancia_km: float

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/v1/predict")
def predict(data: FlightInput):
    try:
        aerolinea_rate = aerolinea_delay_rate.get(
            data.aerolinea, global_delay_rate
        )
        origen_rate = origen_delay_rate.get(
            data.origen, global_delay_rate
        )
        destino_rate = destino_delay_rate.get(
            data.destino, global_delay_rate
        )

        fin_semana_y_noche = es_fin_semana_y_noche(data.fecha_partida)
        bloque_horario = obtener_bloque_horario(data.fecha_partida)
        bloque_risk = BLOQUE_RISK_MAP[bloque_horario]

        X = np.array([[ 
            aerolinea_rate,
            fin_semana_y_noche,
            bloque_risk,
            data.distancia_km,
            origen_rate,
            destino_rate
        ]])

        proba = modelo.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)

        return {
            "prevision": "Retrasado" if pred == 1 else "A tiempo",
            "probabilidad": round(float(proba), 2)
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="prediction_failed"
        )
