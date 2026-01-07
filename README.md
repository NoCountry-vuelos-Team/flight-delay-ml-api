# Flight Delay ML Inference API

Microservicio de inferencia para predecir retrasos de vuelos.

## Endpoint

### POST /predict

**Input**
```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}


Output

{
  "prevision": "Retrasado",
  "probabilidad": 0.78
}

**Tech stack

FastAPI

Scikit-learn

Joblib**
