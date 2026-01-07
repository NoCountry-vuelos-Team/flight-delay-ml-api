# FlightOnTime – Modelo Predictivo de Retrasos de Vuelos

## Descripción general

Este repositorio contiene el modelo de Machine Learning utilizado para predecir si un vuelo tendrá retraso o no. El modelo fue entrenado a partir de un **dataset construido y procesado previamente con Python**, utilizando datos crudos que fueron limpiados, transformados y enriquecidos mediante un pipeline de preparación de datos.

El entrenamiento y la evaluación se realizaron con scikit-learn, y el modelo está pensado para ser exportado y consumido desde una API (por ejemplo, con FastAPI) dentro de un proyecto mayor.

---

## Objetivo del modelo

Predecir la probabilidad de retraso de un vuelo con base en información conocida antes del despegue, permitiendo:

* Mejor toma de decisiones operativas
* Comunicación temprana con clientes
* Optimización de recursos

---

## Modelo utilizado

Se utilizó un Gradient Boosting Classifier, seleccionado por su buen balance entre:

* Capacidad predictiva
* Interpretabilidad
* Robustez frente a relaciones no lineales

### Hiperparámetros principales

* n_estimators = 200
* learning_rate = 0.05
* max_depth = 3
* random_state = 42

---

## Variables de entrada (features)

El modelo fue entrenado utilizando las siguientes variables:

* aerolinea_delay_rate
* fin_semana_y_noche
* bloque_risk
* distancia_km
* origen_delay_rate
* destino_delay_rate

### Variable objetivo

* retraso (0 = no retrasado, 1 = retrasado)

---

## Preparación de datos

El dataset utilizado para el entrenamiento **no fue consumido directamente en su forma original**. Antes de entrenar el modelo, los datos pasaron por un proceso de preparación realizado en Python que incluyó:

* Limpieza de datos
* Selección de variables relevantes
* Eliminación de columnas no necesarias
* Construcción de métricas agregadas (delay rates)

El resultado de este proceso fue un dataset final optimizado para el entrenamiento del modelo predictivo.

---

## Entrenamiento y evaluación

El dataset procesado fue dividido en conjuntos de entrenamiento y prueba.

Durante la evaluación se utilizaron métricas estándar de clasificación:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

El modelo mostró un desempeño consistente en el conjunto de prueba, siendo adecuado para su uso en un entorno de demostración o hackatón.

---

## Consideraciones importantes

* El modelo no reentrena automáticamente
* Los datos de entrada deben ser preprocesados de la misma forma que en el entrenamiento
* Para producción real, se recomienda encapsular el modelo en un Pipeline

---

## Tecnologías utilizadas

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* joblib

---

## Autor

Modelo desarrollado como parte de un proyecto de Data Science / Hackatón, con fines educativos y demostrativos.

---

## Licencia

Uso libre para fines académicos y demostrativos.
