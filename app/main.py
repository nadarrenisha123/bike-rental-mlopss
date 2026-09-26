"""
FastAPI service that loads the best registered MLflow model and exposes
a /predict endpoint for bike rental demand forecasting.
"""

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = "bike-rental-best-model"
MODEL_STAGE = "None"  # use "Production" once you promote a model in the registry

app = FastAPI(title="Bike Rental Demand Prediction API")

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


class BikeRentalFeatures(BaseModel):
    season: int
    yr: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    hum: float
    windspeed: float


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: BikeRentalFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)

    return {"predicted_count": float(prediction[0])}