"""
FastAPI service that loads the best registered MLflow model and exposes
a /predict endpoint for bike rental demand forecasting.
"""
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "bike-rental-best-model"
MODEL_STAGE = "None"

app = FastAPI(title="Bike Rental Demand Prediction API")

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    print(f"Warning: could not load model at startup: {e}")
    model = None


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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)
    return {"predicted_count": float(prediction[0])}