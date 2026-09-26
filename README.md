# Bike Rental Demand Forecasting — MLOps Mini Project

Predicts hourly bike rental demand from weather, season, holiday, and time
features, using the UCI/Kaggle "Bike Sharing Demand" dataset.

## Architecture

```
Raw CSV --(DVC)--> data/bike_rentals.csv
                        |
                        v
                 src/train.py  --(MLflow tracking)--> mlruns/
                        |
                        v
            Best model registered in MLflow Model Registry
                        |
                        v
              app/main.py  (FastAPI /predict)
                        |
                        v
                    Docker container
                        |
                        v
        GitHub Actions CI (lint + test on push)
                        |
                        v
     retrain.py (scheduled via GitHub Actions cron)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Track the dataset with DVC
dvc init
dvc add data/bike_rentals.csv
dvc remote add -d storage <your-remote-url>   # e.g. Google Drive, S3
dvc push
```

## Train + track experiments

```bash
python src/train.py
mlflow ui   # view experiment comparison at http://localhost:5000
```

`train.py` trains Linear Regression, Random Forest, and XGBoost, logs
params/metrics/artifacts for each to MLflow, and registers the
lowest-RMSE model as `bike-rental-best-model` in the Model Registry.

## Serve predictions

```bash
uvicorn app.main:app --reload
# POST http://localhost:8000/predict
```

## Docker

```bash
docker build -t bike-rental-api .
docker run -p 8000:8000 bike-rental-api
```

## Automatic retraining

`retrain.py` re-runs the training pipeline and re-registers the model if
performance (RMSE) is better than the current production model. Wired up
in `.github/workflows/retrain.yml` to run on a weekly schedule.

## Model comparison

```bash
python src/compare_models.py
```

Prints a table of RMSE/MAE across all logged MLflow runs and highlights
the current best model.

## Dataset

Kaggle "Bike Sharing Demand": https://www.kaggle.com/c/bike-sharing-demand
Place the raw CSV at `data/bike_rentals.csv` before running `dvc add`.
