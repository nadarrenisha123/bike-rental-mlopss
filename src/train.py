"""
Train multiple models on the bike rental dataset, log each run to MLflow,
and register the best (lowest RMSE) model to the MLflow Model Registry.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_PATH = "data/bike_rentals.csv"
TARGET_COL = "cnt"
EXPERIMENT_NAME = "bike-rental-demand"
MODEL_REGISTRY_NAME = "bike-rental-best-model"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop columns not used as features:
    # - instant: just a row index
    # - dteday: raw date string; yr/mnth/weekday already capture time info
    # - casual / registered: these sum to "cnt" (the target), so keeping
    #   them would leak the answer directly into the model
    # - atemp: near-duplicate of "temp", dropped to keep features simple
    drop_cols = [c for c in ["instant", "dteday", "casual", "registered", "atemp"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return rmse, mae


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        rmse, mae = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        if name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        print(f"{name}: RMSE={rmse:.3f}  MAE={mae:.3f}")
        return rmse, mlflow.active_run().info.run_id


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = load_data()

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "xgboost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    }

    results = []
    for name, model in candidates.items():
        rmse, run_id = train_and_log(name, model, X_train, y_train, X_test, y_test)
        results.append((name, rmse, run_id))

    best_name, best_rmse, best_run_id = min(results, key=lambda r: r[1])
    print(f"\nBest model: {best_name} (RMSE={best_rmse:.3f})")

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)


if __name__ == "__main__":
    main()