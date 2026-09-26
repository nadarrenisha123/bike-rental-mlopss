"""
Automatic retraining entry point.

Re-runs training on the latest data (pulled via DVC in CI), and only
registers/promotes the new model if it beats the current best RMSE.
Intended to be triggered on a schedule (see .github/workflows/retrain.yml)
or manually after new data is added via `dvc add`.
"""
import mlflow

from src.train import (
    EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME,
    load_data,
    train_and_log,
    LinearRegression,
    RandomForestRegressor,
    XGBRegressor,
)


def get_current_best_rmse(client):
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
    if not versions:
        return float("inf")
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("rmse", float("inf"))


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    current_best_rmse = get_current_best_rmse(client)
    print(f"Current production RMSE: {current_best_rmse:.3f}")

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
    print(f"New best candidate: {best_name} (RMSE={best_rmse:.3f})")

    if best_rmse < current_best_rmse:
        print("New model improves on production — registering.")
        mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_REGISTRY_NAME)
    else:
        print("New model did not improve on production — keeping current model.")


if __name__ == "__main__":
    main()
