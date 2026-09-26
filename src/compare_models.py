"""
Pull all runs for the bike-rental-demand experiment from MLflow and print
a comparison table of RMSE/MAE, highlighting the best run.
"""
import mlflow

EXPERIMENT_NAME = "bike-rental-demand"


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"No experiment named '{EXPERIMENT_NAME}' found. Run train.py first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
    )

    if not runs:
        print("No runs logged yet.")
        return

    print(f"{'Model':<20}{'RMSE':<12}{'MAE':<12}{'Run ID'}")
    print("-" * 70)
    for run in runs:
        name = run.data.params.get("model_type", "unknown")
        rmse = run.data.metrics.get("rmse", float("nan"))
        mae = run.data.metrics.get("mae", float("nan"))
        print(f"{name:<20}{rmse:<12.3f}{mae:<12.3f}{run.info.run_id}")

    best = runs[0]
    print(f"\nBest model: {best.data.params.get('model_type')} "
          f"(RMSE={best.data.metrics.get('rmse'):.3f})")


if __name__ == "__main__":
    main()
