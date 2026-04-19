"""MLflow experiment tracking service."""

import os
from typing import Any

import mlflow

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "../data/mlflow.db")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.abspath(_DEFAULT_DB)}",
)


class MLFlowService:
    def __init__(self, experiment_name: str, run_name: str | None = None, tracking_uri: str = MLFLOW_TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def end(self) -> None:
        mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end()

    @property
    def run_id(self) -> str:
        return self._run.info.run_id
