"""Необязательная интеграция с MLflow.

Базовый пайплайн работает и без настроенного MLflow. Чтобы включить трекинг,
задай MLFLOW_TRACKING_URI и MLFLOW_TRACKING_USERNAME/PASSWORD в окружении.
"""


def start_run(experiment: str, params: dict) -> None:
    import os

    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    mlflow.start_run()
    mlflow.log_params(params)


def log_model(model, feature_names: list[str], sample) -> None:
    import os

    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    import mlflow
    from mlflow.sklearn import log_model as log_sklearn_model

    log_sklearn_model(model, "model", input_example=sample)


def log_metrics(metrics: dict) -> None:
    import os

    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    import mlflow

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
