from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src import tracking
from src.utils import ensure_parent, get_logger, load_params, set_seed

log = get_logger("train")
FEATURES = Path("data/features/train.csv")
MODEL_OUT = Path("models/model.joblib")


def make_model(cfg: dict):
    name = cfg["model"]
    common = {
        "n_estimators": int(cfg["n_estimators"]),
        "max_depth": int(cfg["max_depth"]),
        "min_samples_leaf": int(cfg["min_samples_leaf"]),
        "random_state": int(cfg["seed"]),
    }
    if name == "random_forest":
        return RandomForestRegressor(**common, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            **common, learning_rate=float(cfg["learning_rate"])
        )
    raise ValueError(f"Неизвестная модель: {name}")


def main() -> None:
    params = load_params()
    cfg = params["train"]
    target = params["data"]["target"]
    set_seed(int(cfg["seed"]))
    frame = pd.read_csv(FEATURES)
    x_train = frame.drop(columns=[target])
    y_train = frame[target]
    tracking.start_run(params["mlflow"]["experiment"], dict(cfg))
    model = make_model(cfg)
    model.fit(x_train, y_train)
    joblib.dump(
        {"model": model, "features": list(x_train.columns)},
        ensure_parent(MODEL_OUT),
    )
    tracking.log_model(model, list(x_train.columns), x_train.head(5))
    log.info("обучено: %s на %d строках", cfg["model"], len(frame))


if __name__ == "__main__":
    main()
