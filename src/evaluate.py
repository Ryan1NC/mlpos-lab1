from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import tracking
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("evaluate")
FEATURES = Path("data/features/test.csv")
MODEL = Path("models/model.joblib")
METRICS_OUT = Path("reports/metrics.json")
PLOT_OUT = Path("reports/plots/pred_vs_true.csv")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    bundle = joblib.load(MODEL)
    model, feature_names = bundle["model"], bundle["features"]
    frame = pd.read_csv(FEATURES)
    x_test = frame[feature_names]
    y_test = frame[target]
    y_pred = model.predict(x_test)
    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        "n_test": int(len(frame)),
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sample = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).sample(
        n=min(500, len(frame)), random_state=int(params["train"]["seed"])
    )
    sample.to_csv(ensure_parent(PLOT_OUT), index=False)
    tracking.log_metrics({key: value for key, value in metrics.items() if key != "n_test"})
    log.info("метрики: %s", metrics)
    if metrics["r2"] < float(params["evaluate"]["min_r2"]):
        log.error("R2 ниже установленного порога")
        sys.exit(1)


if __name__ == "__main__":
    main()
