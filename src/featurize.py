from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("featurize")
INPUTS = {
    "train": Path("data/processed/train.csv"),
    "test": Path("data/processed/test.csv"),
}
OUTPUTS = {
    "train": Path("data/features/train.csv"),
    "test": Path("data/features/test.csv"),
}


def build(
    frame: pd.DataFrame, target: str, drop: list[str], cyclic_hour: bool
) -> pd.DataFrame:
    features = frame.drop(
        columns=[col for col in drop if col in frame.columns], errors="ignore"
    )
    features = features.drop(columns=["dteday_hour"], errors="ignore")
    if cyclic_hour and "hr" in features.columns:
        hour = features.pop("hr")
        features["hr_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hr_cos"] = np.cos(2 * np.pi * hour / 24)
    ordered = [col for col in features.columns if col != target]
    if target in features.columns:
        ordered.append(target)
    return features[ordered]


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    drop = list(params["data"]["drop"])
    cyclic = bool(params["featurize"]["cyclic_hour"])
    for part, path in INPUTS.items():
        frame = pd.read_csv(path)
        built = build(frame, target, drop, cyclic)
        built.to_csv(ensure_parent(OUTPUTS[part]), index=False)
        log.info("%s: %d строк, %d признаков", part, len(built), built.shape[1] - 1)


if __name__ == "__main__":
    main()
