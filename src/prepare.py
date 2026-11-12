from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import validate_raw
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("prepare")
TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")


def split(
    frame: pd.DataFrame, test_size: float, by_time: bool, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1")
    if by_time:
        frame = frame.sort_values("dteday_hour").reset_index(drop=True)
    else:
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    edge = int(len(frame) * (1 - test_size))
    return frame.iloc[:edge].copy(), frame.iloc[edge:].copy()


def main() -> None:
    params = load_params()
    cfg = params["data"]
    raw = pd.read_csv(cfg["raw"], parse_dates=["dteday"])
    raw = validate_raw(raw)
    if raw[cfg["target"]].isna().any():
        raise ValueError("В целевой переменной есть пропуски")

    raw["dteday_hour"] = raw["dteday"] + pd.to_timedelta(raw["hr"], unit="h")
    raw = raw.drop_duplicates(subset="dteday_hour", keep="last")
    train, test = split(
        raw,
        float(cfg["test_size"]),
        bool(cfg["split_by_time"]),
        int(params["train"]["seed"]),
    )
    train.to_csv(ensure_parent(TRAIN_OUT), index=False)
    test.to_csv(ensure_parent(TEST_OUT), index=False)
    log.info("обучение: %d строк; тест: %d строк", len(train), len(test))


if __name__ == "__main__":
    main()
