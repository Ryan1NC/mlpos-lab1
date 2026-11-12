from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
    "windspeed", "casual", "registered", "cnt",
}


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Датасет пустой")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("В исходных данных обнаружены пропуски")
    if not frame["cnt"].ge(0).all():
        raise ValueError("В cnt обнаружены отрицательные значения")
    if not frame["hr"].between(0, 23).all():
        raise ValueError("Значения hr должны находиться в диапазоне 0..23")
    return frame.copy()
