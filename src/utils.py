from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise TypeError(f"{path}: ожидался словарь параметров")
    return params


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    return np.random.default_rng(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
