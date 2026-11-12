import pandas as pd

from src.featurize import build


def test_cyclic_hour_features():
    frame = pd.DataFrame({"hr": [0, 6, 12, 18], "cnt": [1, 2, 3, 4]})
    result = build(frame, "cnt", [], True)
    assert "hr_sin" in result.columns
    assert "hr_cos" in result.columns
    assert "hr" not in result.columns
