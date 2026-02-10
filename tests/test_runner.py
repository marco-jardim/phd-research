from __future__ import annotations

import pandas as pd

from gzcmd.calibration import compute_p_cal, fit_platt_from_df


def test_compute_p_cal_stub_clips_and_validates() -> None:
    df = pd.DataFrame({"nota final": [0.0, 5.0, 10.0]})
    p = compute_p_cal(df, method="stub")
    assert p.min() > 0.0
    assert p.max() < 1.0


def test_fit_platt_from_df_returns_sane_model() -> None:
    df = pd.DataFrame(
        {
            "nota final": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 6.0, 7.0, 3.0, 4.0],
            "TARGET": [0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
        }
    )
    model = fit_platt_from_df(df)
    assert model.slope != 0.0
