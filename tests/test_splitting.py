from __future__ import annotations

import numpy as np
import pandas as pd

from gzcmd.splitting import SplitSpec, split_train_test_indices


def test_row_split_is_stratified() -> None:
    df = pd.DataFrame({"x": range(40)})
    y = pd.Series([0] * 20 + [1] * 20)

    train_idx, test_idx = split_train_test_indices(
        df,
        y,
        spec=SplitSpec(split_by="row", test_size=0.25, seed=123),
    )

    y_test = y.iloc[test_idx].to_numpy()
    assert len(test_idx) == 10
    assert int(np.sum(y_test == 1)) == 5
    assert int(np.sum(y_test == 0)) == 5


def test_group_split_keeps_groups_intact() -> None:
    # 12 groups, 4 rows each.
    groups = [f"G{i}" for i in range(12) for _ in range(4)]
    df = pd.DataFrame({"COMPREC": groups})

    # First 6 groups have at least one positive, last 6 are all-negative.
    y = []
    for gi in range(12):
        if gi < 6:
            y.extend([1, 0, 0, 0])
        else:
            y.extend([0, 0, 0, 0])
    y_s = pd.Series(y)

    train_idx, test_idx = split_train_test_indices(
        df,
        y_s,
        spec=SplitSpec(
            split_by="comprec", test_size=0.33, seed=42, group_stratify=True
        ),
    )

    train_groups = set(df.iloc[train_idx]["COMPREC"].unique().tolist())
    test_groups = set(df.iloc[test_idx]["COMPREC"].unique().tolist())
    assert train_groups.isdisjoint(test_groups)
    assert len(train_idx) + len(test_idx) == len(df)
