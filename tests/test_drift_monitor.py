from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from gzcmd.config import (
    GZCMDConfig,
    MonitoringConfig,
)
from gzcmd.drift_monitor import calculate_psi, calculate_ks, DriftMonitor, DriftMetric


def test_calculate_psi_identical_dist():
    a = np.random.normal(0, 1, 1000)
    b = a.copy()
    psi = calculate_psi(a, b)
    assert psi < 0.001


def test_calculate_psi_drifted_dist():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(2, 1, 1000)  # Shifted mean
    psi = calculate_psi(a, b)
    assert psi > 0.1  # Should be significant


def test_calculate_ks_identical():
    a = np.random.normal(0, 1, 1000)
    b = a.copy()
    ks = calculate_ks(a, b)
    assert ks < 0.05


def test_calculate_ks_drifted():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(1, 1, 1000)
    ks = calculate_ks(a, b)
    assert ks > 0.1


def test_drift_monitor_check():
    # Mock config since GZCMDConfig is complex to instantiate fully just for this
    mock_cfg = Mock()
    mock_cfg.monitoring.enabled = True
    mock_cfg.monitoring.monitored_fields = ["score"]
    mock_cfg.monitoring.metrics = {
        "ks": {"threshold_warn": 0.1, "threshold_critical": 0.2},
        "psi": {"threshold_warn": 0.1, "threshold_critical": 0.2},
    }

    monitor = DriftMonitor(mock_cfg)

    df_base = pd.DataFrame({"score": np.random.normal(0, 1, 100)})
    df_curr = pd.DataFrame({"score": np.random.normal(0, 1, 100)})  # Similar

    metrics = monitor.check_drift(df_base, df_curr)
    assert len(metrics) == 2
    # Just check structure, status depends on random data
    assert metrics[0].metric in ["ks", "psi"]
