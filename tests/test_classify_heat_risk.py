"""Tests for the classify_heat_risk step helpers."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from src.classify_heat_risk import is_partial_coverage  # noqa: E402


def test_partial_coverage_tolerates_float_rounding():
    coverage = pd.Series([100.0, 99.99999999999999, 99.9, 0.0])
    assert is_partial_coverage(coverage).tolist() == [False, False, True, True]
