"""Protocol checks with fixed calendar, geometry and missing-value examples."""

import numpy as np
import pandas as pd
import pytest

from src.probability_samples import (
    assert_causal_history, build_date_manifest, build_grid_registry,
    calendar_dates, common_finite_mask, event_labels, posthoc_event_groups,
    required_history, sample_id, spatial_partition, year_split,
)


def test_complete_summers_and_utc_boundaries():
    dates = pd.date_range("2019-06-01", "2019-08-31").append(
        pd.date_range("2020-06-01", "2020-08-31"))
    manifest = build_date_manifest(dates)
    assert manifest.groupby("year")[["lag1_eligible", "lag2_eligible", "common_eligible"]].sum().values.tolist() == [[85, 84, 84], [85, 84, 84]]
    row = manifest.set_index("date").loc["2020-06-09"]
    assert row["split"] == "final_test"
    assert row["issue_utc"] == "2020-06-08T16:00:00Z"
    assert row["target_end_bjt_exclusive"] == "2020-06-10T00:00:00+08:00"
    assert row["lag2_feature_start"] == "2020-06-01"
    assert row["lag2_feature_end"] == "2020-06-07"
    assert row["lag2_source_utc_last_hour"] == "2020-06-07T15:00:00Z"
    assert row["lag1_source_utc_end_exclusive"] == row["issue_utc"]
    assert not manifest.set_index("date").loc["2020-06-08", "common_eligible"]
    assert "2020-06-01" not in set(manifest.date)


def test_missing_date_cannot_be_replaced_by_previous_row_or_previous_summer():
    dates = pd.date_range("2022-06-01", "2022-06-20").difference(pd.DatetimeIndex(["2022-06-05"]))
    manifest = build_date_manifest(dates).set_index("date")
    assert "2022-06-10" not in manifest.index
    assert manifest.loc["2022-06-13", "lag1_eligible"]
    assert not manifest.loc["2022-06-13", "lag2_eligible"]
    assert manifest.loc["2022-06-14", "common_eligible"]
    with pytest.raises(ValueError, match="same JJA"):
        required_history("2022-06-07", 1)
    with pytest.raises(ValueError, match="causal history"):
        assert_causal_history("2022-06-08", pd.date_range("2021-08-25", "2021-08-31"), 1)


def test_d2_guard_rejects_d1_even_with_seven_days():
    assert_causal_history("2022-07-10", pd.date_range("2022-07-02", "2022-07-08"), 2)
    with pytest.raises(ValueError, match="causal history"):
        assert_causal_history("2022-07-10", pd.date_range("2022-07-03", "2022-07-09"), 2)
    with pytest.raises(ValueError, match="Duplicate"):
        calendar_dates(["2022-07-01", "2022-07-01"])
    with pytest.raises(ValueError, match="midnight"):
        calendar_dates(["2022-07-01T01:00"])


def test_closed_spatial_boundaries_and_outer_buffer_corners():
    lon = [118, 121, 117.5, 121.5, 117.4999, 119, 119, 121.5]
    lat = [29, 32, 28.5, 32.5, 30, 28.5, 28.4999, 29]
    assert spatial_partition(lon, lat, (118, 121, 29, 32), .5).tolist() == [
        "holdout", "holdout", "buffer", "buffer", "train", "buffer", "train", "buffer"]


def test_grid_ids_stable_when_axes_reverse_and_duplicate_axes_rejected():
    kwargs = dict(holdout_bounds=(118, 121, 29, 32), buffer_degrees=.5)
    grid = build_grid_registry([29, 30], [118, 119], np.array([[True, False], [True, True]]), **kwargs)
    reversed_grid = build_grid_registry([30, 29], [119, 118], np.array([[True, True], [False, True]]), **kwargs)
    pd.testing.assert_frame_equal(grid, reversed_grid)
    ids = [sample_id(g, "2022-07-01") for g in grid.grid_id]
    assert len(set(ids)) == 3
    with pytest.raises(ValueError, match="unique"):
        build_grid_registry([29, 29], [118], np.ones((2, 1), dtype=bool), **kwargs)


def test_threshold_missing_and_posthoc_groups():
    values = [40.5999, 40.6, np.nan, np.inf, -np.inf]
    np.testing.assert_equal(event_labels(values), [0, 1, np.nan, np.nan, np.nan])
    assert posthoc_event_groups([39, 41, 41, 39, np.nan], [40.6, 42, 39, 38, 42]).tolist() == [
        "new_onset", "persistent", "resolved", "non_event", "missing"]


def test_common_mask_requires_union_even_for_temperature_only_ablation():
    features = {"temperature": np.ones((4, 7)), "dewpoint_mean": np.ones((4, 7))}
    features["dewpoint_mean"][1, 3] = np.nan
    features["temperature"][2, 0] = np.inf
    assert common_finite_mask(features, ["temperature", "dewpoint_mean"], [41, 39, 42, np.nan]).tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="Missing required"):
        common_finite_mask(features, ["temperature", "night"], [41, 39, 42, 40])
    with pytest.raises(ValueError, match="axes mismatch"):
        common_finite_mask({"temperature": np.ones((3, 7))}, ["temperature"], [41, 39, 42, 40])


def test_frozen_year_splits():
    assert [year_split(y) for y in [2000, 2014, 2015, 2017, 2018, 2019, 2020, 2024]] == [
        "train", "train", "validation", "validation", "calibration", "calibration", "final_test", "final_test"]
    with pytest.raises(ValueError, match="outside"):
        year_split(2025)


def test_factorized_cli_uses_coordinates_and_static_mask_without_label_completeness(tmp_path):
    import json
    import xarray as xr
    from tools.probability_sample_manifest import main

    source = tmp_path / "dates.nc"
    mask = tmp_path / "mask.nc"
    output = tmp_path / "manifest"
    xr.Dataset({"hi": (("date", "latitude", "longitude"), np.full((92, 2, 2), np.nan))},
               coords={"date": pd.date_range("2024-06-01", "2024-08-31"),
                       "latitude": [29, 30], "longitude": [117, 118]}).to_netcdf(source)
    xr.Dataset({"land_mask": (("latitude", "longitude"), [[1, 1], [0, np.nan]])},
               coords={"latitude": [29, 30], "longitude": [117, 118]}).to_netcdf(mask)
    main(["--dates-netcdf", str(source), "--time-name", "date", "--mask-netcdf", str(mask),
          "--mask-variable", "land_mask", "--output-dir", str(output)])
    counts = pd.read_csv(output / "yearly_counts.csv").iloc[0]
    assert counts.common_structural_cell_days == 168
    assert counts.common_holdout_cell_days == 84
    schema = json.loads((output / "schema.json").read_text())
    assert schema["qualification"] == "STRUCTURAL_ONLY_NOT_FEATURE_COMPLETE"
    assert schema["grid_count"] == 2
