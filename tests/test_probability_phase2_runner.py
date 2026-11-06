"""Tiny end-to-end experiment proves frozen C reuse and training-only scaling."""

import json

import numpy as np
import pandas as pd
import pytest


def test_tiny_experiment_selects_once_and_never_fits_validation_scaler(tmp_path):
    pytest.importorskip("sklearn")
    from tools.probability_phase2_baselines import main
    from tools.probability_phase2_features import dump, sha256

    features = tmp_path/"features"
    registry = tmp_path/"grid.csv"
    output = tmp_path/"experiment"
    pd.DataFrame({"grid_id": ["a"], "latitude": [30.], "longitude": [115.],
                  "spatial_role": ["train"]}).to_csv(registry, index=False)
    groups = [group for group, count in (("C", 6), ("T", 27), ("M", 54), ("NT", 3),
                                        ("NM", 2), ("RT", 4), ("RM", 8), ("H", 13))
              for _ in range(count)]
    feature_registry = {"names": [f"synthetic_{i}" for i in range(117)], "groups": groups}
    for year in range(2000, 2018):
        directory = features/str(year)
        directory.mkdir(parents=True)
        x = np.tile(np.arange(4.)[:, None], (1, 117)).astype(np.float32)
        if year > 2014:
            x += 100  # A visible validation shift must not alter fitted means.
        path = directory/"temporal.npz"
        np.savez_compressed(path, X1=x, X2=x, day_index=np.arange(4), grid_index=np.zeros(4, dtype=int),
                            y=np.array([0, 0, 1, 1]), previous_y=np.array([0, 1, 0, 1]),
                            state1=np.array([0, 1, 0, 1]), state2=np.array([1, 0, 1, 0]))
        dump(directory/"run.json", {"outputs": {"temporal.npz": {"sha256": sha256(path)}}})
        if year == 2000:
            dump(directory/"feature_registry.json", feature_registry)
    main(["--feature-dir", str(features), "--registry", str(registry), "--output-dir", str(output),
          "--scopes", "temporal"])
    selection = json.loads((output/"lr_selection_temporal.json").read_text())
    assert len(selection["candidates"]) == 4
    state = json.loads((output/"models/temporal/lag2_logistic_F0.json").read_text())
    assert state["C"] == selection["selected_C"]
    assert state["scaler_mean"] == [1.5]*33
    assert state["training_rows"] == 60
    metrics = pd.read_csv(output/"metrics_yearly.csv")
    assert set(metrics.year) == set(range(2000, 2018))
    assert metrics.loc[metrics.risk_set == "onset", "n"].eq(2).all()
    assert metrics.loc[metrics.risk_set == "onset", "events"].eq(1).all()
