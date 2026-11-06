"""Plot the completed temporal F0-F4 validation ladder, without refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=Path("data/probability_phase2/experiments/run1/metrics_yearly.csv"))
    parser.add_argument("--lr", type=Path, default=Path("data/probability_phase3/lr_ablations/run1/metrics_yearly.csv"))
    parser.add_argument("--tree", type=Path, default=Path("data/probability_phase3/lightgbm/run1/metrics_yearly.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/probability_phase3"))
    args = parser.parse_args(argv)
    inputs = [args.phase2, args.lr, args.tree]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    parts = []
    for path, prefix, family in [(args.phase2, "logistic_", "LR"),
                                 (args.lr, "logistic_", "LR"),
                                 (args.tree, "lightgbm_", "LightGBM")]:
        frame = pd.read_csv(path)
        frame = frame.loc[frame.scope.eq("temporal") & frame.split.eq("validation")
                          & frame.risk_set.isin(["all", "onset"])].copy()
        frame["arm"] = frame.model.str.removeprefix(prefix)
        frame = frame.loc[frame.arm.isin(["F0", "F1", "F2", "F3", "F4"])]
        frame["family"] = family
        parts.append(frame[["family", "arm", "lag", "year", "risk_set", "n", "events", "brier"]])
    values = pd.concat(parts, ignore_index=True)
    if len(values) != 120 or values.duplicated(["family", "arm", "lag", "year", "risk_set"]).any():
        raise ValueError("Expected the full five-arm/two-family/two-lag/three-year/two-risk ladder")
    if set(values.year) != {2015, 2016, 2017} or set(values.lag) != {1, 2}:
        raise ValueError("Unexpected years or lags")
    for _, cohort in values.groupby(["lag", "year", "risk_set"]):
        if cohort.n.nunique() != 1 or cohort.events.nunique() != 1:
            raise ValueError("Ladder comparisons do not share sample/outcome counts")
    if input_hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}:
        raise RuntimeError("Input changed during read; rerun after the current output write")
    means = values.groupby(["family", "arm", "lag", "risk_set"], as_index=False).agg(
        mean_annual_brier=("brier", "mean"), years=("year", "count"))
    if len(means) != 40 or not means.years.eq(3).all():
        raise ValueError("Incomplete annual means")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "feature_ladder_validation"
    values.to_csv(stem.with_suffix(".csv"), index=False, lineterminator="\n")
    means.to_csv(args.out / "feature_ladder_validation_summary.csv", index=False, lineterminator="\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "probability-phase3-ladder-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True, sharey="row")
    arms = ["F0", "F1", "F2", "F3", "F4"]
    style = {"LR": ("#4763A4", "o"), "LightGBM": ("#B34D2E", "s")}
    for row, risk in enumerate(["all", "onset"]):
        for column, lag in enumerate([1, 2]):
            ax = axes[row, column]
            for family, (color, marker) in style.items():
                part = means.loc[means.family.eq(family) & means.lag.eq(lag)
                                 & means.risk_set.eq(risk)].set_index("arm").loc[arms]
                ax.plot(np.arange(5), part.mean_annual_brier, color=color, marker=marker,
                        linewidth=2, markersize=5, label=family)
            ax.set_title(f"D-{lag} | {'All samples' if risk == 'all' else 'Onset risk set'}", pad=10)
            ax.set_xticks(np.arange(5), ["F0\nC + T", "F1\n+ M", "F2\n+ Night", "F3\n+ Region", "F4\n+ History"])
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(axis="y", color="#E2E5E7", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.25, 4.25)
            if column == 0:
                ax.set_ylabel("Mean annual Brier (lower is better)")
    fig.suptitle("More features do not guarantee a better validation score", fontsize=16, y=0.98)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=2, frameon=False)
    fig.text(0.5, 0.06,
             "Temporal scope, 2015-2017 validation; equal annual weights; raw probabilities.\n"
             "C: calendar/location; T: temperature; M: moisture. Night, region and history are added in order.\n"
             "Onset includes 00 and 01 outcomes. These years were used for model selection; lines connect discrete feature sets.",
             ha="center", va="center", fontsize=9, color="#444444", linespacing=1.5)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.84, bottom=0.20, hspace=0.42, wspace=0.16)
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)
    metadata = {"inputs_sha256": input_hashes, "annual_records": len(values), "mean_records": len(means),
                "years": [2015, 2016, 2017], "scope": "temporal", "probabilities": "uncalibrated",
                "aggregation": "equal-weight mean annual Brier", "test_or_calibration_years_read": False,
                "interpretation": "Controlled feature ladder under each family's fixed selected settings, not causal effects",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {stem}.*; {len(values)} annual scores and {len(means)} means")


if __name__ == "__main__":
    main()
