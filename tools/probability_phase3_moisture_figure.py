"""Plot completed, predeclared moisture contrasts using validation metrics only."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(path, family):
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["scope"].eq("temporal")
        & frame["split"].eq("validation")
        & frame["risk_set"].isin(["all", "onset"])
    ].copy()
    frame["arm"] = frame["model"].str.removeprefix(family + "_")
    return frame.loc[frame["arm"].isin(["F0", "F1", "F0P", "F1P"])]


def paired_deltas(frame, label):
    keys = ["lag", "year", "risk_set"]
    rows = []
    for candidate, reference in [("F1", "F0"), ("F1P", "F0P")]:
        left = frame.loc[frame.arm.eq(candidate)]
        right = frame.loc[frame.arm.eq(reference)]
        paired = left.merge(right, on=keys, suffixes=("_candidate", "_reference"),
                            validate="one_to_one", how="outer", indicator=True)
        if len(paired) != 12 or not paired["_merge"].eq("both").all():
            raise ValueError(f"Incomplete two-lag validation contrast: {label} {candidate}-{reference}")
        if set(paired.year) != {2015, 2016, 2017} or set(paired.lag) != {1, 2}:
            raise ValueError("Unexpected validation years or lags")
        if not ((paired.n_candidate == paired.n_reference).all()
                and (paired.events_candidate == paired.events_reference).all()):
            raise ValueError("Contrast sample/outcome counts differ")
        for row in paired.itertuples(index=False):
            rows.append({"family": label, "candidate": candidate, "reference": reference,
                         "lag": row.lag, "year": row.year, "risk_set": row.risk_set,
                         "n": row.n_candidate, "events": row.events_candidate,
                         "candidate_brier": row.brier_candidate, "reference_brier": row.brier_reference,
                         "delta_brier": row.brier_candidate - row.brier_reference})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=Path("data/probability_phase2/experiments/run1/metrics_yearly.csv"))
    parser.add_argument("--lr", type=Path, default=Path("data/probability_phase3/lr_ablations/run1/metrics_yearly.csv"))
    parser.add_argument("--tree", type=Path, default=Path("data/probability_phase3/lightgbm/run1/metrics_yearly.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/probability_phase3"))
    args = parser.parse_args(argv)
    inputs = [args.phase2, args.lr, args.tree]
    # Hash the exact inputs read. The final archive can rerun this after its batch closes.
    hashes_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    lr = pd.concat([load_metrics(args.phase2, "logistic"), load_metrics(args.lr, "logistic")])
    tree = load_metrics(args.tree, "lightgbm")
    key = ["arm", "lag", "year", "risk_set"]
    if lr.duplicated(key).any() or tree.duplicated(key).any():
        raise ValueError("Duplicate input model metrics")
    common = lr.merge(tree, on=key, suffixes=("_lr", "_tree"), validate="one_to_one", how="outer", indicator=True)
    if len(common) != 48 or not common["_merge"].eq("both").all():
        raise ValueError("Expected all four arms, two lags, three years and two risk sets in both families")
    if not ((common.n_lr == common.n_tree).all() and (common.events_lr == common.events_tree).all()):
        raise ValueError("Family sample/outcome counts differ")
    points = pd.concat([paired_deltas(lr, "LR"), paired_deltas(tree, "LightGBM")], ignore_index=True)
    hashes_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if hashes_before != hashes_after:
        raise RuntimeError("Input changed during read; rerun after the current output write")
    group = ["family", "candidate", "reference", "lag", "risk_set"]
    means = points.groupby(group, as_index=False).agg(
        mean_delta=("delta_brier", "mean"), years=("year", "count"),
        years_improving=("delta_brier", lambda values: int((values < 0).sum())))
    if len(points) != 48 or len(means) != 16 or not means.years.eq(3).all():
        raise ValueError("Incomplete plot records")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "moisture_validation_contrasts"
    points.to_csv(stem.with_suffix(".csv"), index=False, lineterminator="\n")
    means.to_csv(args.out / "moisture_validation_contrasts_summary.csv", index=False, lineterminator="\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "probability-phase3-moisture-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True, sharey=True)
    row_specs = [("LR", "all"), ("LightGBM", "all"), ("LR", "onset"), ("LightGBM", "onset")]
    colors = {"all": "#17659C", "onset": "#A84A13"}
    offsets = {2015: -0.1, 2016: 0, 2017: 0.1}
    markers = {2015: "o", 2016: "s", 2017: "^"}
    titles = {"F1": "Moisture increment: F1 - F0", "F1P": "Conditional moisture: F1P - F0P"}
    lo, hi = points.delta_brier.min(), points.delta_brier.max()
    for row_idx, candidate in enumerate(["F1", "F1P"]):
        for col_idx, lag in enumerate([1, 2]):
            ax = axes[row_idx, col_idx]
            ax.axvline(0, color="#606060", linewidth=0.8)
            ax.axhspan(1.5, 3.5, color="#FAF2EC", zorder=0)
            for y, (family, risk) in enumerate(row_specs):
                subset = points.loc[points.family.eq(family) & points.candidate.eq(candidate)
                                    & points.lag.eq(lag) & points.risk_set.eq(risk)]
                summary = means.loc[means.family.eq(family) & means.candidate.eq(candidate)
                                    & means.lag.eq(lag) & means.risk_set.eq(risk)].iloc[0]
                for point in subset.itertuples():
                    ax.scatter(point.delta_brier, y + offsets[point.year], marker=markers[point.year],
                               s=25, color=colors[risk], alpha=0.6, edgecolors="none", zorder=3)
                ax.scatter(summary.mean_delta, y, marker="D", s=64, facecolors="white",
                           edgecolors=colors[risk], linewidth=1.6, zorder=4)
                ax.annotate(f"{summary.mean_delta:+.4f}", (summary.mean_delta, y),
                            xytext=(0, 13), textcoords="offset points", ha="center",
                            fontsize=9, color=colors[risk])
            ax.set_title(f"{titles[candidate]}  |  D-{lag}", fontsize=11, pad=12)
            ax.set_yticks(range(4), ["LR / all", "LightGBM / all", "LR / onset", "LightGBM / onset"])
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylim(3.5, -0.65)
            ax.set_xlim(lo - 0.002, hi + 0.0025)
            ax.grid(axis="x", color="#E2E5E7", linewidth=0.6)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if row_idx == 1:
                ax.set_xlabel("Brier difference (with moisture - reference); lower is better")
    for year, marker in markers.items():
        axes[0, 0].scatter([], [], marker=marker, s=25, color="#777777", label=str(year))
    axes[0, 0].scatter([], [], marker="D", s=55, facecolors="white", edgecolors="#444444", label="Annual mean")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=4, frameon=False)
    fig.suptitle("Moisture helps overall scores; D-2 onset results differ", fontsize=16, y=0.98)
    fig.text(0.5, 0.055, "2015-2017 validation, temporal scope; raw probabilities; same samples per contrast.\n"
             "Onset risk set: previous-day non-event (00 and 01), evaluated retrospectively.\n"
             "P = latest allowed event state. Validation was used for model selection; no final-test or causal claim.",
             ha="center", va="center", fontsize=9, color="#444444", linespacing=1.5)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.18, hspace=0.42, wspace=0.38)
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)
    metadata = {"inputs_sha256": hashes_after, "point_records": len(points), "summary_records": len(means),
                "years": [2015, 2016, 2017], "scope": "temporal", "probabilities": "uncalibrated",
                "aggregation": "equal-weight mean of annual candidate-minus-reference Brier differences",
                "year_markers_are_not_uncertainty_intervals": True,
                "test_or_calibration_years_read": False,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {stem}.*; {len(points)} year points and {len(means)} annual means")


if __name__ == "__main__":
    main()
