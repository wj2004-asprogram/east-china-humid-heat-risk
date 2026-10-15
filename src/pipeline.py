"""
pipeline.py

Run the humid heat risk workflow for one configuration.

Usage (from the project root):
    python -m src.pipeline --config configs/july_2022.yaml
    python -m src.pipeline --config configs/july_2022.yaml --download
    python -m src.pipeline --config configs/july_2022.yaml --steps process_humid_heat classify_heat_risk
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # write figures to files only; no GUI windows

from src import (  # noqa: E402  (must come after selecting the backend)
    classify_heat_risk,
    download_era5,
    plot_city_tmax,
    plot_figures,
    plot_heat_index_figures,
    plot_heat_risk_maps,
    process_humid_heat,
    process_temperature,
)
from src.config import load_config  # noqa: E402


# Processing steps in execution order. Each step reads the outputs of
# earlier steps from disk, so any suffix of this list can be re-run alone.
STEPS = {
    "process_temperature": process_temperature.run,
    "plot_figures": plot_figures.run,
    "plot_city_tmax": plot_city_tmax.run,
    "process_humid_heat": process_humid_heat.run,
    "plot_heat_index_figures": plot_heat_index_figures.run,
    "classify_heat_risk": classify_heat_risk.run,
    "plot_heat_risk_maps": plot_heat_risk_maps.run,
}

# Steps that read the raw ERA5 file (the others read processed files).
RAW_INPUT_STEPS = {"process_temperature", "process_humid_heat"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="East China humid heat risk workflow")
    parser.add_argument("--config", required=True, help="Path to a YAML file in configs/")
    parser.add_argument(
        "--download", action="store_true",
        help="Download the raw ERA5 file first (skipped if it already exists)",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=list(STEPS), metavar="STEP",
        help=f"Run only these steps (in pipeline order). Choices: {', '.join(STEPS)}",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    print(f"Configuration: {cfg.name} ({cfg.period_label})")

    if args.download:
        print("\n=== download ===")
        download_era5.run(cfg)

    selected = args.steps or list(STEPS)
    steps = [name for name in STEPS if name in selected]  # keep pipeline order

    if RAW_INPUT_STEPS.intersection(steps) and not cfg.paths.raw_era5.exists():
        raise SystemExit(
            f"Raw ERA5 file not found: {cfg.paths.raw_era5}\n"
            "Run again with --download, or copy the file there."
        )

    for i, name in enumerate(steps, start=1):
        print(f"\n=== Step {i}/{len(steps)}: {name} ===")
        STEPS[name](cfg)

    print(f"\nWorkflow '{cfg.name}' completed.")


if __name__ == "__main__":
    main()
