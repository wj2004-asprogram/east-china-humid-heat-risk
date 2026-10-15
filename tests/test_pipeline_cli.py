"""Tests for the pipeline entry point and the download step (no network)."""

import sys
import types

import pytest

from src import download_era5, pipeline
from src.config import PROJECT_ROOT, load_config


CONFIG = PROJECT_ROOT / "configs" / "july_2022.yaml"


def test_raw_file_is_required_only_by_raw_input_steps(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text(
        CONFIG.read_text(encoding="utf-8").replace('file_tag: "2022_07"', 'file_tag: "no_such_run"'),
        encoding="utf-8",
    )

    # A step that reads the raw file stops early with a clear message ...
    with pytest.raises(SystemExit, match="Raw ERA5 file not found"):
        pipeline.main(["--config", str(missing), "--steps", "process_temperature"])

    # ... while a later step only needs its own (processed) input.
    with pytest.raises(FileNotFoundError, match="daily_tmax_no_such_run"):
        pipeline.main(["--config", str(missing), "--steps", "plot_figures"])


def _fake_cdsapi(monkeypatch, fail: bool):
    """Install a fake `cdsapi` module whose retrieve() writes the target file."""

    class Client:
        def retrieve(self, dataset, request, target):
            with open(target, "wb") as f:
                f.write(b"partial" if fail else b"complete")
            if fail:
                raise ConnectionError("download interrupted")

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=Client))


def test_interrupted_download_leaves_no_file_at_final_path(tmp_path, monkeypatch):
    cfg = load_config(CONFIG, project_root=tmp_path)

    _fake_cdsapi(monkeypatch, fail=True)
    with pytest.raises(ConnectionError):
        download_era5.run(cfg)
    assert not cfg.paths.raw_era5.exists()

    # A retry is not skipped as "already exists" and completes the file.
    _fake_cdsapi(monkeypatch, fail=False)
    download_era5.run(cfg)
    assert cfg.paths.raw_era5.read_bytes() == b"complete"
