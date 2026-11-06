"""Record byte hashes for phase-2 artifacts; never copy large arrays or raw data."""

import argparse
from pathlib import Path

from tools.probability_phase2_features import dump, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/probability_phase2"))
    args = parser.parse_args(argv)
    destination = args.phase2_dir/"artifact_manifest.json"
    paths = [path for path in args.phase2_dir.rglob("*") if path.is_file() and path != destination
             and ("2000-r1-incomplete" not in path.parts or path.name == "failure.json")]
    for directory, pattern in (("src", "probability_*.py"), ("tools", "probability_*.py"),
                               ("tests", "test_probability_*.py"), ("docs", "PROBABILITY_PHASE2*.md"),
                               ("figures/probability_phase2", "*.png")):
        paths.extend(Path(directory).glob(pattern))
    paths.append(Path("configs/probability_protocol_v1.json"))
    entries = {path.as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path),
                                "git_expected": path.suffix not in (".nc", ".npz", ".npy")}
               for path in sorted(set(paths))}
    dump(destination, {"schema_version": 1, "archival_policy": "Git and bundle contain small evidence only; regenerate local arrays from read-only raw inputs",
                       "large_arrays_copied": False, "incomplete_2000_arrays": "retained locally, excluded from official artifact list; failure.json retained",
                       "files": entries, "file_count": len(entries),
                       "total_bytes": sum(item["bytes"] for item in entries.values()),
                       "git_expected_bytes": sum(item["bytes"] for item in entries.values() if item["git_expected"])})
    print(f"Recorded {len(entries)} artifacts in {destination}")


if __name__ == "__main__":
    main()
