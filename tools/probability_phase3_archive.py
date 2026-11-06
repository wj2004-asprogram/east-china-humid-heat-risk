"""Hash completed phase-3 evidence once; never copy arrays or original data."""

import argparse
from pathlib import Path

from tools.probability_phase2_features import dump, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, default=Path("data/probability_phase3"))
    args = parser.parse_args(argv)
    destination = args.phase3_dir/"artifact_manifest.json"
    paths = [path for path in args.phase3_dir.rglob("*") if path.is_file() and path != destination]
    for directory, pattern in (("src", "probability_*.py"), ("tools", "probability_*.py"),
                               ("tests", "test_probability_*.py"), ("docs", "PROBABILITY_PHASE3*.md"),
                               ("figures/probability_phase3", "*")):
        paths.extend(Path(directory).glob(pattern))
    paths.extend(Path(path) for path in ("configs/probability_phase3_run.json",
                                       "configs/probability_protocol_v1.json", ".gitignore", ".gitattributes"))
    entries = {path.as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path),
                               "git_expected": path.suffix not in (".nc", ".npz", ".npy")}
               for path in sorted(set(paths)) if path.is_file()}
    dump(destination, {"schema_version": 1,
                       "archival_policy": "Git and bundle retain small evidence; local arrays are hashed without copying",
                       "large_arrays_copied": False,
                       "phase2_feature_inputs": "Reused unchanged; hashes bound by run execution locks",
                       "execution_may_be_partial": "Consult each execution registry for complete, failed, pending items",
                       "files": entries, "file_count": len(entries),
                       "total_bytes": sum(item["bytes"] for item in entries.values()),
                       "git_expected_bytes": sum(item["bytes"] for item in entries.values() if item["git_expected"])})
    print(f"Recorded {len(entries)} artifacts in {destination}")


if __name__ == "__main__":
    main()
