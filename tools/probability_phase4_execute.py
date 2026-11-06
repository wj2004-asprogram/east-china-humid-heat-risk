"""Record a single authorized phase-four command, log, runtime and peak memory."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys
import time

from src.probability_phase4_data import sha256
from tools.probability_phase2_features import peak_memory_bytes


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if Path(a.name).name != a.name or not a.module.startswith("tools.probability_phase4_"):
        raise ValueError("Require local phase-four module and simple invocation name")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("data/probability_phase4/invocations") / f"{stamp}-{a.name}"
    output.mkdir(parents=True, exist_ok=False)
    arguments = a.args[1:] if a.args[:1] == ["--"] else a.args
    receipt = {"started_utc": stamp, "module": a.module, "arguments": arguments,
               "python_executable": sys.executable, "working_directory": str(Path.cwd()),
               "source_sha256": sha256(a.module.replace(".", "/") + ".py"), "status": "running"}
    target = output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    started = time.perf_counter()
    old_argv = sys.argv
    try:
        sys.argv = [a.module, *arguments]
        with (output / "output.log").open("x", encoding="utf-8", newline="\n") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                runpy.run_module(a.module, run_name="__main__")
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        raise
    finally:
        sys.argv = old_argv
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                       runtime_seconds=time.perf_counter()-started,
                       peak_working_set_bytes=peak_memory_bytes(),
                       output_log_sha256=sha256(output / "output.log"))
        target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"invocation_receipt": str(target), **receipt}), flush=True)


if __name__ == "__main__":
    main()
