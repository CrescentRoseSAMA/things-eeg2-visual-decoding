"""Run the data-only experiment with a durable log and one local GPU worker."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "pipeline.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "pipeline.log").open("a", encoding="utf-8", buffering=1) as log:
            for stage in ("smoke", "verify-smoke", "develop", "formal", "evaluate"):
                print(datetime.now().astimezone().isoformat(), stage, "started", file=log, flush=True)
                result = subprocess.run(
                    [sys.executable, "-B", "-u", "eeg_train.py", stage],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                print(datetime.now().astimezone().isoformat(), stage,
                      "exit", result.returncode, file=log, flush=True)
                if result.returncode:
                    return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
