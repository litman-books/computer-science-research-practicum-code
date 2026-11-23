"""provenance.py — capture full provenance for every experimental run.

Import this into your experiment harness and create a `Run(...)` at the top
of each run, BEFORE any randomness or compute. It writes one append-only row
to your run log so every number is reproducible from the record.

Rule of the week (Ch11): a result you cannot tie to a commit, seed, config,
and dataset is a rumor, not a measurement. Commit first, then run.
"""
from __future__ import annotations
import csv, json, hashlib, subprocess, datetime, os
from dataclasses import dataclass
try:
    import fcntl                        # POSIX file lock (macOS/Linux)
except ImportError:                     # Windows: no lock; run seeds one at a time
    fcntl = None

LOG_PATH = "results/run_log.csv"
# The run log lives inside the repo but is NOT code: it grows on every run by
# design, so the dirty-tree check below ignores this directory. If you move
# LOG_PATH, the exclusion follows it. (A log with no directory excludes just itself.)
RESULTS_DIR = os.path.dirname(LOG_PATH) or LOG_PATH
FIELDS = [
    "run_id", "timestamp_utc", "commit", "dirty", "seed", "condition",
    "config_json", "dataset_sha256_16", "split", "model_version",
    "model_provider", "decoding", "metric_name", "metric_value",
    "status", "exclusion_reason", "notes",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args]).decode().strip()


def _is_dirty() -> bool:
    """True if any code/config in the repo has uncommitted or untracked changes.

    Same as `git status --porcelain`, but over the whole repo (":/") and with the
    results directory excluded, so your own run log never makes a run "dirty".
    Python's own __pycache__/ bytecode is excluded too; your Chapter 9 .gitignore
    should list it anyway.
    """
    return bool(_git("status", "--porcelain", "--", ":/", f":(exclude){RESULTS_DIR}",
                     ":(exclude,glob)**/__pycache__/**"))


def _hash_file(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass
class Run:
    seed: int
    condition: str
    config: dict
    dataset_path: str
    split: str = "test"
    model_version: str = "n/a"      # exact string, e.g. "claude-opus-5-5" — NEVER just the family
    model_provider: str = "n/a"
    decoding: str = "n/a"           # every decoding knob you set, e.g. "effort=medium"; for an open-weight model, its full sampling config
    run_id: str = ""                # leave blank for a fresh id; pass the old one to RESUME a run

    def __post_init__(self) -> None:
        # Unique even when seeds run in parallel: microseconds + seed + process id.
        if not self.run_id:
            stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
            self.run_id = f"{stamp}-s{self.seed}-p{os.getpid()}"

    def row(self, metric_name: str, metric_value, status: str = "ok",
            exclusion_reason: str = "", notes: str = "") -> dict:
        dirty = _is_dirty()
        if dirty:
            print("WARNING: dirty working tree — this result is NOT reproducible. Commit first.")
        return {
            "run_id": self.run_id,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": str(dirty).lower(),
            "seed": self.seed,
            "condition": self.condition,
            "config_json": json.dumps(self.config, sort_keys=True),
            "dataset_sha256_16": _hash_file(self.dataset_path),
            "split": self.split,
            "model_version": self.model_version,
            "model_provider": self.model_provider,
            "decoding": self.decoding,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "status": status,
            "exclusion_reason": exclusion_reason,
            "notes": notes,
        }


def log(row: dict, path: str = LOG_PATH) -> None:
    """Append-only: never overwrite a result. A rerun is a NEW row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        if fcntl:                       # parallel seeds: one writer at a time
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:               # empty file -> exactly one header row
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    r = Run(seed=3, condition="method", config={"lr": 1e-3, "epochs": 20},
            dataset_path="data/dataset.jsonl")
    log(r.row("macro_f1", 0.761, notes="KEPT: real variance"))
    print(f"logged run {r.run_id} -> {LOG_PATH}")
