"""set_seeds.py — one utility to make a single run repeatable.

Reproducibility caveats (as of 2026):
  * A seed makes YOUR runs repeatable on YOUR machine. It does NOT make a CUDA
    result bit-identical on someone else's GPU, across PyTorch releases, or
    between CPU and GPU. Document the hardware, not just the seed.
  * torch.use_deterministic_algorithms(True) raises a RuntimeError with CUDA
    >= 10.2 unless CUBLAS_WORKSPACE_CONFIG is set BEFORE the process starts
    (set it in your run script: export CUBLAS_WORKSPACE_CONFIG=:4096:8).
  * PYTHONHASHSEED (the seed for str/bytes hashing, which drives set
    iteration order) is read ONCE, when the interpreter starts. Setting it
    from inside Python cannot change the running process; it only reaches
    child processes. Export it before Python starts:
        export PYTHONHASHSEED=1337
    (the Dockerfile does this with ENV). set_all_seeds() warns if you forgot.
  * Repeatability (same seed -> same answer) is necessary but NOT sufficient
    for a result. A single run is not a claim; vary the seed for the claim
    (Bouthillier et al. 2021, "Accounting for Variance in ML Benchmarks").
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

SEED = 1337


def set_all_seeds(seed: int = SEED) -> None:
    """Seed every randomness source this project can touch."""
    # Hash seeding is fixed at interpreter start-up, so this line does NOT
    # affect the current process; it only passes the value to subprocesses
    # (e.g. DataLoader workers started with "spawn"). Check the real value:
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        print(f"WARNING: PYTHONHASHSEED is not {seed} for this process, so string "
              f"hashing (set order) is NOT repeatable. Before starting Python, run: "
              f"export PYTHONHASHSEED={seed}", file=sys.stderr)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional — only seed it if the project uses it.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Demand deterministic kernels; requires CUBLAS_WORKSPACE_CONFIG (see above).
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=...) so workers are seeded too."""
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    set_all_seeds()
    # Proof of repeatability: this prints the same numbers on every run.
    print("random:", random.random())
    print("numpy :", np.random.rand(3).round(6).tolist())
