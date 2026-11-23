<!-- README_template.md — the front door to your reproducible foundation.
     The test: could a competent stranger, on a clean machine, go from
     `git clone` to your headline number following ONLY this file? -->

# <Project Title> — <one-line description>

<One paragraph: what question this repository answers, in your chosen CS domain,
and what modern AI method it investigates.>

## What this produces

This repository reproduces **<the headline result>** — e.g., "Table 2 and Figure 3
of the paper": <metric> = <value> on <dataset>.

## Environment

```bash
# Pick the layer this project uses (see chapter §9.3):
conda env create -f environment.yml && conda activate <env-name>
# or:
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Pinned with: <pip freeze | conda-lock | Docker>. Hardware used: <CPU/GPU model, driver>.
(A seed makes runs repeatable on this hardware; results are not guaranteed bit-identical
across different GPUs or library releases.)

## Data

Download the dataset from <DOI / URL> (accessed <date>) into `data/raw/`.
See [`data/README.md`](data/README.md) for full provenance and the split rule.

```
Expected: data/raw/dataset.csv
SHA-256:  a1b2c3...   # verify you have the same file we used
```

## Reproduce

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # only if using CUDA determinism
python src/run_experiment.py --config src/config.yaml --seed 1337
# writes results/metrics.json and results/figure3.png
```

## Expected output

```json
{ "accuracy": 0.873, "f1": 0.861 }
```

A single run is repeatable; the claim is reported as mean ± spread across seeds 1–10
(see `results/seed_*.json`). One run is not a result.

## Cite / License

Licensed under <MIT / Apache-2.0>. If you use this work, please cite <preprint / Zenodo DOI>.

## AI assistance

AI tools used in building this repository are noted in [`docs/agent-log.md`],
disclosed per the target venue's policy.
