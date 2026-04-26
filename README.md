# Reliability-Aware Video Face-Swap Detection

This repository contains the code and reproducibility scaffolding for a
protected-real-FPR-controlled video face-swap detection study.

The central question is whether a detector can still catch protected and
unprotected face-swap fakes while avoiding false accusations on real media that
has passed through a provenance/protection pipeline.

## Repository Layout

- `prototype/src/`: reusable local helpers for path/config handling and manifest
  schema validation.
- `prototype/scripts/`: manifest builders, calibration scripts, comparator
  probes, leakage audits, and paper-facing result assembly.
- `prototype/configs/`: public experiment configs and `paths.example.yaml`.
- `prototype/reports/`: report documentation. Full report artifacts are kept
  outside Git by default.
- `prototype/manifests/`: manifest documentation; full CSV manifests are kept
  outside Git by default.
- `pilot/`: exploratory training scripts. Large run folders are ignored.
- `docs/`: release and dataset-preparation notes.

Large datasets, checkpoints, external baseline repositories, generated features,
paper LaTeX/PDF files, and dense score dumps are intentionally excluded from
Git. See `DATA.md` and `DATASETS.md`.

## Quick Start

Create a local path file:

```powershell
Copy-Item prototype/configs/paths.example.yaml prototype/configs/paths.local.yaml
```

Edit `prototype/configs/paths.local.yaml` so `project_root`, `asset_root`,
dataset roots, external repos, and output paths match your machine.

Install the core analysis and figure dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run a basic data/path audit:

```powershell
python prototype/scripts/audit_topconf_data.py --paths prototype/configs/paths.local.yaml
```

## Artifact Policy

This public code package does not include the manuscript LaTeX, raw benchmark
media, generated protected media, extracted features, model weights, or dense
per-row score dumps. Those belong in a submission package or a separate artifact
release when licensing allows.

## GitHub Upload Notes

Before pushing, read:

- `DATA.md`
- `DATASETS.md`
- `docs/GITHUB_RELEASE_CHECKLIST.md`

The `.gitignore` is configured to avoid committing local environments, raw
datasets, downloaded weights, third-party baseline repos, dense score dumps, and
LaTeX scratch files.

## License

Choose a code license before making the repository public. Dataset and model
weights remain governed by their original providers' licenses.
