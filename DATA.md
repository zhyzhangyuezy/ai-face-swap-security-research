# Data, Weights, and Artifact Policy

This repository is intended to host code and reproducibility scaffolding. It
intentionally does not commit manuscript LaTeX/PDF files, raw benchmark media,
downloaded third-party repositories, model weights, dense per-row score dumps,
or extracted feature tensors.

## Not Stored in Git

- `workspace/datasets/`: raw and derived benchmark media.
- `workspace/checkpoints/`, `workspace/downloads/weights/`, and external
  checkpoint folders: model weights.
- `workspace/features/`: extracted passive/provenance features.
- `workspace/repos/`: local copies of third-party baselines such as
  DeepfakeBench, FTCN, AltFreezing, M2F2-Det, and VideoSeal.
- `pilot/passive_baseline/runs/`: local training checkpoints and run artifacts.
- Large `prototype/manifests/*.csv` and dense `prototype/reports/*.csv` files:
  these can exceed normal GitHub-friendly sizes and may contain local path
  identifiers.
- `paper/`: manuscript source and compiled submission PDFs. Keep these in the
  journal submission package or a private branch if needed.

## Recommended Release Layout

Use GitHub for source code and lightweight summaries. Use GitHub Releases,
Zenodo, institutional storage, or a private artifact bucket for large auxiliary
files when licensing permits redistribution.
The repository-level `ARTIFACT_MANIFEST.md` and `ARTIFACT_CHECKSUMS.sha256`
provide the paper-facing file index and integrity hashes for locally available
listed artifacts.

```text
repo/
  prototype/scripts/       # experiment and evaluation entry points
  prototype/src/           # reusable local helpers
  prototype/configs/       # public configs and path template
  prototype/reports/       # report directory documentation
  DATA.md                  # this file
  DATASETS.md              # dataset and baseline acquisition notes
```

Large artifact bundle, stored outside Git:

```text
artifact-root/
  manifests/               # full paper-facing CSV manifests
  reports/                 # dense per-row CSV score tables
  features/                # optional precomputed feature tensors
  checkpoints/             # model weights allowed by their licenses
```

## Rebuilding Local Paths

Copy `prototype/configs/paths.example.yaml` to
`prototype/configs/paths.local.yaml` and fill in local paths for datasets,
features, third-party repositories, and outputs. `paths.local.yaml` is ignored by
Git because it is machine-specific.

See `DATASETS.md` for dataset-specific acquisition and placement notes.

## Benchmark and Biometric-Media Restrictions

The paper uses face/video datasets and third-party detector weights whose
licenses and consent terms may restrict redistribution. Public releases should
therefore expose scripts, manifests, hashes, result summaries, and score files
where allowed, while requiring users to obtain restricted media from the
original providers.

## Git LFS Guidance

If you decide to publish selected large artifacts in the same GitHub repository,
use Git LFS for files above roughly 50 MB:

```powershell
git lfs install
git lfs track "*.pt" "*.pth" "*.ckpt" "*.onnx" "*.zip" "*.mp4" "*.csv"
```

For ordinary code review and paper reproduction, prefer keeping those artifacts
outside the repository and documenting their locations/checksums instead.
