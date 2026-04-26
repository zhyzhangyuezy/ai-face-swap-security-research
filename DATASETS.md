# Dataset and Baseline Preparation

This repository does not redistribute face/video benchmark media, protected
derivatives, downloaded model weights, or external baseline repositories. Users
must obtain restricted assets from the original providers and place them in a
local `workspace/` tree or another path configured in
`prototype/configs/paths.local.yaml`.

The datasets used by the experiments contain biometric media and often have
non-redistribution or request-based access terms. Even when a dataset is public,
do not upload raw videos, extracted frames, face crops, or protected-media
derivatives to this repository unless the original license explicitly permits
redistribution.

## Local Directory Template

```text
workspace/
  datasets/
    raw/
      faceforensicspp/
      celebdf/
      dfdcp/
      uadfv/
      vcf/
      deeperforensics/
      df40/
    derived/
      protected/
  checkpoints/
    deepfakebench/
    effort/
    ftcn/
    altfreezing/
    m2f2/
    videoseal/
  features/
    passive/
    provenance/
  repos/
    DeepfakeBench/
    FTCN/
    AltFreezing/
    M2F2_Det/
    videoseal/
```

Configure the exact local paths in:

```text
prototype/configs/paths.local.yaml
```

Create it from:

```text
prototype/configs/paths.example.yaml
```

## Benchmark Sources and Citations

Use the official source or author-maintained project page whenever possible.
Third-party mirrors are convenient for local experiments, but they should not be
cited as the dataset origin.

| Dataset | Used for | Official/source location | Paper citation key |
| --- | --- | --- | --- |
| FaceForensics++ / FF++ | Canonical real/fake frames, paired generator rows, FF++-based hard-generator slices | https://github.com/ondyari/FaceForensics. Access is request-based through the project form. | `rossler2019faceforensics` |
| Celeb-DF v1/v2 | Cross-dataset face-forgery evaluation and held-out Celeb-DF rows | https://github.com/yuezunli/celeb-deepfakeforensics. Access is request-based through the project forms. | `li2020celebdf` |
| DFDC / DFDCP | DFDC-style and preview-style evaluation slices where locally available | https://ai.meta.com/datasets/dfdc/ and https://www.kaggle.com/competitions/deepfake-detection-challenge/data. | `dolhansky2020dfdc`, `dolhansky2019dfdcpreview` |
| UADFV | Early small-scale face-forgery benchmark used in pilot checks | Original source paper: https://arxiv.org/abs/1811.00661. The stable public download channel is less standardized than FF++/Celeb-DF; record the exact local source in `paths.local.yaml` or an internal run note. | `yang2019inconsistentheadposes` |
| VCF | Video-conference face-swap benchmark and temporal stress tests | Article/source page: https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/169/2025/. Follow the article/provider instructions for dataset access. | `vcf` |
| DeeperForensics-1.0 | Distortion and transfer stress tests | https://github.com/EndlessSora/DeeperForensics-1.0. The dataset is released for non-commercial research under its terms of use. | `jiang2020deeperforensics` |
| DF40-style generator slices | Next-generation generator and paired SimSwap/InSwap-style diagnostics where locally available | https://github.com/YZY-stack/DF40. Follow the repository's dataset and CC BY-NC 4.0 licensing instructions. | `df40` |

The paper bibliography in `paper/refs/references.bib` contains these citation
keys. If a new dataset slice is added, update both this table and the paper
bibliography before reporting results.

## Baseline Repositories and Weights

The experiments may call locally cloned third-party baselines, including:

- DeepfakeBench checkpoints and detector implementations.
- Effort official face checkpoint or locally reproduced adapter.
- FTCN / FTCN+TT style video detector probes.
- AltFreezing native video diagnostic probes.
- M2F2-Det detector-only diagnostic probes.
- VideoSeal/provenance tooling for protected-media experiments.

These external repositories and weights are not vendored here. Put them under
`workspace/repos/` and `workspace/checkpoints/`, or point `paths.local.yaml` to
their actual locations.

## Provided Download Helpers

Some scripts help prepare public or user-authorized assets:

- `prototype/scripts/download_google_drive_file.py`
- `prototype/scripts/download_deepfakebench_rgb.py`
- `prototype/scripts/download_altfreezing_recdrive_weight.py`
- `prototype/scripts/build_deepfakebench_json_manifest.py`
- `prototype/scripts/build_video_dir_smoke_manifest.py`

These helpers do not bypass dataset/model licenses. They are convenience
wrappers for assets that the user is already authorized to access.

## What Can Be Released Separately

When licenses permit, a separate artifact release may include:

- paper-facing manifests with relative paths or hashed IDs;
- dense per-row score tables;
- checksums for generated protected media;
- precomputed features when permitted by the upstream dataset terms;
- selected lightweight model outputs.

Raw media, extracted frames, face crops, protected-media derivatives, and model
weights should only be redistributed if the original license explicitly allows
it.
