# Prototype Code

This directory contains the project-local orchestration layer for the
protected-real face-swap security experiments.

## Layout

- `configs/`: public experiment configs and the `paths.example.yaml` template.
- `scripts/`: manifest construction, probe execution, calibration, comparator,
  leakage-audit, and paper-result assembly scripts.
- `src/hybrid_router/`: reusable path/config and manifest-schema helpers.
- `manifests/`: documentation for full CSV manifests; large CSV manifests are
  ignored by Git.
- `reports/`: lightweight result summaries; dense per-row CSV score dumps are
  ignored by Git.

## Local Setup

Copy the path template and fill it in:

```powershell
Copy-Item prototype/configs/paths.example.yaml prototype/configs/paths.local.yaml
```

Install the lightweight smoke-test dependencies:

```powershell
python -m pip install -r prototype/requirements-smoke.txt
```

Run the data/path audit before launching experiments:

```powershell
python prototype/scripts/audit_topconf_data.py --paths prototype/configs/paths.local.yaml
```

## Minimal Smoke Workflow

If a small FaceForensics++ subset is available locally, build a smoke manifest:

```powershell
python prototype/scripts/build_ffpp_smoke_manifest.py `
  --config prototype/configs/ffpp_smoke_v0_1.yaml
```

Create provenance and passive-detector probe job sheets:

```powershell
python prototype/scripts/run_videoseal_probe.py `
  --config prototype/configs/ffpp_smoke_v0_1.yaml `
  --paths prototype/configs/paths.local.yaml

python prototype/scripts/run_passive_probe.py `
  --config prototype/configs/ffpp_smoke_v0_1.yaml `
  --paths prototype/configs/paths.local.yaml
```

Merge outputs after the external probes finish:

```powershell
python prototype/scripts/merge_smoke_outputs.py `
  --config prototype/configs/ffpp_smoke_v0_1.yaml
```

## Paper-Facing Diagnostics

The TCSVT-oriented paper uses scripts such as:

- `build_vcf_ci_safe_row.py`
- `build_vcf_subject_source_audit.py`
- `build_targeted_tcsvt_diagnostics.py`
- `build_vcf_temporal_standard_metrics.py`
- `execute_altfreezing_native_facetrack_manifest_probe.py`
- `execute_m2f2_manifest_probe.py`

The exact result files are listed in the paper supplement and summarized in the
root `DATA.md`.

## Release Boundary

Do not commit raw datasets, model weights, third-party baseline repos, extracted
features, dense per-row score dumps, or local path files. They belong in
`workspace/` or an external artifact release, not in the Git repository.
