# GitHub Release Checklist

Use this checklist before creating the public repository or submission artifact.

## Must Include

- `README.md` with project purpose, repository layout, and quick-start commands.
- `DATA.md` explaining what data/weights are excluded and how to recreate local
  paths.
- `DATASETS.md` explaining dataset/baseline acquisition and license boundaries.
- `prototype/scripts/` and `prototype/src/` for experiment orchestration.
- `prototype/configs/paths.example.yaml` as a public path template.
- `prototype/reports/README.md` and `prototype/manifests/README.md` explaining
  where full artifacts live.

## Must Not Include

- `.conda/` or any virtual environment.
- `workspace/` raw datasets, external repos, checkpoints, features, downloads,
  or cache files.
- `pilot/passive_baseline/runs/` checkpoint-heavy training outputs.
- `prototype/reports/*.csv` dense per-row score dumps unless explicitly moved to
  Git LFS or an external artifact release.
- `prototype/manifests/*.csv` full paper-facing manifests unless licensing and
  repository-size constraints are handled.
- Local path files such as `prototype/configs/paths.local.yaml`.
- `paper/` LaTeX source, compiled manuscript PDFs, LaTeX build logs, and
  rendered page-check PNGs unless you explicitly choose a paper-release branch.

## Suggested First Commit

```powershell
git init
git add .gitignore README.md DATA.md DATASETS.md docs prototype pilot requirements.txt
git status
git commit -m "Prepare reproducible face-swap security research release"
```

Before pushing, inspect the staged size:

```powershell
git diff --cached --stat
git status --short
```

If any large dataset/checkpoint file appears, unstage it and update
`.gitignore`.

## After GitHub Upload

- Add repository description and topics: `deepfake-detection`,
  `video-forensics`, `face-swap`, `protected-media`, `reproducibility`.
- Decide a code license before making the repository public.
- For large artifacts, create a GitHub Release or Zenodo archive and link it
  from `DATA.md`.
- If the paper is still under review, check the journal's anonymity and
  supplementary-material rules before linking public code.
