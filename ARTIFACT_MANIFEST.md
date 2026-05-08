# Artifact Manifest

This manifest lists the report files, scripts, and figure generators used by the paper and supplement. Paths are relative to the project root.

Integrity hashes for the locally available listed artifacts are recorded in `ARTIFACT_CHECKSUMS.sha256`. Large or license-restricted artifacts may be distributed only through a journal supplement, GitHub Release, Zenodo record, institutional storage, or authorized local regeneration.

## Core Result Files

- `prototype/reports/journal_final_unified_sota_ablation_stability_bundle_2026-04-23.zh-CN.md`
- `prototype/reports/journal_final_unified_sota_rows_2026-04-23.tsv`
- `prototype/reports/vcf_temporal_ablation_rows_2026-04-22.tsv`
- `prototype/reports/df40_ffpp_test_paired_all_methods_2026-04-20.tsv`
- `prototype/reports/external_sota_breadth_vcf_temporal_update_2026-04-24.tsv`
- `prototype/reports/vcf_temporal_standard_metrics_2026-04-24.tsv`
- `prototype/reports/video_sota_ftcn_tt_vcf_temporal_2026-04-23.zh-CN.md`
- `prototype/reports/vcf_c23_clip32_ftcn_tt_videoseal_calibrated_summary_2026-04-24.json`
- `prototype/reports/vcf_c23_temporal_real_concept_summary_2026-04-24.json`
- `prototype/reports/vcf_fpr_confidence_operating_points_2026-04-24.tsv`
- `prototype/reports/protection_pipeline_shift_smoke_2026-04-24.tsv`
- `prototype/reports/tcsvt_risk_closure_2026-05-07.tsv`
- `prototype/reports/tcsvt_risk_closure_2026-05-07.md`
- `prototype/reports/targeted_tcsvt_diagnostics_2026-04-26.json`
- `prototype/reports/source_family_error_decomposition_2026-04-26.tsv`
- `prototype/reports/source_family_robust_guarded_summary_2026-05-08.tsv`
- `prototype/reports/source_family_robust_guarded_summary_2026-05-08.json`
- `prototype/reports/temporal_ablation_review_table_2026-04-26.tsv`
- `prototype/reports/temporal_order_control_2026-04-26.tsv`
- `prototype/reports/calibration_fraction_sensitivity_2026-04-26.tsv`
- `prototype/reports/paper_readiness_gate_final_unified_2026-04-23.zh-CN.md`

## Frontier Comparator Diagnostics

- `prototype/reports/frontier_altfreezing_native_fullscale_mf64_mc4_2026-04-25.md`
- `prototype/reports/altfreezing_native_fullscale_mf64_mc4_2026-04-25/vcf_c23_altfreezing_native_fullscale_summary_2026-04-25.json`
- `prototype/reports/frontier_m2f2_slice2048_2026-04-25.md`
- `prototype/reports/vcf_c23_m2f2_stage1_slice2048_summary_2026-04-25.json`

## Leakage and Audit Scripts

- `prototype/scripts/build_vcf_subject_source_audit.py`
- `prototype/reports/vcf_temporal_subject_source_audit_2026-04-25.tsv`
- `prototype/reports/vcf_temporal_subject_source_audit_2026-04-25.json`
- `prototype/scripts/build_vcf_ci_safe_row.py`

## Figure Generation

- `paper/figures/gen_paper_figures.py`
- `paper/figures/gen_qualitative_figures.py`

The figure scripts read stored result files and manifests, then write vector PDFs plus PNG previews to `paper/figures/`. Reported metrics are not manually typed into plotting scripts except axis labels, visual labels, and the fixed FPR08 gate value.
