# EVIDENCE — 23L-0578_A2_Ops VisionGuard

Status key: ✅ evidence exists in the repo · ⏳ needs an action from the student (account/login/other network)

| Sub-task | Evidence (file in repo) | Report § | Status |
|----------|-------------------------|----------|--------|
| 1.1 | `report/screenshots/SS_01_env_versions.png`, `requirements.txt` | §1.1 | ✅ |
| 1.2 | `scripts/audit_dataset.py`, `data/audit_*.csv`, `SS_03_audit_summary.png` | §1.2 | ✅ |
| 1.3 | `report/figures/sample_grid.png`, `hist_*.png` | §1.3 | ✅ |
| 1.4 | `data/raw.dvc`, `SS_07_dvc_roundtrip.png`, toy `SS_08_dvc_toy.png` | §1.4 | ✅ (DagsHub repo page screenshot SS_06 ⏳ — needs login) |
| 2.1–2.6 | `scripts/train.py`, augmented batch, curves, CM/ROC, misclassified grid | §2.1–2.6 | ✅ |
| 2.7 | `scripts/tune.py`, `SS_15_tuner_trials.png`, toy `SS_16_fashion_tuner.png` | §2.7 | ✅ |
| 3.1 | `labs/03_local_mlflow.py`, `SS_17*.png` | §3.1 | ✅ |
| 3.2 | `SS_18_env_gitignored.png`, `SS_19_dagshub_mlflow.png`, `SS_19b_toy_dagshub_run.png` | §3.2 | ✅ |
| 3.3 | `SS_20_run_manual_overview.png`, `SS_20b_run_manual_metrics.png`, `SS_21_run_autolog_overview.png` | §3.3 | ✅ |
| 3.4 | `SS_22_run_artifacts.png` | §3.4 | ✅ |
| 3.5 | `SS_23a_dagshub_runs_list.png`, `SS_23b_dagshub_compare.png`, `sweep_comparison.csv` | §3.5 | ✅ |
| 3.6 | `SS_24_registry.png`, `SS_24b_registry_alias_api.png`, `SS_25_predict_by_alias.png` | §3.6 | ✅ |
| 4.1–4.4 | `app/`, `SS_27`–`SS_33` | §4 | ✅ |
| 5 | Postgres trail `logs/27_postgres_trail.log`, `SS_34`, `SS_35b` | §5 | ✅ |
| 6.1 | `logs/20_docker_warmup.log`, `SS_36` | §6.1 | ✅ |
| 6.2 | `Dockerfile`, `SS_32b_layer_cache.png`, `SS_37` (appuser) | §6.2 | ✅ |
| 6.3 | `docker-compose.yml`, `SS_38_compose_ps_healthy.png`, `SS_37` (closed DB port) | §6.3 | ✅ |
| 6.4 | `SS_40_image_size_before_after.png` (2.18 GB → 2.00 GB) | §6.4 | ✅ |
| 6.5 | `SS_41a`–`SS_41d`, `logs/32`–`35_*.log` | §6.5 | ✅ |
| 6.6 apply | `.github/workflows/ci.yml`, green run `SS_42_ci_green.png`, badge in README | §6.6 | ✅ |
| 6.6 toy | `labs/06_ci_toy/`, repo `zayn-zeny/23L-0578_ci_toy` (private): runs #1 green, #2 red, #3 green, `SS_43` | §6.6 | ✅ (optional: browser screenshots of the Actions tab) |
| 7 | `vercel_api/`, `scripts/export_tflite.py`, live URL https://23l-0578-a2-ops.vercel.app, `SS_44a`–`SS_45` | §7 | ✅ deployed + tested, Postman collection via Newman `SS_46` (10/10) · ⏳ hotspot test, Vercel dashboard screenshots |
| 8 | Flask (Compose) → live Vercel API: `SS_49b`, `SS_49c`, `SS_50b`–`SS_50d`, `SS_51b` | §8 | ✅ |
| Bonus | `report/figures/gradcam_grid.png`, `scripts/gradcam.py` | Bonus | ✅ |
| Report | `report/23L-0578_VisionGuard.pdf` built by `scripts/build_report.py` | — | ✅ (pending boxes for the ⏳ rows) |
