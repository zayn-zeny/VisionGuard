# RUNLOG — 23L-0578_A2_Ops VisionGuard
# Format: [TIMESTAMP] [TASK] [ACTION] [RESULT/NOTE]

## 2026-11-06

[08:50] [ENV]    Docker 29.8.2 + Compose v5.6.0 confirmed ✓
[08:50] [ENV]    Python 3.13.5 confirmed ✓
[08:51] [ENV]    venv created at d:\Zayn\mlops_Assignment\A2\venv ✓
[08:51] [GIT]    git init + user config ✓
[08:52] [SETUP]  Directory scaffold created ✓
[08:52] [SETUP]  .gitignore, .env.example, README.md written ✓
[09:49] [TASK 1.1] Python 3.11.9 venv established, exact package versions pinned in requirements.txt ✓
[09:49] [TASK 1.1] Verified TF 2.17.0, NumPy 1.26.4, PIL 10.4.0, MLflow 2.17.0, DVC 3.55.2, Flask 3.0.3 ✓
[09:49] [TASK 1.1] GPU check: GPUs: [] (CPU only confirmed) ✓
[11:08] [TASK 1.2] Kaggle dataset downloaded & extracted (974 images: 539 AiArtData, 435 RealArt) ✓
[11:15] [TASK 1.2] scripts/audit_dataset.py executed: 0 corrupt files, 2 duplicates (f3268a4fc047...) ✓
[11:17] [TASK 1.3] scripts/plot_dataset.py executed: generated sample_grid.png, hist_width.png, hist_height.png, hist_filesize.png ✓
[11:18] [TASK 1.5] scripts/split_dataset.py executed: seeded split (SEED=42) -> Train: 681, Val: 146, Test: 146 ✓
[11:24] [TASK 1.4] dvc init executed, data/raw tracked via DVC (data/raw.dvc, md5: b2cec67839f742df4653f66d4bcbbafb.dir) ✓
[11:40] [TASK 1.4] DagsHub repo zayn-zeny/VisionGuard created & verified via API ✓
[11:42] [TASK 1.4] DagsHub DVC remote configured (https://dagshub.com/zayn-zeny/VisionGuard.dvc) ✓
[11:44] [TASK 1.4] dvc push executed: 975 files pushed to DagsHub remote storage ✓
[11:45] [TASK 1.4] dvc round-trip proven: deleted data/raw locally and restored via dvc pull (975 files restored) ✓
[11:46] [TASK 3.2] .env git-ignored proof verified via git check-ignore -v .env ✓

