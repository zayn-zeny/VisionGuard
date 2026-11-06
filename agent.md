# VisionGuard (23L-0578_A2_Ops) — Agent Context Transfer Document

> **Purpose:** This file preserves the complete state, architecture, completed milestones, pending deliverables, and execution instructions for VisionGuard (Assignment 2, MLOps, 100 marks). Any AI agent or developer opening this project in any IDE can read this document and immediately resume work with zero context loss.

---

## 1. Project & Student Identity

| Attribute | Value |
|---|---|
| **Student Name** | Zain Ul Abideen |
| **Roll Number** | `23L-0578` |
| **Course & Section** | MLOps — Section 7A |
| **Instructor** | *(Explicitly omitted per student instruction — do not add placeholders)* |
| **Project & Repo Name** | `23L-0578_A2_Ops` |
| **GitHub Target** | `https://github.com/zayn-zeny/23L-0578_A2_Ops` (Public) |
| **DagsHub Target** | `https://dagshub.com/zayn-zeny/23L-0578_A2_Ops` |
| **Dataset** | Kaggle `cashbowman/ai-generated-images-vs-real-images` |
| **Primary Report** | `report/23L-0578_VisionGuard.pdf` (compiled from `report/report.md`) |

---

## 2. Core Architecture & Design Choices

```
                        ┌────────────────────────────────────────┐
                        │   Cloud Inference API (Task 7)         │
                        │   (FastAPI/Uvicorn, Public Internet)   │
                        │   POST /predict (X-API-Key header)     │
                        └────────────────────────────────────────┘
                                           ▲
                                           │ HTTP POST (Remote Mode)
                                           │
┌─────────────────────────┐                │
│   Client Browser        │                │
│   (Uploads Image)       │                │
└───────────┬─────────────┘                │
            │                              │
            ▼                              ▼
┌────────────────────────────────────────────────────────┐
│   Flask Web Application (localhost:5000)               │
│   - Inference Layer: swappable via INFERENCE_MODE env  │
│       * local: in-process Keras MobileNetV2            │
│       * remote: HTTP client calling Cloud API          │
│   - Memory-only file decoding (zero disk persistence)  │
│   - Enforces extensions, MIME, 10MB limit, PIL verify  │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ Writes every prediction row
                            ▼
┌────────────────────────────────────────────────────────┐
│   PostgreSQL 16 Database                               │
│   - Service name 'db' on internal docker network       │
│   - Named persistent volume: 23l0578_visionguard_pgdata│
│   - No host port published (isolated from outside)     │
│   - Healthcheck: pg_isready                            │
└────────────────────────────────────────────────────────┘
```

### Key Rubric Decisions & Justifications
1. **Swappable Inference Layer:** `INFERENCE_MODE=local` runs for Tasks 4–6 inside the full Docker Compose stack; `INFERENCE_MODE=remote` runs for Tasks 7–8 calling the cloud inference microservice over HTTP.
2. **Model Loading:** Champion model loaded *once* at startup in module memory (`load_model_once`). Per-request loading is prohibited due to memory churn and 2–5s latency penalty.
3. **Preprocessing Distribution:** `mobilenet_v2.preprocess_input` scales pixels to `[-1, 1]` identically in both training and serving to eliminate silent training-serving skew.
4. **Resizing Strategy:** Squash to (224, 224) using bilinear interpolation rather than center crop to avoid discarding peripheral synthetic artefacts.
5. **No Upload Disk Persistence:** Uploaded image bytes are verified with `PIL.Image.open(io.BytesIO(file_bytes)).verify()` and processed entirely in RAM.
6. **Hardware Constraints:** Host is CPU-only. Training budget is ≤15 mins/run (stratified subsample to 2,000 images/class with SEED=42, while keeping test set untouched).

---

## 3. Environment & Local Machine Setup

- **Host OS:** Windows 10/11
- **Python Runtime:** Python 3.11.9 (`C:\Users\saady\AppData\Local\Programs\Python\Python311\python.exe`)
- **Virtual Environment:** `D:\Zayn\mlops_Assignment\A2\venv`
- **Docker & Compose:** Docker 29.8.2, Docker Compose v5.6.0
- **Disk Configuration Note:** Drive `C:` has low free space (~1–2 GB). Drive `D:` has 20 GB free. All pip temp directories and wheel download caches must be directed to `D:\Zayn\tmp` and `D:\Zayn\.cache\pip`.
- **Active Git Commit:** Initial commit `754be94` ("feat(setup): init VisionGuard scaffold, app, tests, scripts and Task 1.1 environment").

### To Activate the Environment in Any Shell:
```powershell
cd D:\Zayn\mlops_Assignment\A2
.\venv\Scripts\Activate.ps1
```

---

## 4. Current Milestone Status Checklist

| Task | Title | Status | Evidence / Artifact |
|---|---|---|---|
| **0** | Pre-flight interview & confirmed settings | **DONE** | Logged in chat & confirmed settings |
| **1.1** | Isolated env, pinned deps, GPU check transcript | **DONE** | `requirements.txt`, `report/figures/task_1_1_env_transcript.txt` |
| **1.2** | Audit script & Kaggle dataset download | **PAUSED** | Waiting for user to supply `kaggle.json` |
| **1.3** | Class sample grid & distributions | Pending | `scripts/plot_dataset.py` ready to execute |
| **1.4** | DVC init, track `data/raw`, DagsHub remote | Pending | Code & commands ready; needs DagsHub token |
| **1.4 (toy)** | Practice CSV DVC versioning | Queued | Runs after Tasks 1–8 |
| **1.5 / Splits** | Stratified train/val/test split lists (seeded) | Pending | `scripts/split_dataset.py` ready to execute |
| **2.1** | `tf.data` pipeline, squash resize, aug figure | Ready | Implemented in `scripts/train.py` |
| **2.2** | Hand-crafted CNN baseline model summary | Ready | Implemented in `scripts/train.py` |
| **2.3** | MobileNetV2 2-stage transfer learning | Ready | Implemented in `scripts/train.py` |
| **2.4** | Seeds (42), EarlyStopping, ReduceLROnPlateau | Ready | Implemented in `scripts/train.py` |
| **2.5** | Single test-set evaluation, val threshold sweep | Ready | Implemented in `scripts/train.py` |
| **2.6** | Misclassified 10-image grid with hypotheses | Ready | Implemented in `scripts/train.py` |
| **2.7** | Keras Tuner ≥8 trials (LR, dropout, unfreeze) | Ready | `scripts/train.py --sweep` |
| **2.7 (toy)** | Fashion-MNIST tuner | Queued | Runs after Tasks 1–8 |
| **3.1 (toy)** | Local MLflow lab & glossary | Glossary Done | Lab runs after Tasks 1–8 |
| **3.2** | DagsHub project & MLflow remote tracking URI | Pending | Needs DagsHub token at Task 1.4/3.2 |
| **3.3** | MLflow logging (manual callback vs autolog) | Ready | Implemented in `scripts/train.py` |
| **3.4** | Attach figures, splits, requirements, model | Ready | Implemented in `scripts/train.py` |
| **3.5** | ≥6 distinct MLflow runs | Ready | Sweep matrix defined in `scripts/train.py` |
| **3.6** | Model registry, 2 versions, `champion` alias | Ready | `scripts/register_champion.py`, `scripts/predict_by_alias.py` |
| **4.1** | `inference.py` startup loading & validation | **DONE** | Written in `app/inference.py` |
| **4.2** | Safe routes, MIME/extension/PIL checks | **DONE** | Written in `app/routes.py` |
| **4.3** | Dark UI HTML/CSS templates (upload, result, history) | **DONE** | `app/templates/` & `app/static/css/style.css` |
| **4.4** | Env config & `debug=False` production health | **DONE** | `app/__init__.py`, `app/routes.py` |
| **5** | PostgreSQL prediction audit trail table | **DONE** | `app/database.py` (SQLAlchemy model) |
| **6.1 (toy)** | Docker warm-up image | Queued | Runs after Tasks 1–8 |
| **6.2** | Optimized Dockerfile (slim base, layer order, non-root) | **DONE** | `Dockerfile` |
| **6.3** | Compose `web` + `db` (healthy check, isolated network) | **DONE** | `docker-compose.yml` |
| **6.4** | Image size before/after optimization | Ready | `docker history` analysis |
| **6.5** | Logs, restart, intentional break & diagnostic fix | Ready | Test script prepared |
| **6.6** | GitHub Actions CI workflow & badge | **DONE** | `.github/workflows/ci.yml`, test suite in `tests/` |
| **6.6 (toy)** | Red-then-green matrix CI repo | Queued | Runs after Tasks 1–8 |
| **7** | Standalone Cloud Inference API deployment | Ready | `cloud_api/` FastAPI service prepared |
| **8** | Remote mode end-to-end integration & Postman test | Ready | Tested via `INFERENCE_MODE=remote` |
| **Bonus** | Grad-CAM visualization (4 correct + 4 wrong) | Ready | Implement after Tasks 1–8 |
| **Report** | PDF report `23L-0578_VisionGuard.pdf` | In Progress | Drafted in `report/report.md` |

---

## 5. File Map & Directory Inventory

```
D:\Zayn\mlops_Assignment\A2\
├── agent.md                        # THIS FILE (cross-IDE context handover)
├── requirements.txt                # Pinned exact versions (Python 3.11)
├── .env.example                    # Template with dummy keys
├── .gitignore                      # Strictly ignores .env, venv, data/raw, models/*.keras
├── .dockerignore                   # Strips git, venv, logs, secrets, data from Docker context
├── Dockerfile                      # Python 3.11-slim, non-root appuser, cached pip layer
├── docker-compose.yml              # 2 services: web + db (Postgres 16, named volume, no host port)
├── README.md                       # Architecture diagram, quick start, CI badge
├── RUNLOG.md                       # Chronological log of execution & bug fixes
├── EVIDENCE.md                     # Checklist mapping Sub-tasks to artifacts
│
├── app/                            # VisionGuard Flask Web Application
│   ├── __init__.py                 # Application factory, env loader, debug=False
│   ├── routes.py                   # Routes: /, /predict, /result, /history, /health
│   ├── inference.py                # Swappable inference engine (local Keras / remote Cloud API)
│   ├── database.py                 # SQLAlchemy Prediction model
│   ├── wsgi.py                     # Entry point for Gunicorn
│   ├── templates/
│   │   ├── base.html               # Shared navbar, notifications, theme
│   │   ├── upload.html             # Drag-and-drop file upload form
│   │   ├── result.html             # Verdict, confidence bar, metadata, caveat
│   │   ├── history.html            # Paginated history table
│   │   └── error.html              # Custom friendly HTTP 400/413/415/422/500/502 errors
│   └── static/css/
│       └── style.css               # Clean modern dark mode responsive styling
│
├── cloud_api/                      # Task 7 Standalone Cloud Inference API
│   ├── main.py                     # FastAPI application with X-API-Key auth
│   ├── requirements.txt            # Minimal inference dependencies
│   └── Dockerfile                  # Lightweight container for cloud deployment
│
├── scripts/                        # Automation & Pipeline Scripts
│   ├── audit_dataset.py            # Task 1.2 image integrity, dimensions, SHA-256 audit
│   ├── plot_dataset.py             # Task 1.3 class grids and histograms
│   ├── split_dataset.py            # Task 1.4 deterministic stratified split generator
│   ├── train.py                    # Tasks 2 & 3 training, evaluation, MLflow logger & sweep
│   ├── register_champion.py        # Task 3.6 MLflow model registry champion alias promoter
│   └── predict_by_alias.py         # Task 3.6 script predicting via champion alias
│
├── tests/                          # Automated Pytest Suite (Task 6.6)
│   ├── __init__.py
│   ├── conftest.py                 # In-memory SQLite fixtures & sample test images
│   └── test_app.py                 # Tests: shape, thresholds, extension/PIL rejection, /health
│
└── report/                         # University PDF Deliverable Source
    ├── report.md                   # Full assignment report structured chapter-by-chapter
    ├── SCREENSHOTS.md              # Screenshot manifest tracking numbers and capture criteria
    └── figures/                    # Saved plots, confusion matrices, ROC curves, logs
        └── task_1_1_env_transcript.txt
```

---

## 6. How the Next Agent / IDE Resumes Immediately

1. **Next Step:** As soon as the user provides `kaggle.json` (either in `C:\Users\saady\.kaggle\kaggle.json` or `D:\Zayn\mlops_Assignment\A2\kaggle.json`):
   - Run dataset download:
     ```powershell
     .\venv\Scripts\python.exe -m kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data/raw --unzip
     ```
   - Execute dataset audit:
     ```powershell
     .\venv\Scripts\python.exe scripts/audit_dataset.py --data-dir data/raw
     ```
   - Generate dataset visualizations:
     ```powershell
     .\venv\Scripts\python.exe scripts/plot_dataset.py
     ```
   - Generate stratified splits (2,000 images per class for CPU budget):
     ```powershell
     .\venv\Scripts\python.exe scripts/split_dataset.py --data-dir data/raw --max-per-class 2000
     ```
2. **Next Checkpoint:** Prompt user for GitHub token and DagsHub credentials to initialize git remote and DVC remote (Task 1.4 & Task 3.2).
3. **Keep `RUNLOG.md`, `EVIDENCE.md`, and `agent.md` synchronized.**
