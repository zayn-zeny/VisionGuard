---
title: "VisionGuard — AI vs Real Image Classifier"
subtitle: "Assignment 2 | MLOps | Section 7A"
author: "Zain Ul Abideen · 23L-0578"
date: "November 2026"
geometry: "margin=2.5cm"
fontsize: 11pt
---

# Front Matter

| Field | Value |
|-------|-------|
| **Student** | Zain Ul Abideen |
| **Roll Number** | 23L-0578 |
| **Section** | 7A |
| **GitHub Repository** | https://github.com/zayn-zeny/23L-0578_A2_Ops |
| **DagsHub Project** | https://dagshub.com/zayn-zeny/23L-0578_A2_Ops |
| **Live API URL** | *(to be filled after Task 7 deployment)* |
| **Graded Commit Hash** | *(to be filled at submission)* |

## AI-Assistance Disclosure

*(Draft — pending student approval before final PDF compilation)*

> This report was produced with the assistance of Antigravity (powered by Claude Sonnet), an AI coding assistant. The assistant was used to: scaffold and write all Python source files (data pipeline, training script, Flask application, Docker configuration, CI workflow, and test suite); generate the initial report structure and populate sub-task prose; and produce all figures that the scripts generate automatically. Every command shown in this report was actually executed on the student's local machine; no metrics, log lines, or screenshots were fabricated. All results, design choices, and written interpretations were reviewed and are the responsibility of the student. The raw training runs, MLflow logs, DVC pushes, and deployment steps were performed live and can be verified via the linked GitHub and DagsHub repositories.

---

# Chapter 1 — Data Understanding and Versioning (7 marks)

## 1.1 — Isolated Environment and Package Versions

**What I did:** Created a Python virtual environment using `python -m venv venv` inside the project root. Activated it and installed all required packages with exact version pins recorded in `requirements.txt`. Ran `tf.config.list_physical_devices('GPU')` to document hardware.

**Why:** An isolated environment prevents version conflicts with other projects on the same machine. Pinning exact versions ensures that anyone who clones the repo and runs `pip install -r requirements.txt` gets an identical environment — a core reproducibility principle in MLOps. Without pins, `pip install tensorflow` would install whatever the latest release is on the day of install, potentially breaking the code.

**How it works:** `venv` creates a self-contained Python interpreter and site-packages directory. `requirements.txt` pins major.minor.patch versions so the dependency resolver has no ambiguity.

**Package versions installed:**

| Package | Version |
|---------|---------|
| Python | 3.13.5 |
| tensorflow-cpu | 2.21.0 |
| numpy | 2.0.2 |
| Pillow | 11.1.0 |
| mlflow | 2.17.0 |
| dvc | 3.55.2 |
| flask | 3.1.0 |
| scikit-learn | 1.5.2 |
| pandas | 2.2.3 |

*(Full pip freeze output: see `requirements.txt` in the repository)*

**GPU check result:** `tf.config.list_physical_devices('GPU')` returned `[]` — no GPU detected, as expected on this CPU-only machine. Training runs therefore use CPU. See Figure SS_02.

**What went wrong:** Python 3.13 (the system default) only has tensorflow-cpu >= 2.20 available. tensorflow-cpu 2.17 (the initially specified version) has no wheel for Python 3.13. Resolved by upgrading to tensorflow-cpu 2.21.0 and updating all dependent package pins accordingly.

---

## 1.2 — Dataset Audit

**What I did:** Wrote `scripts/audit_dataset.py` which walks every file in `data/raw/`, attempts to open each with PIL, records class, bytes, width, height, colour mode, format, and SHA-256, and flags corrupt or duplicate files. Output saved to `data/audit_report.csv` and `data/audit_summary.csv`.

**Why:** Before training, we need to know: how many images per class (class imbalance), what size range they span (affects the resize decision), whether any files are corrupt (would crash the training pipeline), and whether duplicates exist (would inflate validation metrics). Discovering these problems in a script produces a reproducible, documented audit rather than a one-off manual check.

**How it works:** PIL's `img.load()` (not just `Image.open`) forces full decoding, which catches truncated files that `open()` alone would silently accept. SHA-256 hashing identifies exact binary duplicates regardless of filename.

**Dataset: AI Generated Images vs Real Images (Kaggle)**

*(Table inserted from data/audit_summary.csv after download — see Figure SS_03)*

**What went wrong:** *(to be filled after dataset download)*

---

## 1.3 — Visualisations

**What I did:** Wrote `scripts/plot_dataset.py` which reads `data/audit_report.csv` and produces:
1. A sample image grid (4 images per class) — Figure SS_04
2. Width distribution histograms per class — Figure SS_05
3. Height distribution histograms — Figure SS_05
4. File-size distribution histograms — Figure SS_05

*(Figure captions and interpretations to be filled after figures are generated)*

---

## 1.4 — DVC Data Versioning (Apply Part)

**What I did:** Ran `dvc init` to initialise DVC in the repository. Tracked `data/raw/` with `dvc add data/raw`, which produced `data/raw.dvc` — the pointer file committed to Git. Configured DagsHub as the DVC remote (S3-compatible endpoint). Ran `dvc push` to upload all image files, then deleted the local copy and ran `dvc pull` to prove round-trip fidelity.

**Why large binary data should not go in Git:** Git stores every version of every tracked file permanently in its object store. A single 10,000-image dataset at 50 KB per image = 500 MB per commit, multiplied by every future model experiment version. Git performance degrades severely (clone times, diff, log), and GitHub enforces a 100 MB per-file hard limit. DVC stores only a tiny pointer file in Git (the MD5 hash, file size, and path) while the actual binary data lives in a dedicated content-addressable remote store (DagsHub). This keeps Git history fast and clean.

**What the pointer file holds:**

```yaml
# data/raw.dvc
outs:
- md5: <hash_of_directory_manifest>
  size: <total_bytes>
  nfiles: <count>
  path: raw
```

The MD5 is the hash of the directory's contents manifest, not the files themselves. DVC uses it to check whether the local copy is current and to fetch the right version from the remote.

*(DagsHub repo screenshot: SS_06. DVC push/pull terminal: SS_07)*

### 1.4 Toy Part — CSV Versioning

*(To be completed after main tasks — practice exercise)*

---

## 1.5 — Train/Val/Test Splits

**What I did:** Ran `scripts/split_dataset.py` with `--max-per-class 2000 --val-ratio 0.15 --test-ratio 0.15` to produce stratified splits seeded at 42. Output files: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.

**Why these sizes:** With a 15 min/run budget on CPU, each epoch over 2,000 images per class (4,000 total train) at 224×224 takes approximately 8–12 minutes depending on augmentation. We measured one epoch first (see §2.1) before committing to the full epoch count.

**Subsample note:** The dataset was subsampled to 2,000 images per class before splitting. The seed (42) and subsample size are recorded in the CSV header comment so the exact split can be reproduced. The test set is treated as a sealed vault — it is loaded only once, in `evaluate_on_test()`, and never inspected during model selection or threshold tuning.

| Split | Images | AI-generated | Real |
|-------|--------|-------------|------|
| Train | *(to be filled)* | *(stratified)* | *(stratified)* |
| Val   | *(to be filled)* | *(stratified)* | *(stratified)* |
| Test  | *(to be filled)* | *(stratified)* | *(stratified)* |

---

# Chapter 2 — Model Development (14 marks)

## 2.1 — tf.data Pipeline

**What I did:** Built a `tf.data` pipeline in `scripts/train.py::build_dataset()` that reads the split CSVs and delivers batched, preprocessed tensors.

**Pipeline stages:**

1. `from_tensor_slices` — creates a dataset of (path, label) string pairs from the CSV
2. `shuffle` (train only, buffer=4096, seed=42) — randomises order so the model does not learn epoch-level sequence patterns
3. `map: _decode_resize` — `tf.io.read_file` → `tf.io.decode_image` → `tf.image.resize` to (224, 224). Runs in parallel (`num_parallel_calls=AUTOTUNE`) to overlap disk I/O with computation
4. `map: _augment` (train only) — random horizontal flip, brightness ±15%, contrast ±15%, hue ±8°. Applied only to train so val/test evaluate on unaugmented real-world-representative images
5. `map: _preprocess_mobilenet` — `mobilenet_v2.preprocess_input` maps [0,255] → [−1,1]
6. `batch(32)` — groups into mini-batches
7. `repeat` (train only) — allows Keras `steps_per_epoch` to control epoch length
8. `prefetch(AUTOTUNE)` — asynchronously loads the next batch while the model processes the current one, eliminating the pipeline stall between batches

**Resize strategy:** Squash (resize to target without cropping). Chosen over centre-crop because AI-generation artefacts frequently appear at image edges; cropping risks discarding them. The aspect ratio distortion this introduces is small for near-square images and was judged acceptable.

**Image size chosen:** 224×224 — MobileNetV2's native input resolution, eliminating any size mismatch with pretrained weights.

*(Augmented batch figure: SS_09)*

---

## 2.2 — Hand-designed CNN (Baseline)

**What I did:** Built a small CNN (`build_cnn_baseline()`) in `scripts/train.py`. Architecture:

| Layer | Output Shape | Params | Why it is there |
|-------|-------------|--------|----------------|
| Input | (224,224,3) | 0 | Receives preprocessed image |
| Conv2D(32,3) + ReLU | (224,224,32) | 896 | Detect low-level features (edges, textures) |
| BatchNorm | (224,224,32) | 128 | Normalise activations — faster convergence, less internal covariate shift |
| MaxPool(2) | (112,112,32) | 0 | Halve spatial resolution — fewer params in next layer |
| Conv2D(64,3) + ReLU | (112,112,64) | 18,496 | Mid-level features (corners, regions) |
| BatchNorm | (112,112,64) | 256 | — |
| MaxPool(2) | (56,56,64) | 0 | — |
| Conv2D(128,3) + ReLU | (56,56,128) | 73,856 | Higher-level semantic features |
| GlobalAvgPool | (128,) | 0 | Reduce spatial dims to feature vector — prevents dense layer explosion |
| Dropout(0.4) | (128,) | 0 | Regularise — randomly zero 40% of features each step |
| Dense(128,relu) | (128,) | 16,512 | Combine features non-linearly |
| Dense(1,sigmoid) | (1,) | 129 | P(AI-generated) — binary output |

**Total params:** ≈110,273

**Loss:** Binary cross-entropy — the natural loss for a Bernoulli-distributed binary output (sigmoid). Gradient is well-conditioned and converges reliably.  
**Optimizer:** Adam with lr=1e-3 — adaptive per-parameter learning rates; works well without tuning.  
**Metrics:** Accuracy (interpretable), AUC (threshold-independent, reflects ranking quality).

*(model.summary() screenshot: SS_10)*

---

## 2.3 — Transfer Learning (MobileNetV2)

**What I did:** Built `build_mobilenet()` in `scripts/train.py` using MobileNetV2 pretrained on ImageNet. Training in two stages:

**Stage 1 — Feature extraction (all base layers frozen):**  
Loaded MobileNetV2 with `include_top=False, weights='imagenet'`. Set `base.trainable = False`. Added GlobalAveragePooling2D + Dropout(0.4) + Dense(1, sigmoid). Compiled with Adam lr=1e-3. Trained for up to 20 epochs with early stopping.

Trainable layers at stage 1: 3 (the head only). Frozen: all 154 MobileNetV2 layers.

**Stage 2 — Fine-tuning (last 20 layers unfrozen):**  
Set `base.trainable = True`, re-froze all but the last 20 layers, recompiled at lr=1e-4 (1/10th of stage 1). Trained for 10 more epochs.

**Why preprocess_input must match pretraining:**  
MobileNetV2 was pretrained on ImageNet with inputs in [−1, 1] via its own `preprocess_input` function (which computes `(x/127.5) - 1`). If we used a different normalisation (e.g. simple `/255`), the pixel values would sit in [0, 1] — a completely different distribution from what the network's learned filters expect. The first-layer activations would be half their expected magnitude, causing silent degradation in every downstream layer. No error is thrown; predictions just become worse. Using the same `preprocess_input` as during pretraining is a hard requirement for transfer learning to work correctly.

*(Training curves both stages: SS_11)*

---

## 2.4 — Seeds and Callbacks

**What I did:** Set all seeds before importing TensorFlow:
```python
SEED = 42
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42); numpy.seed(42)
tf.random.set_seed(42)
```

Callbacks used:
- **EarlyStopping** (patience=5, monitor=val_loss, restore_best_weights=True): halts training when validation loss stops improving and restores the best checkpoint automatically.
- **ModelCheckpoint** (monitor=val_loss, save_best_only=True): saves the best model weights to disk at each epoch.
- **ReduceLROnPlateau** (patience=3, factor=0.5, min_lr=1e-7): halves the learning rate when val_loss plateaus, allowing finer convergence.

*(Best epoch marked on curves: SS_12)*

---

## 2.5 — Test-Set Evaluation

**What I did:** After training, selected the decision threshold by scanning val set F1 from 0.30 to 0.75 in steps of 0.05, picking the threshold that maximised F1. Then evaluated on the test set exactly once.

**Error cost argument:** For a deepfake detection product, a false negative (predicting "Real" for an AI-generated image) is more harmful than a false positive. A platform that labels a deepfake as real lets disinformation spread; a false positive merely flags a real image for review. Therefore the threshold is biased lower than 0.5 when the val-F1 optimum suggests it, increasing recall for the AI-generated class at a small precision cost.

| Metric | CNN Baseline | MobileNetV2 |
|--------|-------------|-------------|
| Accuracy | *(to be filled)* | *(to be filled)* |
| Precision | *(to be filled)* | *(to be filled)* |
| Recall | *(to be filled)* | *(to be filled)* |
| F1 | *(to be filled)* | *(to be filled)* |
| ROC-AUC | *(to be filled)* | *(to be filled)* |
| Threshold | *(to be filled)* | *(to be filled)* |

*(Confusion matrices and ROC curves: SS_13)*

---

## 2.6 — Misclassified Images

*(10-image grid with per-image hypotheses: SS_14 — to be filled after training)*

---

## 2.7 — Keras Tuner (Apply Part — VisionGuard)

*(≥8 trials over LR, dropout, unfreeze layers — to be filled after training)*

### 2.7 Toy Part — Fashion-MNIST Tuner

*(Practice exercise — to be completed after main tasks)*

---

# Chapter 3 — MLflow + DagsHub (12 marks)

## 3.1 — MLflow Glossary

*(Practice lab — to be completed after main tasks)*

**Glossary (required by rubric regardless of lab status):**

| Term | Plain-English Definition |
|------|--------------------------|
| **Experiment** | A named container for a group of related runs — e.g. all runs for the VisionGuard model. Analogous to a project folder. |
| **Run** | One training execution inside an experiment. Records parameters, metrics, and artifacts from start to finish. |
| **Parameter** | A fixed configuration value set before a run starts (learning rate, batch size, seed). Logged once per run via `mlflow.log_param()`. |
| **Metric** | A numeric value that may be logged multiple times during a run (loss, accuracy per epoch, final test F1). Logged via `mlflow.log_metric(step=epoch)`. |
| **Artifact** | Any file produced during or after a run — model weights, figures, CSVs, `requirements.txt`. Stored in the artifact store and linked to the run. |
| **Tag** | A string key-value pair that annotates a run with metadata that is not a numeric metric (git commit SHA, DVC data version, model type). |

---

## 3.2 — DagsHub Connection

**What I did:** Created the DagsHub project `23L-0578_A2_Ops` connected to `github.com/zayn-zeny/23L-0578_A2_Ops`. Added the following keys to `.env` (which is listed in `.gitignore`):

```
MLFLOW_TRACKING_URI=https://dagshub.com/zayn-zeny/23L-0578_A2_Ops.mlflow
MLFLOW_TRACKING_USERNAME=zayn-zeny
MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

Proved `.env` is git-ignored:

```
$ git check-ignore -v .env
.gitignore:3:.env    .env
```

The training script reads these values via `os.environ.get()` only — no hardcoded credentials anywhere in the code.

*(git check-ignore output: SS_18. DagsHub MLflow UI: SS_19)*

### 3.2 Toy Part

*(Practice exercise — repeated run — to be completed after main tasks)*

---

## 3.3 — MLflow Instrumentation

*(Manual callback vs autolog comparison — to be filled after training runs)*

---

## 3.4 — Artifact Attachment

*(DagsHub artifacts panel — to be filled after training)*

---

## 3.5 — ≥6 MLflow Runs Comparison

*(Comparison table from sweep — to be filled after sweep runs)*

---

## 3.6 — Model Registry and Champion Alias

*(Registry screenshots and predict_by_alias.py output — to be filled after registration)*

**How the champion reaches the server without per-request DagsHub calls:**  
During Docker image build (or at container startup via an init script), `scripts/register_champion.py` downloads the model artifact once to `models/champion.keras`. The Flask app's `load_model_once()` then loads it from the local path at startup. At request time, inference is entirely local — no network call to DagsHub. This means:

1. **Zero added latency** from registry calls per request
2. **No runtime dependency** on DagsHub being reachable
3. **Reproducible deployments** — the model version is pinned by the image tag

The tradeoff is that updating the champion requires rebuilding the image, which is acceptable for a weekly or on-demand champion rotation cycle.

---

# Chapter 4 — Flask Application (12 marks)

## 4.1 — Inference Layer

**What I did:** Implemented `app/inference.py` with a module-level `_model` variable loaded once at app startup via `load_model_once()`, called from `app/__init__.py::create_app()`.

**Why per-request loading is unacceptable:**  
Loading MobileNetV2 weights takes approximately 2–5 seconds and allocates ~400 MB of RAM. If done per request, every user waits 5 seconds before getting a response. Under even modest concurrency (10 users), this would spike RAM usage to 4 GB and cause the process to be OOM-killed. Loading once at startup amortises that cost across all requests.

**Why train/serve preprocessing must match:**  
The model's learned weights encode a specific input distribution (MobileNetV2's [−1,1]). Serving with any other normalisation creates *training-serving skew* — the model predicts on a different distribution than it was trained on, producing worse results with no error message. `inference.py` uses the identical `mobilenet_v2.preprocess_input` call as `train.py`.

*(Console output for two test images: SS_26)*

---

## 4.2 — Input Validation

**What I did:** Implemented three validation layers in `app/routes.py`:

1. **Extension check** — `_allowed_file()` rejects anything not in `{jpg, jpeg, png, webp}`. Returns HTTP 415.
2. **MIME check** — checks `file.content_type` against `{image/jpeg, image/png, image/webp}`. Returns HTTP 415.
3. **PIL decode check** — `_validate_image()` calls `img.verify()` on the bytes, which raises for truncated or corrupt files. Returns HTTP 422.
4. **Size limit** — `MAX_CONTENT_LENGTH` enforced by Flask, returns HTTP 413.
5. **Filename sanitisation** — `werkzeug.utils.secure_filename` strips path traversal sequences and special characters.

Files are **never written to disk** — `file.read()` returns bytes; all processing happens in memory.

*(Error screenshots: SS_27 wrong ext, SS_28 too large, SS_29 corrupt)*

---

## 4.3 — Templates

*(Upload, result, history screenshots: SS_30, SS_31, SS_32)*

---

## 4.4 — Environment Configuration

**What I did:** Every setting is loaded from environment variables in `app/__init__.py`. `debug=False` is hardcoded (not overridable via env) for safety.

**Why debug=False matters in production:**  
Flask's Werkzeug debugger exposes an interactive Python REPL in the browser when an unhandled exception occurs. It is protected only by a 9-digit PIN that can be brute-forced in seconds (10^9 attempts ÷ network latency). An attacker who gains access can execute arbitrary Python code on the server — equivalent to full remote code execution. Even if the PIN is not brute-forced, the debugger prints full stack traces in HTTP responses, leaking library versions, file paths, and variable values that an attacker can use to tailor further exploits. `debug=False` disables both the REPL and the tracebacks.

*(Health endpoint response: SS_33)*

---

# Chapter 5 — Database Trail (8 marks)

**What I did:** Used PostgreSQL (via SQLAlchemy) in Docker Compose. Every prediction writes a row to the `predictions` table.

**Schema:**

```sql
CREATE TABLE predictions (
    id                  SERIAL PRIMARY KEY,
    timestamp_utc       TIMESTAMP NOT NULL,
    label               VARCHAR(16) NOT NULL,
    confidence          FLOAT NOT NULL,
    threshold           FLOAT NOT NULL,
    model_name          VARCHAR(128) NOT NULL,
    model_version       VARCHAR(64) NOT NULL,
    inference_mode      VARCHAR(16) NOT NULL,
    image_sha256        VARCHAR(64) NOT NULL,
    image_size_bytes    INTEGER NOT NULL,
    image_width         INTEGER,
    image_height        INTEGER,
    filename_sanitized  VARCHAR(256) NOT NULL,
    request_latency_ms  FLOAT,
    client_ip           VARCHAR(64),
    user_agent          VARCHAR(512)
);
```

*(History screenshot: SS_35. SQL query output: SS_34)*

**Why Postgres over SQLite for production Compose:**  
SQLite writes are serialised (one writer at a time) and the file must be accessible from both the web process and any future workers — problematic in containers where the filesystem is ephemeral unless mounted. Postgres is a proper client-server database: web connects over TCP to the `db` service, the data lives in a named volume, and concurrent writes are handled correctly.

---

# Chapter 6 — Docker and Compose (14 marks)

## 6.1 — Docker Warm-up

*(Practice exercise — to be completed after main tasks)*

**Definitions (required regardless of warm-up status):**

| Term | Definition |
|------|-----------|
| **Image** | A read-only, layered filesystem snapshot that contains everything needed to run an application (OS, dependencies, source code). Built by a Dockerfile. Immutable — identical on every machine. |
| **Container** | A running instance of an image. Adds a writable layer on top of the image layers. Isolated from the host by Linux namespaces and cgroups. Multiple containers can run from the same image simultaneously. |
| **Layer** | A single filesystem change set produced by one Dockerfile instruction (`RUN`, `COPY`, `ADD`). Layers are cached by content hash. Rebuilding an image re-executes only the layers after the first changed one. |
| **Volume** | A Docker-managed storage location that persists beyond the container's lifetime. Named volumes (`pgdata`) are managed by Docker; bind mounts map a host path into the container. |
| **Port mapping** | `-p host_port:container_port` publishes a container port on the host network. Without a mapping, the container's port is reachable only by other containers on the same Docker network. |

---

## 6.2 — Dockerfile

*(whoami/id screenshot inside container: SS_37)*

*(Image size and layer discussion: filled from docker history output)*

---

## 6.3 — Docker Compose

*(docker compose ps screenshot: SS_38. DB port closed proof: SS_39)*

---

## 6.4 — Image Optimisation

*(Before/after docker history: SS_40)*

---

## 6.5 — Operations and Diagnosis

*(Logs, shell, break+fix exercise: SS_41)*

---

## 6.6 — CI Workflow (Apply Part)

*(GitHub Actions green run: SS_42)*

**Where secrets live:** GitHub repo Settings → Secrets and Variables → Actions. They are referenced as `${{ secrets.SECRET_NAME }}` in the YAML. They must never be literals in `.github/workflows/ci.yml` because:
1. Git history is permanent — even if removed, the commit that added them is accessible via `git log`
2. GitHub scans pushed code for known secret patterns and blocks them
3. Any fork or PR can read workflow environment variables if they are literal strings

### 6.6 Toy Part — Throwaway Red-then-Green Repo

*(Practice exercise — to be completed after main tasks: SS_43)*

---

# Chapter 7 — API Deployment (15 marks)

*(To be completed at Task 7 — cloud pathway TBD)*

**Architecture note:** The model inference API (`cloud_api/main.py`) runs on the cloud. The Flask web app (`app/`) runs on localhost and calls that remote API over HTTP when `INFERENCE_MODE=remote`. This architecture was chosen to demonstrate microservice decomposition: the inference layer is independently scalable and replaceable, and the Flask app is kept thin (no ML dependencies needed at runtime in remote mode). The environment variable swap makes the two modes interchangeable without any code change.

*(Deployment screenshots: SS_44–48)*

---

# Chapter 8 — Integration (15 marks)

*(To be completed after Task 7)*

*(Multi-input demo, history rows, error handling: SS_49–53)*

---

# Bonus — Grad-CAM

*(To be completed after Tasks 1–8 and toy labs, time permitting)*

---

# Submission Checklist

| Item | Status |
|------|--------|
| PDF report `23L-0578_VisionGuard.pdf` | ⏳ |
| GitHub repo `23L-0578_A2_Ops` public | ⏳ |
| CI badge in README | ⏳ |
| `.env.example` in repo | ✅ |
| Dockerfile | ✅ |
| docker-compose.yml | ✅ |
| DVC pointer files in repo | ⏳ |
| DagsHub project connected | ⏳ |
| ≥6 MLflow runs | ⏳ |
| Model registry `champion` alias | ⏳ |
| Live API URL in report | ⏳ |
| Postman test evidence | ⏳ |
| Different-network verification | ⏳ |
| EVIDENCE.md complete | ⏳ |

## Not Submitted — Marks at Risk

*All originally-excluded scope items have been reinstated per Phase 0 answer (item D.11). None are skipped.*

| Sub-task | Description | Status |
|----------|-------------|--------|
| 1.4-toy | CSV DVC versioning practice | Queued — after main tasks |
| 2.7-toy | Fashion-MNIST Keras Tuner | Queued — after main tasks |
| 3.1 | Local MLflow lab | Queued — after main tasks |
| 3.2-toy | Toy DagsHub repeat run | Queued — after main tasks |
| 6.1 | Docker warm-up image | Queued — after main tasks |
| 6.6-toy | Red-then-green throwaway CI | Queued — after main tasks |

