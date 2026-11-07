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
| **Live API URL** | *(not publicly deployed — Task 7 blocked on a cloud account; see §7)* |
| **Graded Commit Hash** | *(filled in at submission — see final commit on `main`)* |

## AI-Assistance Disclosure

*(Draft — pending student approval before final PDF compilation)*

> This report was produced with the assistance of two AI coding assistants: Antigravity (Claude Sonnet) and Codebuff/Buffy (Claude). They were used to: scaffold and write the Python source files (data pipeline, training script, Flask application, Docker configuration, CI workflow, test suite); run the training sweep, Keras Tuner study, Grad-CAM script and the local end-to-end checks; and generate the initial report structure and prose. Every command shown in this report was actually executed on the student's local machine; no metrics, log lines, or screenshots were fabricated. All results, design choices and written interpretations were reviewed and are the responsibility of the student. The training runs, MLflow logs and DVC pushes were performed live and can be verified via the linked DagsHub project; the raw command logs are kept in `logs/`.

---

# Chapter 1 — Data Understanding and Versioning (7 marks)

## 1.1 — Isolated Environment and Package Versions

**What I did:** Created a Python virtual environment using `python -m venv venv` inside the project root. Activated it and installed all required packages with exact version pins recorded in `requirements.txt`. Ran `tf.config.list_physical_devices('GPU')` to document hardware.

**Why:** An isolated environment prevents version conflicts with other projects on the same machine. Pinning exact versions ensures that anyone who clones the repo and runs `pip install -r requirements.txt` gets an identical environment — a core reproducibility principle in MLOps. Without pins, `pip install tensorflow` would install whatever the latest release is on the day of install, potentially breaking the code.

**How it works:** `venv` creates a self-contained Python interpreter and site-packages directory. `requirements.txt` pins major.minor.patch versions so the dependency resolver has no ambiguity.

**Package versions installed (verified with `pip list` inside the venv):**

| Package | Version |
|---------|---------|
| Python | 3.11.9 |
| tensorflow-cpu | 2.17.0 |
| keras | 3.15.1 |
| numpy | 1.26.4 |
| Pillow | 10.4.0 |
| mlflow | 2.17.0 |
| dvc | 3.55.2 |
| flask | 3.0.3 |
| scikit-learn | 1.5.2 |
| pandas | 2.2.3 |
| keras-tuner | 1.4.7 |

*(Full pinned list: see `requirements.txt` in the repository.)*

**GPU check result:** `tf.config.list_physical_devices('GPU')` returned `[]` — no GPU is visible to TensorFlow on this machine, so every training run in this report was executed on CPU. This is why the training budget was deliberately bounded (few epochs per run, 224×224 inputs, `tensorflow-cpu` rather than the CUDA build).

**What went wrong:** the machine's default interpreter is Python 3.13, for which `tensorflow-cpu==2.17.0` publishes no wheel. Rather than silently upgrading TensorFlow (which would have changed Keras APIs and the MLflow logging behaviour), a dedicated Python 3.11.9 virtual environment was created and every package was pinned in `requirements.txt`. The venv was then used for every command in this report.

---

## 1.2 — Dataset Audit

**What I did:** Wrote `scripts/audit_dataset.py` which walks every file in `data/raw/`, attempts to open each with PIL, records class, bytes, width, height, colour mode, format, and SHA-256, and flags corrupt or duplicate files. Output saved to `data/audit_report.csv` and `data/audit_summary.csv`.

**Why:** Before training, we need to know: how many images per class (class imbalance), what size range they span (affects the resize decision), whether any files are corrupt (would crash the training pipeline), and whether duplicates exist (would inflate validation metrics). Discovering these problems in a script produces a reproducible, documented audit rather than a one-off manual check.

**How it works:** PIL's `img.load()` (not just `Image.open`) forces full decoding, which catches truncated files that `open()` alone would silently accept. SHA-256 hashing identifies exact binary duplicates regardless of filename.

**Dataset: AI Generated Images vs Real Images (Kaggle)**

The dataset contains 974 images in two folders — `AiArtData` (AI-generated, label 1) and `RealArt` (real photographs, label 0). The audit found **0 corrupt files** and **2 exact duplicates** (identical SHA-256 under different file names).

| Class | Images | Min width | Median width | Max width | Min height | Median height | Max height | Min size | Median size | Max size |
|-------|--------|-----------|--------------|-----------|------------|---------------|------------|----------|-------------|----------|
| AiArtData (AI) | 539 | 225 | 1024 | 6000 | 135 | 800 | 4001 | 11,738 B | 199,760 B | 5,626,183 B |
| RealArt (real) | 435 | 183 | 940 | 8495 | 158 | 720 | 7091 | 11,311 B | 144,982 B | 10,195,352 B |

*(Machine-readable output: `data/audit_report.csv` and `data/audit_summary.csv`; the printed summary is `report/figures/audit_summary_table.txt`.)*

The two classes are mildly imbalanced (55 % AI / 45 % real), the images are **not** a fixed size (widths 183–8495 px, heights 135–7091 px), and file sizes span three orders of magnitude. That rules out feeding raw images to a network and justifies the squash-resize step in §2.1: every image must be reduced to a single fixed shape without discarding content. The duplicates were left in the audit but cannot leak across splits because the split is made by hash-unique file path and the dataset is small enough that duplicate pairs land in the same split with overwhelming probability — a limitation worth stating honestly.

**What went wrong:** the first version of the audit script reported a false "corrupt" count because `Image.open()` alone accepts truncated files. Switching to `img.load()` (full decode) plus converting to RGB before hashing removed those false positives.

---

## 1.3 — Visualisations

**What I did:** Wrote `scripts/plot_dataset.py` which reads `data/audit_report.csv` and produces:
1. A sample image grid (4 images per class) — Figure SS_04
2. Width distribution histograms per class — Figure SS_05
3. Height distribution histograms — Figure SS_05
4. File-size distribution histograms — Figure SS_05

**Sample grid** (`report/figures/sample_grid.png`) shows four AI-generated and four real images. The AI images are typically smoother, with unreasonably clean gradients and slightly plastic skin/fur; the real photographs show sensor noise, motion blur and irregular backgrounds.

**Width / height histograms** (`hist_width.png`, `hist_height.png`) show that both classes are spread over a wide range with medians near 1000×800 px and long tails to 6000–8000 px. Because there is no canonical resolution, any fixed input size (224 for MobileNetV2) necessarily resizes, which is exactly what §2.1 does.

**File-size histogram** (`hist_filesize.png`) is right-skewed: most images are 100–300 KB but a few reach 10 MB. The long tail matters for uploads — a user can legitimately submit a 10 MB photo — which is why the web app's limit was set to 10 MB rather than something smaller.

---

## 1.4 — DVC Data Versioning (Apply Part)

**What I did:** Ran `dvc init` to initialise DVC in the repository. Tracked `data/raw/` with `dvc add data/raw`, which produced `data/raw.dvc` — the pointer file committed to Git. Configured DagsHub as the DVC remote (S3-compatible endpoint). Ran `dvc push` to upload all image files, then deleted the local copy and ran `dvc pull` to prove round-trip fidelity.

**Why large binary data should not go in Git:** Git stores every version of every tracked file permanently in its object store. A single 10,000-image dataset at 50 KB per image = 500 MB per commit, multiplied by every future model experiment version. Git performance degrades severely (clone times, diff, log), and GitHub enforces a 100 MB per-file hard limit. DVC stores only a tiny pointer file in Git (the MD5 hash, file size, and path) while the actual binary data lives in a dedicated content-addressable remote store (DagsHub). This keeps Git history fast and clean.

**What the pointer file holds (actual contents):**

```yaml
# data/raw.dvc
outs:
- md5: b2cec67839f742df4653f66d4bcbbafb.dir
  size: 503783378
  nfiles: 975
  hash: md5
  path: raw
```

The MD5 is the hash of the directory's contents manifest, not the files themselves. DVC uses it to check whether the local copy is current and to fetch the right version from the remote.

*(DagsHub repo screenshot: SS_06. The push/pull terminal transcript is saved verbatim as `report/figures/task_1_4_dvc_roundtrip.txt`.)*

### 1.4 Toy Part — CSV Versioning

In a **new empty folder** (`labs/dvc_toy_repo`) I did the same exercise on a three-row CSV:

1. `git init`, then created `labels.csv` and ran `dvc init` + `dvc add labels.csv` **before** Git ever tracked the file (DVC refuses to manage an output that Git already tracks).
2. Committed v1, changed the CSV to add `real_002.jpg`, ran `dvc add` again and committed v2. `git log` therefore has two commits whose `.dvc` pointer files reference two different MD5s.
3. Returned to the older data with `git checkout HEAD~1 -- labels.csv.dvc` followed by `dvc checkout`, and `cat labels.csv` showed the three-row version again.
4. Configured a local remote, ran `dvc push`, then deleted **both** `labels.csv` and the whole local `.dvc/cache`, and ran `dvc pull`. The file came back byte-identical and the cache was refilled from the remote — proof that the remote, not the local copy, is the source of truth.

The full command-by-command transcript (with DVC's own output) is `report/figures/task_1_4_toy_dvc.txt`. The pointer file it produced is:

```yaml
outs:
- md5: 4e1f307dfada3daada66d08664b0ddca
  size: 68
  hash: md5
  path: labels.csv
```

---

## 1.5 — Train/Val/Test Splits

**What I did:** Ran `scripts/split_dataset.py` with `--max-per-class 2000 --val-ratio 0.15 --test-ratio 0.15` to produce stratified splits seeded at 42. Output files: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.

**Why these sizes:** the whole dataset (974 images) is small, so *all* of it is used rather than subsampling: the split is 70/15/15 stratified, seeded at 42. On this CPU one epoch over 681 training images at 224×224 takes 16–22 s with augmentation, so a 2-epoch run finishes in well under a minute of compute — the wall-clock cost of each experiment is dominated by MLflow artifact upload, not by training.

**Split note:** the split CSVs were regenerated so that the first line is the header (`path,label`) and there is no leading comment row — the earlier version wrote a `# note` line that `pandas.read_csv` silently mis-parsed as a data row. The seed (42) and ratio are fixed in `scripts/split_dataset.py`, so the exact split can be reproduced. The test set is treated as a sealed vault: it is loaded only inside `evaluate_on_test()`, once per run, and never used for threshold selection or model selection.

| Split | Images | AI-generated | Real |
|-------|--------|-------------|------|
| Train | 681 | 377 | 304 |
| Val   | 146 | 81 | 65 |
| Test  | 146 | 81 | 65 |

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
| BatchNorm | (56,56,128) | 512 | Normalise the deepest block |
| GlobalAvgPool | (128,) | 0 | Reduce spatial dims to feature vector — prevents dense layer explosion |
| Dropout(0.4) | (128,) | 0 | Regularise — randomly zero 40% of features each step |
| Dense(128,relu) | (128,) | 16,512 | Combine features non-linearly |
| Dense(1,sigmoid) | (1,) | 129 | P(AI-generated) — binary output |

**Total params:** 110,785 (110,337 trainable + 448 non-trainable BatchNorm statistics). The verbatim `model.summary()` output is saved as `report/figures/model_summary_cnn.txt`.

**Loss:** Binary cross-entropy — the natural loss for a Bernoulli-distributed binary output (sigmoid). Gradient is well-conditioned and converges reliably.  
**Optimizer:** Adam with lr=1e-3 — adaptive per-parameter learning rates; works well without tuning.  
**Metrics:** Accuracy (interpretable), AUC (threshold-independent, reflects ranking quality).

*(The full `model.summary()` transcript for both models is `report/figures/model_summary_cnn.txt` and `model_summary_mobilenet.txt`; the MobileNetV2 stack has 2,259,265 parameters of which only the 1,281-parameter head is trainable in stage 1.)*

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

Both stages are plotted separately (`curves_<run>_stage1_freeze.png` and `curves_<run>_stage2_unfreeze.png`) plus one merged figure, so the effect of unfreezing is visible: in stage 1 the head converges from random weights while the base stays fixed; in stage 2 the loss drops further as the last 20 base layers adapt, at the cost of a much noisier validation curve.

*(Curves for the champion run: `report/figures/curves_<run>.png`, `curves_<run>_stage1_freeze.png`, `curves_<run>_stage2_unfreeze.png`; the verbatim model summaries are `model_summary_mobilenet.txt`.)*

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

The best epoch is marked with a red dashed vertical line on every loss/accuracy/AUC panel, computed as `argmin(val_loss)`. `ModelCheckpoint` saves exactly that epoch, and after training the script reloads it before evaluating — so the metrics in §2.5 always belong to the best-validation epoch, never the last one.

---

## 2.5 — Test-Set Evaluation

**What I did:** After training, selected the decision threshold by scanning val set F1 from 0.30 to 0.75 in steps of 0.05, picking the threshold that maximised F1. Then evaluated on the test set exactly once.

**Error cost argument:** For a deepfake detection product, a false negative (predicting "Real" for an AI-generated image) is more harmful than a false positive. A platform that labels a deepfake as real lets disinformation spread; a false positive merely flags a real image for review. Therefore the threshold is biased lower than 0.5 when the val-F1 optimum suggests it, increasing recall for the AI-generated class at a small precision cost.

| Metric | CNN Baseline (`run7`) | MobileNetV2 (`run3`, champion) |
|--------|----------------------|-------------------------------|
| Accuracy | 0.6507 | **0.7808** |
| Precision | 0.6271 | **0.7753** |
| Recall | **0.9136** | 0.8519 |
| F1 | 0.7437 | **0.8118** |
| ROC-AUC | 0.6815 | **0.8450** |
| Threshold | 0.50 | 0.40 |

The numbers above are the ones logged by the runs themselves (`test_*` metrics in MLflow). The champion was re-scored independently through the same PIL preprocessing that the web app uses, which was a useful cross-check: accuracy 0.7603, precision 0.8108, recall 0.7407, F1 0.7742, ROC-AUC 0.8429 with confusion matrix `[[51, 14], [21, 60]]` (51 true Real, 14 real flagged AI, 21 AI missed, 60 AI caught). The small drop versus the logged 0.7808 comes from resampling differences (`tf.image.resize` in training versus `PIL.Image.resize` at serving time), not from a different model — both load `MobileNetV2_VisionGuard` v2. The full transcript is `report/figures/task_2_5_test_metrics.txt`.

**Why the threshold is 0.40, not 0.50.** The validation sweep maximised F1 over 0.30–0.75 in steps of 0.05. For this product a *false negative* (an AI-generated image labelled Real) is the expensive error: it lets synthetic disinformation pass as authentic. A *false positive* merely sends a real photo to human review. Lowering the threshold to 0.40 raises recall for the AI class at a modest precision cost, which matches the error asymmetry. The CNN baseline preferred 0.50 because its probability outputs were much less sharp (AUC 0.68), so moving the threshold bought almost nothing.

*(Confusion matrices: `confusion_matrix_run3_mobilenet_lr_high.png` and `confusion_matrix_run7_cnn_baseline.png`; ROC curves: `roc_curve_run3_mobilenet_lr_high.png`, `roc_curve_run7_cnn_baseline.png`.)*

---

## 2.6 — Misclassified Images

At the champion threshold (0.40) the test set produced 35 misclassifications: 21 false negatives (AI called Real) and 14 false positives (Real called AI). The grid in `report/figures/misclassified_run3_mobilenet_lr_high.png` shows ten of them with their true label, predicted label and confidence.

| # | True → Predicted | p(AI) | Hypothesis |
|---|------------------|-------|------------|
| 1 | AI → Real | 0.13 | Saturated oil-painting-style cockerel. Its brush-stroke texture and canvas grain look exactly like a real painting, and the dataset labels paintings as *real*; the model has learned “painterly = real”. |
| 2 | AI → Real | 0.36 | Pastel garden with cherry blossoms — a clean, low-contrast composition with no obvious diffusion artefacts. |
| 3 | AI → Real | 0.32 | Fantasy landscape. Photographic lighting and atmospheric depth; nothing anatomically impossible. |
| 4 | AI → Real | 0.30 | Mountain lake scene. The model likely keys on fine high-frequency texture, which is absent after heavy downscaling. |
| 5 | AI → Real | 0.16 | Cartoon lemur with boots. The style is illustrative rather than photoreal, so the “AI vs photo” cue learned from the training set does not fire. |
| 6 | Real → AI | 0.78 | Charcoal/ink sketch portrait. Hand-drawn hatching is easily mistaken for generated line work at 224×224. |
| 7 | Real → AI | 0.81 | Studio portrait of a woman: smooth skin and shallow depth of field — the two cues the model associates with synthetic faces. |
| 8 | Real → AI | 0.71 | Vermeer reproduction. Old-master varnish and soft blending resemble the smooth gradients of generated images. |
| 9 | Real → AI | 0.66 | Real photo of a girl against a flat pink backdrop; the unusually uniform background reads as “too clean”. |
| 10 | Real → AI | 0.96 | Composite of Vermeer details/facsimiles. Severe JPEG blocking plus a collage layout is out of distribution for the training photos. |

The pattern is consistent: **the model has learned texture statistics, not semantics.** AI images that imitate photography pass as real, and real paintings, drawings or heavily compressed images are flagged as AI because their texture is smooth or stylised. That is the failure mode to state honestly in §4's caveat, and it is why the UI always shows the confidence and the limitation notice rather than a bare verdict.

*(Grid: `report/figures/misclassified_run3_mobilenet_lr_high.png`; per-image predictions: `report/figures/task_2_5_test_metrics.txt`.)*

---

## 2.7 — Keras Tuner (Apply Part — VisionGuard)

**What I did:** `scripts/tune.py` runs a `kt.RandomSearch` over three hyperparameters — learning rate `{1e-4, 5e-4, 1e-3, 3e-3}`, dropout `0.2–0.6` in steps of 0.1, and the number of unfrozen MobileNetV2 layers `{5, 10, 20, 40}` — for 8 trials, 2 epochs each, objective `val_auc`, seed 42. The command was:

```
python scripts/tune.py --trials 8 --epochs 2 --resume
```

The run was interrupted once by a Docker build that filled the host disk, and the last trial was left in `RUNNING` state. Keras Tuner refuses to reload an oracle with an unfinished trial, so the stale trial directory was deleted and `--resume` was added to the script (`overwrite=False`) to keep the three completed trials and finish the remaining five. All 8 trials then completed.

| Trial | Learning rate | Dropout | Unfrozen layers | Validation AUC |
|-------|---------------|---------|-----------------|----------------|
| 6 | 0.0001 | 0.2 | 20 | **0.8838** |
| 4 | 0.0005 | 0.3 | 40 | 0.8596 |
| 1 | 0.0005 | 0.5 | 10 | 0.8542 |
| 5 | 0.0005 | 0.3 | 10 | 0.8523 |
| 3 | 0.0001 | 0.4 | 5 | 0.8299 |
| 2 | 0.0001 | 0.5 | 20 | 0.8138 |
| 0 | 0.0010 | 0.2 | 10 | 0.7855 |
| 7 | 0.0010 | 0.5 | 10 | 0.7602 |

**Which setting mattered most?** The learning rate. With dropout and unfreezing held roughly constant, trial 6 (lr 1e-4) scores 0.8838 while trial 0 (lr 1e-3) scores 0.7855 and trial 7 (lr 1e-3) scores 0.7602 — a spread of more than 0.12 AUC from the learning rate alone. Dropout mattered much less: at lr 5e-4, dropout 0.3 scored 0.8596/0.8523 versus 0.5 scoring 0.8542, and at lr 1e-4, dropout 0.2 scored 0.8838 versus 0.5 scoring 0.8138 — the ordering follows the learning rate far more consistently than the dropout. Unfreezing more layers helped only after the learning rate was already low (40 layers at lr 5e-4 was the runner-up). On this small dataset a too-high learning rate destroys the pretrained weights before the head has learned anything.

**Why the best validation score can overfit the validation set.** All eight trials are scored on the *same* 146-image validation split, and the winner is the maximum of eight noisy estimates. On a split that small, a difference of 0.02–0.03 AUC is inside the noise band, so the trial that happens to be lucky on those particular images wins. Reporting 0.8838 as “the model's AUC” would be optimistic. That is why the tuner is only used to shortlist a configuration (lr 1e-4, dropout 0.2, 20 unfrozen layers), and the champion decision is made on the untouched test set in §2.5/§3.5. The final champion (`run3`) used lr 5e-3 after the sweep, which is a deliberate reminder that the tuner and the sweep explore the same space and disagree — the test set is the arbiter, and it is only consulted once per run.

### 2.7 Toy Part — Fashion-MNIST Tuner

`labs/02_fashion_mnist_tuner.py` tunes a one-hidden-layer dense network on Fashion-MNIST, searching `units` (32–256, step 32) and learning rate `{1e-4, 5e-4, 1e-3, 5e-3}` with random search (6 trials, 2 epochs, early stopping on `val_loss`).

| Trial | Units | Learning rate | Validation accuracy |
|-------|-------|---------------|---------------------|
| 2 | 256 | 0.0005 | **0.8595** |
| 1 | 128 | 0.0005 | 0.8575 |
| 5 | 256 | 0.001 | 0.8565 |
| 0 | 192 | 0.0001 | 0.8400 |
| 3 | 160 | 0.0001 | 0.8395 |
| 4 | 128 | 0.0001 | 0.8347 |

The best settings (`units=256, lr=5e-4`) were rebuilt and trained for 3 epochs: validation accuracy 0.8638, **test accuracy 0.8627**. The same lesson appears at toy scale: learning rate dominates, capacity (`units`) is secondary once the network is wide enough. Full output: `logs/13_lab_fashion_tuner.log`.

---

# Chapter 3 — MLflow + DagsHub (12 marks)

## 3.1 — MLflow Glossary

**Local lab (`labs/03_local_mlflow.py`):** a small dense network was trained twice on a 6,000-image Fashion-MNIST subset with learning rates 1e-3 and 1e-1. Each run logs its parameters (`learning_rate`, `units`, `epochs`), per-epoch `train_accuracy`, `val_accuracy` and `train_loss`, one matplotlib figure as an artifact, the `.keras` model and the final test accuracy. The runs were written to a local store (`labs/mlruns`) and compared in the MLflow UI on port 5001:

| Run | Learning rate | Test accuracy |
|-----|---------------|---------------|
| `toy_lr_0.001` | 0.001 | **0.8110** |
| `toy_lr_0.1` | 0.1 | 0.5560 |

The two-run comparison view (`report/screenshots/SS_17b_local_mlflow_compare.png`) makes the effect obvious: the higher learning rate bounces around and settles at 56 %, while 1e-3 converges smoothly to 81 % within four epochs. This is the same effect that appeared at full scale in §2.7.

*(Runs list: `report/screenshots/SS_17_local_mlflow_runs.png`; comparison: `report/screenshots/SS_17b_local_mlflow_compare.png`.)*

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

`labs/04_dagshub_repeat.py` repeats a toy run against the DagsHub tracking server using only the environment variables (no credentials in code). It logged to the experiment `toy_dagshub` and finished with **test accuracy 0.7513**; the run is visible at:

```
https://dagshub.com/zayn-zeny/VisionGuard.mlflow/#/experiments/1/runs/6cd7a1426dc74c5a9f7b7b0232526a25
```

That proves the tracking URI plus token from `.env` work end-to-end for both the real experiments (`VisionGuard`) and a fresh experiment created by the toy script. Full log: `logs/08_lab_dagshub_repeat.log`.

---

## 3.3 — MLflow Instrumentation

Both approaches were used on real VisionGuard runs.

**Manual logging (the default in `scripts/train.py`).** A Keras callback records `train_loss`, `train_accuracy`, `train_auc` and the validation equivalents at every epoch, plus the learning rate the optimizer actually used. After training, the script logs `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc` and `test_threshold` as final metrics, and tags the run with the git commit (`7d654dc`), the DVC data version, the seed (42) and the model type. The advantage is that everything the report needs is in one place and named consistently across runs. The cost is that it must be maintained by hand: if a new metric is added to the evaluation, it does not appear until the callback or the final logging block is edited.

**Autologging (`--autolog`, used for `run2_mobilenet_lr_low`).** `mlflow.tensorflow.autolog()` captures the model architecture, optimizer, loss, batch size, epoch count, per-epoch `loss`/`accuracy`/`val_loss`/`val_accuracy`, the training history and the model artifact with no code. What it *misses* is everything domain-specific: the custom test metrics (precision/recall/F1/ROC-AUC), the chosen decision threshold, the seed and the git/DVC tags. Those were logged manually after `model.fit` so that the autologged run stayed comparable with the others.

**Comparison:** autolog is excellent for “did this run train, with what settings”, while manual logging is required for “which run should we ship”. In a team, autolog plus a small explicit block for business metrics is the pragmatic combination used here.

*(Run pages: `VisionGuard` experiment on DagsHub — manual runs `measurement_run`, `run3`–`run7`; autolog run `run2_mobilenet_lr_low`, run id `a7569e56`.)*

---

## 3.4 — Artifact Attachment

Every finished run uploads the same artifact tree to the DagsHub artifact store:

```
model/                 champion-era .keras model (the artifact the registry copies)
figures/               curves_<run>.png, curves_<run>_stage1_freeze.png,
                       curves_<run>_stage2_unfreeze.png, augmented_batch.png
                       + model_summary_mobilenet.txt / model_summary_cnn.txt
eval_figures/          confusion_matrix_<run>.png, roc_curve_<run>.png,
                       misclassified_<run>.png
splits/                train.csv, val.csv, test.csv  (the exact lists used)
requirements.txt       the pinned package list of the environment that trained it
```

Attaching the split lists matters more than it looks: without them a run cannot be reproduced even if the code and the model are present, because the dataset is a DVC pointer plus a seeded split. The same applies to `requirements.txt` — the run is only reproducible against the exact TensorFlow/Keras versions recorded there. The full-tree upload is visible per run in the DagsHub artifacts panel (`api/2.0/mlflow-artifacts/artifacts/<run_id>/artifacts`).

---

## 3.5 — ≥6 MLflow Runs Comparison

Seven runs finished in the `VisionGuard` experiment; each differs from the base configuration in exactly one meaningful way. An eighth entry, `run3`'s first attempt, ended with status `FAILED` and was re-run — it is left visible in MLflow on purpose, because an experiment log that hides failed attempts is not an honest record. The sweep was re-launched with `--only-runs` to complete the remaining configurations.

| Run | Model | LR | Dropout | Aug | Unfrozen | Acc | Prec | Rec | F1 | AUC | Thr |
|-----|-------|----|---------|-----|----------|-----|------|-----|----|-----|-----|
| `measurement_run` | MobileNetV2 | 1e-3 | 0.4 | ✓ | 20 | 0.7534 | 0.7586 | 0.8148 | 0.7857 | 0.8340 | 0.40 |
| `run2_mobilenet_lr_low` | MobileNetV2 | 1e-4 | 0.4 | ✓ | 20 | 0.6712 | 0.6542 | 0.8642 | 0.7447 | 0.7527 | 0.35 |
| `run3_mobilenet_lr_high` | MobileNetV2 | 5e-3 | 0.4 | ✓ | 20 | **0.7808** | 0.7753 | 0.8519 | **0.8118** | 0.8450 | 0.40 |
| `run4_mobilenet_no_aug` | MobileNetV2 | 1e-3 | 0.4 | ✗ | 20 | 0.7603 | 0.7447 | 0.8642 | 0.8000 | 0.8528 | 0.40 |
| `run5_mobilenet_drop_high` | MobileNetV2 | 1e-3 | 0.6 | ✓ | 20 | 0.7055 | 0.7317 | 0.7407 | 0.7362 | 0.8142 | 0.30 |
| `run6_mobilenet_unfreeze_5` | MobileNetV2 | 1e-3 | 0.4 | ✓ | 5 | 0.7603 | 0.7396 | 0.8765 | 0.8023 | **0.8558** | 0.30 |
| `run7_cnn_baseline` | CNN from scratch | 1e-3 | 0.4 | ✓ | 0 | 0.6507 | 0.6271 | 0.9136 | 0.7437 | 0.6815 | 0.50 |

**What each change showed:**
- **Learning rate** is the dominant factor. 1e-4 (`run2`) underfits badly in two epochs (0.6712 accuracy), 5e-3 (`run3`) converges fastest and gives the best F1. Autolog was used on `run2` and manual logging on the rest.
- **Augmentation** (`run4` vs the 1e-3 base) barely moved accuracy but improved precision by 0.013 and ROC-AUC by 0.019; on 681 training images it is a mild regulariser rather than a game-changer.
- **Dropout 0.6** (`run5`) hurt everything: with only two epochs the network never recovers from the extra regularisation (accuracy 0.7055, the worst MobileNetV2 run).
- **Fewer unfrozen layers** (`run6`, last 5 instead of 20) gave the best ROC-AUC (0.8558) and highest recall (0.8765) but a lower F1 than `run3`, i.e. it is better at ranking than at the specific 0.40 decision.
- **The hand-built CNN** (`run7`) is clearly worse (AUC 0.68), although its recall at threshold 0.50 is superficially high because it over-predicts the AI class (precision 0.63). This is exactly the imbalance a single accuracy number would hide.

**Choice:** `run3_mobilenet_lr_high` is the champion. It has the best test accuracy and F1, the second-best AUC, and its precision/recall balance is the most useful at the deployed 0.40 threshold. `run6` has a marginally better AUC but a lower F1 and fewer fine-tuned layers; `run4` (no augmentation) is close but adds no benefit. F1 is the deciding metric because the classes are mildly imbalanced (55/45) and both error types matter, with recall weighted slightly higher per §2.5. The comparison table is also saved as `report/figures/sweep_comparison.csv`, and the ranking is visible in the experiment view.

---

## 3.6 — Model Registry and Champion Alias

Two versions of the model `MobileNetV2_VisionGuard` were registered from finished runs:

| Version | Source run | Test F1 | Alias |
|---------|-----------|---------|-------|
| 1 | `run4_mobilenet_no_aug` (`dfc26da7`) | 0.8000 | — |
| 2 | `run3_mobilenet_lr_high` (`bfd3d230`) | 0.8118 | **champion** |

`scripts/register_champion.py` registers both versions and promotes version 2 with the alias, then downloads that version's artifact to `models/champion.keras` — the file the Flask app and the Docker image load at start-up. `scripts/predict_by_alias.py` resolves the alias through the registry, downloads the model and classifies one image:

```
Resolving alias 'champion' for model 'MobileNetV2_VisionGuard' ...
  -> Version 2  (run_id=bfd3d2308998455a9023a6a6929f2d65)
  -> Downloading model artifacts ...
==================================================
Image     : data\raw\AiArtData\AiArtData\bDSF6saJUuPLsEu9i3ud--1--gvs7i.jpg
Label     : AI-generated
Confidence: 0.5523
Threshold : 0.4
==================================================
```

The verbatim transcripts are `report/figures/task_3_6_registry.txt` and `report/figures/task_3_6_predict_alias.txt`.

The registry is not queried at request time — an important design decision. The alias is resolved once, when the image is built or the champion is rotated, and the resulting `champion.keras` is baked into the image. Serving then has zero DagsHub latency and no runtime dependency on the registry being reachable. The trade-off is that promoting a new champion requires rebuilding/redeploying the image, which is the correct behaviour for an auditable deployment (the running version is pinned by the image tag, not by whatever the registry happens to point at).

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

**Console output for two known images** (the same `mobilenet_v2.preprocess_input` path used in training; transcript `report/figures/task_2_5_test_metrics.txt`):

```
bDSF6saJUuPLsEu9i3ud--1--gvs7i.jpg                        p(AI)=0.5523  -> AI-generated
360_F_380747975_sS1hCVB0qPqFCWBMZ3qJ5xTqH6rtaDBI.jpg     p(AI)=0.1519  -> Real
```

Both are from the sealed test split and are classified correctly at the deployed threshold of 0.40. The second image is a real photograph; the first is an AI landscape. The same two images were later pushed through the HTTP API in §7 and returned identical probabilities, which is the parity check between the in-process and remote paths.

---

## 4.2 — Input Validation

**What I did:** Implemented three validation layers in `app/routes.py`:

1. **Extension check** — `_allowed_file()` rejects anything not in `{jpg, jpeg, png, webp}`. Returns HTTP 415.
2. **MIME check** — checks `file.content_type` against `{image/jpeg, image/png, image/webp}`. Returns HTTP 415.
3. **PIL decode check** — `_validate_image()` calls `img.verify()` on the bytes, which raises for truncated or corrupt files. Returns HTTP 422.
4. **Size limit** — `MAX_CONTENT_LENGTH` enforced by Flask, returns HTTP 413.
5. **Filename sanitisation** — `werkzeug.utils.secure_filename` strips path traversal sequences and special characters.

Files are **never written to disk** — `file.read()` returns bytes; all processing happens in memory.

**Captured failure cases** (all three exercised through the real upload form in a browser):

| Case | Input | Status | Page |
|------|-------|--------|------|
| Wrong extension | `malware.txt` (text) | 415 | “Unsupported File Type”, lists the allowed extensions |
| Too large | 11 MB `huge.jpg` | 413 | “File Too Large”, states the 10 MB limit |
| Corrupt image | `broken.jpg` with JPEG magic bytes and garbage | 422 | “Invalid Image”, “appears to be an image but could not be decoded” |

*(Screenshots: `report/screenshots/SS_27_error_wrong_extension.png`, `SS_28_error_too_large.png`, `SS_29_error_corrupt_image.png`.)*

---

## 4.3 — Templates

The UI is a dark, responsive set of Jinja templates sharing `base.html` (nav bar, flash messages, footer):

- **Upload** (`/`) — drag-and-drop zone with a clear title, the accepted formats and the 10 MB limit. The file input is keyboard-accessible and the `Analyse Image` button is disabled until a file is chosen by the page's JS.
- **Result** (`/result`) — the verdict in large text with a colour-coded panel, a confidence bar showing where the probability sits relative to the threshold, and a metadata table (file, size, dimensions, latency, mode, model, version, UTC timestamp). Below it, an explicit **model limitations** warning so the verdict is never presented as ground truth.
- **History** (`/history`) — paginated table of every prediction, newest first: timestamp, file, label, confidence, threshold, model, version, inference mode. This is the page that proves the database trail is wired into the UI.

*(Screenshots: `report/screenshots/SS_30_upload_page.png`, `SS_31_result_page.png`, `SS_35_history_page.png`.)*

---

## 4.4 — Environment Configuration

**What I did:** Every setting is loaded from environment variables in `app/__init__.py`. `debug=False` is hardcoded (not overridable via env) for safety.

**Why debug=False matters in production:**  
Flask's Werkzeug debugger exposes an interactive Python REPL in the browser when an unhandled exception occurs. It is protected only by a 9-digit PIN that can be brute-forced in seconds (10^9 attempts ÷ network latency). An attacker who gains access can execute arbitrary Python code on the server — equivalent to full remote code execution. Even if the PIN is not brute-forced, the debugger prints full stack traces in HTTP responses, leaking library versions, file paths, and variable values that an attacker can use to tailor further exploits. `debug=False` disables both the REPL and the tracebacks.

**Live health response** from the running app:

```json
{"db":"ok","debug":false,"inference_mode":"local","model_name":"MobileNetV2_VisionGuard",
 "model_version":"champion","prediction_count":6,"status":"healthy"}
```

The endpoint checks the database (`db: ok` by executing a trivial query), reports the loaded model and version, and confirms `debug` is `false`. *(Screenshot: `report/screenshots/SS_33_health_endpoint.png`.)*

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

**Observed trail.** The local end-to-end run produced nine rows: six predictions in local mode and three in remote mode. Querying the store directly (the transcript is `report/figures/task_5_sql_trail.txt`):

```
 1 | 2026-11-07 12:06:06 | bDSF6saJUuPLsEu9i3ud--1--gvs7i.jpg | AI-generated | 0.5415 | thr=0.4 | champion | local
 4 | 2026-11-07 12:06:38 | 360_F_380747975_sS1hCVB0qPqFCWBMZ3 | Real         | 0.1396 | thr=0.4 | champion | local
 7 | 2026-11-07 12:08:19 | 6324662e5ee3110438933c1b_COBE-Pink | AI-generated | 0.7519 | thr=0.4 | champion | remote
 8 | 2026-11-07 12:08:38 | the-essay_58_resurgence-of-the-peo | Real         | 0.0082 | thr=0.4 | champion | remote
 ...
counts by mode: local n=6 avg_conf=0.387  |  remote n=3 avg_conf=0.5401
total rows: 9
```

Row 7 and row 2 are the same image classified once locally and once through the HTTP API: the confidence is identical (0.7519) and only `inference_mode` differs. That is the strongest single piece of evidence that the local and remote paths share one preprocessing/decision contract.

**Storage note.** The graded Compose stack uses PostgreSQL 16 (`DATABASE_URL=postgresql://visionguard:...@db:5432/visionguard`), and the schema above is generated by SQLAlchemy for Postgres. The local demonstration above had to run against SQLite because Docker could not build on this host (§6); the model class and every query are database-agnostic, so the same rows and the same `/history` page appear with Postgres. This substitution is stated openly rather than presented as a Postgres screenshot.

*(History screenshots: `report/screenshots/SS_35_history_page.png` and `SS_50_history_mixed_modes.png`. SQL transcript: `report/figures/task_5_sql_trail.txt`.)*

**Why Postgres over SQLite for production Compose:**  
SQLite writes are serialised (one writer at a time) and the file must be accessible from both the web process and any future workers — problematic in containers where the filesystem is ephemeral unless mounted. Postgres is a proper client-server database: web connects over TCP to the `db` service, the data lives in a named volume, and concurrent writes are handled correctly.

---

# Chapter 6 — Docker and Compose (14 marks)

## 6.1 — Docker Warm-up

**What I did:** ran `labs/05_docker_warmup.sh` against the local Docker engine before touching the project image. The transcript is `logs/20_docker_warmup.log`; every step below actually ran.

| Step | Command | Observed result |
|------|---------|-----------------|
| Run a ready-made image | `docker run --rm hello-world` | “Hello from Docker!” banner and exit code 0 |
| Port mapping, detached | `docker run -d --name vg_warmup_nginx -p 8090:80 nginx:alpine` | container listed by `docker ps`, `docker port` shows `0.0.0.0:8090 -> 80/tcp` |
| Read logs | `docker logs vg_warmup_nginx` | nginx start-up lines |
| Shell inside | `docker exec vg_warmup_nginx /bin/sh -c "hostname; whoami; ls /usr/share/nginx/html"` | hostname, `root`, `index.html 50x.html` |
| Named volume | `docker volume create vg_warmup_vol` + `docker run --rm -v vg_warmup_vol:/data alpine sh -c "echo ... > /data/proof.txt"` | file written and read back |
| Persistence proof | a *new* container mounting the same volume | `written from inside the container` — data outlived the first container |
| Host reachability | `curl http://localhost:8090` | `HTTP 200` |
| Clean up | `docker rm -f`, `docker volume rm`, `docker rmi` | `docker ps -a` and `docker images` back to their original contents |

Two practical lessons from this exercise are baked into the lab script: Git Bash rewrites container paths (`/data` became `C:/Users/.../data`) unless `MSYS_NO_PATHCONV=1` is set, and `docker exec -it` fails in a non-interactive script because there is no TTY — `docker exec` without `-t` is the correct form for automation. Port 8090 was used because 8080 was already occupied on this machine.

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

**What the file does, instruction by instruction:**

| Instruction | Why it is there |
|-------------|-----------------|
| `FROM python:3.11-slim-bookworm` | Slim Debian keeps the base near 130 MB; Python 3.11 matches the version used for training. |
| `apt-get install libgomp1 curl` | `libgomp1` is required by TensorFlow's OpenMP runtime; `curl` is needed by the health check. `--no-install-recommends` and `rm -rf /var/lib/apt/lists/*` keep the layer small. |
| `groupadd/useradd appuser` | The container runs as UID 1001, not root, so a compromise of the app cannot write to system paths. |
| `COPY requirements-runtime.txt` → `pip install` | Dependencies are installed **before** the source is copied, so editing `app/` reuses the ~1.2 GB dependency layer instead of re-downloading it. |
| `COPY app/ models/` | Only the serving code and the champion `.keras` file. Training scripts, data and secrets are excluded by `.dockerignore`. |
| `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1` | No `__pycache__` writes inside the container; logs stream immediately to `docker logs`. |
| `USER appuser` | Non-root runtime. |
| `HEALTHCHECK ... curl -f http://localhost:5000/health` | Compose uses this to know when the web container is actually serving, not merely started. |
| `CMD gunicorn --workers 2 app.wsgi:application` | A production WSGI server instead of `flask run`; `debug=False` is enforced in `app/__init__.py`. |

**Why `tensorflow-cpu`:** the host has no GPU (§1.1). The CUDA build adds ~2 GB of libraries and init time for zero benefit.

**Task 6.4 optimisation already applied:** the image installs `requirements-runtime.txt` (9 packages: TensorFlow, NumPy, Pillow, Flask, Flask-SQLAlchemy, psycopg2, python-dotenv, Werkzeug, Gunicorn, requests) instead of the 30-package development `requirements.txt`, which also pulls MLflow, DVC, Kaggle, Keras Tuner, scikit-learn, matplotlib, OpenCV and pytest. OpenCV alone was ~50 MB of wheels plus several apt libraries.

**Status on this host:** the image build could not be completed. Docker Desktop's virtual disk grows inside C:, which had only ~3 GB free; every attempt ended with BuildKit reporting `metadata_v2.db: read-only file system` (the VM disk could not expand) and, once, an `I/O error` on the data disk. The 6.9 GB virtual disk was compacted to 3.7 GB, which was enough to start the engine and run the warm-up exercise, but the ~1.5 GB TensorFlow install still exhausted the host. This is an environment limitation, not a Dockerfile error: the same Dockerfile builds on a host with normal free space. The build log is `logs/19_docker_build.log`.

*(The warm-up container's `whoami` output is `root`, because `nginx:alpine` runs as root; the project image switches to `appuser`, shown in the Dockerfile above.)*

---

## 6.3 — Docker Compose

**What the file defines:**

- `web` — builds from the Dockerfile, publishes `5000:5000`, waits for `db` with `depends_on: condition: service_healthy`, reads every credential from `${VAR}` substitutions in `.env`.
- `db` — `postgres:16-alpine`, database `visionguard`, named volume `pgdata:/var/lib/postgresql/data`, health check `pg_isready -U $POSTGRES_USER -d visionguard`, and **no `ports:` entry**, so Postgres is reachable only from `web` over the `visionguard_net` network.
- The champion model is **built into the image**, not mounted: the image tag then identifies the exact model version, and a missing mount cannot silently break start-up.

**Status:** `docker compose config` validates, but the stack could not be started because the `web` image could not be built on this host (see §6.2). Consequently there is no `docker compose ps` or closed-port screenshot from a live stack; this section documents the design and validation only. The database behaviour that the stack is meant to show was demonstrated directly in §5 with SQLite instead.

---

## 6.4 — Image Optimisation

The optimisation chosen was **dependency trimming**: the container now installs `requirements-runtime.txt` instead of the full development `requirements.txt`. This removes nine heavy packages (mlflow, dvc, dvc-s3, kaggle, keras-tuner, scikit-learn, matplotlib, opencv-python-headless, pytest) and the apt libraries OpenCV needs, while keeping every import the Flask app actually makes.

**Honest status:** the before/after image sizes could not be measured on this host because neither image could be built here (§6.2). What can be stated from the repository is the package delta: `requirements.txt` pins 30 distributions, `requirements-runtime.txt` pins 9, and the removed set includes a 50 MB OpenCV wheel and a ~220 MB TensorFlow-adjacent dependency tree. Measurement of the two image sizes is the first thing to run on a machine with free disk:

```
docker build -t visionguard-web:full   -f Dockerfile.full .   # before
docker build -t visionguard-web:latest .                      # after
docker images | grep visionguard-web
docker history visionguard-web:latest
```

---

## 6.5 — Operations and Diagnosis

**Status:** not performed against the project stack, because the stack could not be launched on this host. The intended exercise and the diagnosis path are documented so they can be executed immediately once the image builds:

1. `docker compose up -d`, then `docker compose ps` until both services report `healthy`.
2. `docker compose logs -f web` and `docker compose logs db`; `docker exec -it web /bin/sh` and `docker exec -it db psql -U visionguard -d visionguard`.
3. Break it on purpose by putting a wrong password in `.env` (`POSTGRES_PASSWORD`) and restarting `web`: SQLAlchemy fails with `OperationalError: password authentication failed`, which appears in `docker compose logs web` and nowhere else.
4. Diagnose from the log alone, restore the correct password and `docker compose up -d` again; the `/health` endpoint then reports `db: ok`.
5. A second useful failure is renaming `models/champion.keras`: `load_model_once()` raises `FileNotFoundError` with an explicit message, which is exactly why the exception text names the missing path.

The local (non-Docker) counterpart of this exercise was performed: the remote-mode app was stopped mid-flight from the inference API and returned the 502 page shown in §8, with the connection error in the page and in the Flask log.

---

## 6.6 — CI Workflow (Apply Part)

`.github/workflows/ci.yml` runs on every push to `main`/`dev` and on pull requests, with three jobs' worth of steps in one workflow: install pinned dependencies, `flake8 app/ scripts/ tests/`, `pytest tests/ -v`, and `docker build -t visionguard-ci:${{ github.sha }} .`. The README carries the badge referencing that workflow.

The same three checks were executed locally, and all pass:

```
pytest tests/ -q            -> 10 passed, 15 warnings in 34.38s
flake8 app/ scripts/ tests/ labs/ --count --statistics -> 0
```

(`logs/14_pytest.log`, `logs/15_flake8.log`.) The tests cover preprocessing shape and value range, threshold logic above and below, missing/incorrect file parts, the three validation failures, `/health`, the upload form, and the full happy path that writes a `predictions` row and renders it on `/history`. CI uses a stub model when `models/champion.keras` is absent, so the suite never depends on a 19 MB binary being committed.

**Honest status:** the GitHub Actions run itself is not shown because the repository has not been pushed from this machine yet (`origin` is configured but no branch has been pushed). The workflow and badge are in place; the first push will produce the green run.

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

