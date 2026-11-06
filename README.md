# 23L-0578_A2_Ops — VisionGuard

[![CI](https://github.com/zayn-zeny/23L-0578_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/zayn-zeny/23L-0578_A2_Ops/actions/workflows/ci.yml)

A web service that classifies an uploaded image as **AI-generated** or **real**, built as Assignment 2 for the MLOps course.

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   Flask Web App      │──────▶│  Cloud Inference API      │
│   (localhost:5000)   │  HTTP  │  (public internet)        │
│                      │◀──────│  POST /predict            │
└─────────────────────┘        └──────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   PostgreSQL DB      │
│   (prediction log)   │
└─────────────────────┘
```

The inference layer is **swappable** via `INFERENCE_MODE=local|remote` in `.env`.

## Quick Start

```bash
cp .env.example .env        # fill in your values
docker compose up --build   # starts web + db
# open http://localhost:5000
```

## Project Layout

```
23L-0578_A2_Ops/
├── app/                  # Flask application
│   ├── __init__.py
│   ├── routes.py
│   ├── inference.py      # local + remote inference adapters
│   ├── database.py
│   ├── templates/
│   └── static/
├── scripts/              # standalone utility scripts
│   ├── audit_dataset.py
│   ├── split_dataset.py
│   ├── predict_by_alias.py
│   └── train.py
├── tests/                # pytest suite
├── report/               # report source + figures
├── models/               # .keras file (not in Git — DVC)
├── data/                 # dataset (not in Git — DVC)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .dvc/
```

## Credentials & Secrets

All secrets live in `.env` (git-ignored). See `.env.example` for the full key list. GitHub Actions secrets live in repo Settings → Secrets.

## DagsHub

Project: [https://dagshub.com/zayn-zeny/23L-0578_A2_Ops](https://dagshub.com/zayn-zeny/23L-0578_A2_Ops)  
MLflow runs and DVC-tracked data are stored there.

## Live API

URL: *(filled in after Task 7 deployment)*

## Report

`report/23L-0578_VisionGuard.pdf` — compiled from `report/report.md`.

