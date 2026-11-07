"""
scripts/deploy_hf_space.py — Task 7: deploy the cloud inference API to a
Hugging Face Docker Space.

Reads credentials from environment variables only (.env is git-ignored):
    HF_TOKEN        write-scoped Hugging Face access token
    HF_SPACE_ID     e.g. "<hf-username>/23L-0578_A2_Ops"   (optional)
    REMOTE_API_KEY  shared secret the Flask app sends as X-API-Key

What it does:
    1. creates the Space (sdk=docker, public) if it does not exist;
    2. stores REMOTE_API_KEY as a Space *secret* (never in the repo);
    3. uploads cloud_api/{Dockerfile,main.py,requirements.txt}, a Space
       README with the docker front-matter, and models/champion.keras.
Hugging Face then builds the image and serves it at
    https://<user>-<space-name>.hf.space   (lower-case, "_" -> "-")
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "23L-0578_A2_Ops"

README = """---
title: VisionGuard 23L-0578_A2_Ops
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# VisionGuard inference API (23L-0578_A2_Ops)

`POST /predict` with multipart field `image` and header `X-API-Key`.
Returns `{"label", "confidence", "threshold", "model", "sha256"}`.
`GET /health` reports whether the champion model is loaded.
"""


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ["HF_TOKEN"]
    api_key = os.environ["REMOTE_API_KEY"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    space_id = os.environ.get("HF_SPACE_ID") or f"{user}/{SPACE_NAME}"

    api.create_repo(space_id, repo_type="space", space_sdk="docker",
                    private=False, exist_ok=True)
    api.add_space_secret(space_id, "REMOTE_API_KEY", api_key)
    api.add_space_variable(space_id, "THRESHOLD", os.environ.get("THRESHOLD", "0.4"))

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("Dockerfile", "main.py", "requirements.txt"):
            shutil.copy2(ROOT / "cloud_api" / name, stage / name)
        (stage / "models").mkdir()
        shutil.copy2(ROOT / "models" / "champion.keras",
                     stage / "models" / "champion.keras")
        (stage / "README.md").write_text(README, encoding="utf-8")
        api.upload_folder(folder_path=str(stage), repo_id=space_id,
                          repo_type="space",
                          commit_message="Deploy VisionGuard inference API")

    host = space_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    print(f"Space page : https://huggingface.co/spaces/{space_id}")
    print(f"API URL    : https://{host}.hf.space/predict")


if __name__ == "__main__":
    main()
