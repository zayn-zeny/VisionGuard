#!/usr/bin/env bash
# labs/01_dvc_csv_toy.sh — Task 1.4 Toy Part
# ==========================================
# In a NEW empty folder: init Git + DVC, track a small CSV, change it and
# commit both versions, go back to the older version, configure a remote,
# push, delete the local copy AND the local cache, then pull it back.
#
# Run from the project root:
#   bash labs/01_dvc_csv_toy.sh
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOY="$ROOT/labs/dvc_toy_repo"
REMOTE="$ROOT/labs/dvc_toy_remote"
DVC="$ROOT/venv/Scripts/dvc.exe"

rm -rf "$TOY" "$REMOTE"
mkdir -p "$TOY" "$REMOTE"
cd "$TOY"

git init -q
git config user.email "23L-0578@student.example"
git config user.name "Zain Ul Abideen (23L-0578)"

# ── v1: create the CSV and put it under DVC BEFORE Git sees it ────────
printf 'image,label\nai_001.jpg,1\nai_002.jpg,1\nreal_001.jpg,0\n' > labels.csv
"$DVC" init -q
"$DVC" add labels.csv
git add .dvc .dvcignore labels.csv.dvc .gitignore
git commit -qm "v1: track labels.csv with DVC"

# ── v2: change the data and commit the new pointer ────────────────────
printf 'image,label\nai_001.jpg,1\nai_002.jpg,1\nreal_001.jpg,0\nreal_002.jpg,0\n' > labels.csv
"$DVC" add labels.csv
git add labels.csv.dvc
git commit -qm "v2: add real_002.jpg"
git log --oneline

# ── return to the older (v1) data ─────────────────────────────────────
git checkout HEAD~1 -- labels.csv.dvc
"$DVC" checkout -q
echo "--- labels.csv after returning to v1 ---"
cat labels.csv

# ── back to the latest version for the remote round trip ──────────────
git checkout HEAD -- labels.csv.dvc
"$DVC" checkout -q
echo "--- labels.csv back at v2 ---"
cat labels.csv

# ── remote storage: push, delete local data + cache, pull it back ─────
"$DVC" remote add -d localstore "$REMOTE"
"$DVC" push
rm -f labels.csv
rm -rf .dvc/cache
echo "--- labels.csv and the local cache are now deleted ---"
ls labels.csv .dvc/cache 2>&1
"$DVC" pull
echo "--- labels.csv restored by dvc pull ---"
cat labels.csv
echo "--- pointer file (labels.csv.dvc) ---"
cat labels.csv.dvc

echo "=== TOY DVC EXERCISE COMPLETE ==="
