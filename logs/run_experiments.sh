#!/usr/bin/env bash
set -x
echo "=== START $(date) ==="
./venv/Scripts/python.exe -u scripts/train.py --run-name run2_mobilenet_lr_low --lr 1e-4 --epochs 2 --unfreeze-epochs 2 --augment --autolog > logs/01_autolog.log 2>&1
echo "=== autolog run exit=$? $(date) ==="
./venv/Scripts/python.exe -u scripts/train.py --sweep --epochs 2 --unfreeze-epochs 2 --only-runs run3_mobilenet_lr_high run4_mobilenet_no_aug run5_mobilenet_drop_high run6_mobilenet_unfreeze_5 run7_cnn_baseline > logs/02_sweep.log 2>&1
echo "=== sweep exit=$? $(date) ==="
