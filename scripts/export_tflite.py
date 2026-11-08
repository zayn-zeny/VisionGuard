"""
scripts/export_tflite.py — Task 7: export the champion Keras model to TFLite
for the cloud API, and prove the export changes no prediction.

Why: free container hosts give 512 MB RAM. Full TensorFlow + the Keras model
needs roughly 700-900 MB resident, while the TFLite interpreter (ai-edge-litert)
runs the same graph in well under 200 MB and its wheel is ~5 MB instead of
~500 MB. Float32 export (no quantisation) keeps the arithmetic identical.

Output: models/champion.tflite and a parity table on the test split.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parents[1]
KERAS_PATH = ROOT / "models" / "champion.keras"
TFLITE_PATH = ROOT / "models" / "champion.tflite"
THRESHOLD = 0.4


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(io.BytesIO(path.read_bytes())).convert("RGB")
    arr = np.array(img.resize((224, 224), Image.BILINEAR), dtype=np.float32)
    return preprocess_input(arr)[None, ...]


def main() -> None:
    model = keras.models.load_model(KERAS_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    TFLITE_PATH.write_bytes(converter.convert())
    print(f"keras  : {KERAS_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"tflite : {TFLITE_PATH.stat().st_size / 1e6:.1f} MB")

    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]

    test = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    diffs, flips = [], 0
    for rel in test["path"]:
        x = preprocess(ROOT / rel.replace("\\", "/"))
        pk = float(model.predict(x, verbose=0)[0][0])
        interp.set_tensor(inp, x)
        interp.invoke()
        pt = float(interp.get_tensor(out)[0][0])
        diffs.append(abs(pk - pt))
        flips += (pk >= THRESHOLD) != (pt >= THRESHOLD)
    print(f"test images compared : {len(diffs)}")
    print(f"max |p_keras - p_tflite| : {max(diffs):.2e}")
    print(f"mean |p_keras - p_tflite|: {np.mean(diffs):.2e}")
    print(f"label flips at threshold {THRESHOLD}: {flips}")


if __name__ == "__main__":
    main()
