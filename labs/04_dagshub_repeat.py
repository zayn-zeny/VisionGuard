"""
labs/04_dagshub_repeat.py — Task 3.2 Toy Part
=============================================
Repeat the toy run against the DagsHub MLflow tracking server, proving the
tracking URI + token from .env work end to end (no credentials in code).

Usage:
  python labs/04_dagshub_repeat.py
"""
import os

import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

SEED = 42


def main() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set in .env")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("toy_dagshub")
    print(f"Tracking URI: {uri}")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train[:4000] / 255.0).reshape(-1, 784).astype("float32")
    x_test = (x_test[:800] / 255.0).reshape(-1, 784).astype("float32")
    y_train, y_test = y_train[:4000], y_test[:800]

    with mlflow.start_run(run_name="toy_dagshub_repeat"):
        mlflow.log_params({"learning_rate": 1e-3, "units": 64, "epochs": 2,
                           "dataset": "fashion_mnist_subset"})
        mlflow.set_tags({"purpose": "Task 3.2 toy repeat run on DagsHub"})

        tf.keras.utils.set_random_seed(SEED)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(x_train, y_train, validation_split=0.1,
                            epochs=2, batch_size=128, verbose=2)

        for epoch, acc in enumerate(history.history["accuracy"]):
            mlflow.log_metric("accuracy", float(acc), step=epoch)
        for epoch, acc in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", float(acc), step=epoch)

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", float(acc))
        print(f"\nLogged toy run to DagsHub. test accuracy={acc:.4f}")
        print(f"Run id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
