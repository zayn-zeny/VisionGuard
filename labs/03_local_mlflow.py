"""
labs/03_local_mlflow.py — Task 3.1 Toy Part
===========================================
Train a tiny Fashion-MNIST model twice with different learning rates, logging
parameters, per-epoch metrics, one figure and the model to a LOCAL MLflow store.

Usage:
  python labs/03_local_mlflow.py

Then open the UI:
  mlflow ui --backend-store-uri ./labs/mlruns --port 5001
and compare the two runs in the experiment "toy_local".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

SEED = 42
TRACKING_URI = "./labs/mlruns"


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train[:6000] / 255.0).reshape(-1, 784).astype("float32")
    y_train = y_train[:6000]
    x_test = (x_test[:1000] / 255.0).reshape(-1, 784).astype("float32")
    y_test = y_test[:1000]
    return (x_train, y_train), (x_test, y_test)


def build(units=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model


def run_one(lr: float, x_train, y_train, x_test, y_test) -> float:
    tf.keras.utils.set_random_seed(SEED)
    with mlflow.start_run(run_name=f"toy_lr_{lr}"):
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("units", 64)
        mlflow.log_param("epochs", 4)
        mlflow.set_tag("model_family", "dense-mnist")

        model = build()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            x_train, y_train, validation_split=0.1,
            epochs=4, batch_size=128, verbose=2,
        )

        for epoch, acc in enumerate(history.history["accuracy"]):
            mlflow.log_metric("train_accuracy", float(acc), step=epoch)
        for epoch, acc in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", float(acc), step=epoch)
        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("train_loss", float(loss), step=epoch)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history["accuracy"], label="train")
        ax.plot(history.history["val_accuracy"], label="val")
        ax.set_title(f"Fashion-MNIST toy — lr={lr}")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()
        fig_path = f"labs/toy_curve_lr_{lr}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="plots")

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", float(acc))

        # Log the model as a plain Keras artifact (keeps the toy run light).
        model_path = f"labs/toy_model_lr_{lr}.keras"
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"lr={lr}: test accuracy {acc:.4f}  run {mlflow.active_run().info.run_id}")
        return float(acc)


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("toy_local")
    (x_train, y_train), (x_test, y_test) = load_data()
    for lr in (1e-3, 1e-1):
        run_one(lr, x_train, y_train, x_test, y_test)
    print(f"\nRuns logged to {TRACKING_URI}. "
          "Open: mlflow ui --backend-store-uri ./labs/mlruns --port 5001")
