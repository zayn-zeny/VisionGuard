"""
labs/02_fashion_mnist_tuner.py — Task 2.7 Toy Part
==================================================
Tune a tiny dense network on Fashion-MNIST with Keras Tuner (random search):
learning rate and number of units, exactly as the assignment asks.

Usage:
  python labs/02_fashion_mnist_tuner.py --trials 6 --epochs 3

Output: prints the trial table and the best settings.
"""
import argparse

import keras_tuner as kt
import numpy as np
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 784).astype("float32")
    x_test = (x_test / 255.0).reshape(-1, 784).astype("float32")
    return (x_train, y_train), (x_test, y_test)


def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    lr = hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3, 5e-3])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_data()

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=args.trials,
        executions_per_trial=1,
        overwrite=True,
        directory="labs/tuner_fashion",
        project_name="fashion_mnist",
        seed=SEED,
    )
    tuner.search(
        x_train, y_train,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1)],
        verbose=2,
    )

    print("\n=== TRIALS ===")
    for trial_id, trial in tuner.oracle.trials.items():
        print(f"trial {trial_id:>2}  {dict(trial.hyperparameters.values)}  "
              f"score={trial.score}")

    best = tuner.get_best_hyperparameters(1)[0]
    print("\n=== BEST HYPERPARAMETERS ===")
    print(best.values)
    best_model = tuner.hypermodel.build(best)
    best_model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=2)
    loss, acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nBest model test accuracy: {acc:.4f}  (loss {loss:.4f})")
