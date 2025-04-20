#!/usr/bin/env python3
# bayesian_cnn_optimization.py
""" Task 6: 6. Bayesian Optimization with GPyOpt """
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import GPyOpt

# --- Load MNIST Data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- Define Model Creation Function ---
def build_and_train_model(params):
    learning_rate = float(params[0][0])
    num_filters = int(params[0][1])
    dense_units = int(params[0][2])
    dropout_rate = float(params[0][3])
    batch_size = int(params[0][4])

    model = Sequential([
        Conv2D(num_filters, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint with hyperparameter values in filename
    checkpoint_path = f"model_lr{learning_rate}_filters{num_filters}_dense{dense_units}_dropout{dropout_rate}_batch{batch_size}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=batch_size,
        verbose=0,
        callbacks=[checkpoint, early_stop]
    )

    val_acc = max(history.history['val_accuracy'])
    with open("bayes_opt.txt", "a") as log_file:
        log_file.write(f"Params: {params} -> Val Accuracy: {val_acc}\n")

    return -val_acc  # Negative because BayesianOptimization minimizes by default

# --- Define Hyperparameter Search Space ---
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'num_filters', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.2, 0.5)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# --- Run Bayesian Optimization ---
optimizer = GPyOpt.methods.BayesianOptimization(
    f=build_and_train_model,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=False,
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# --- Plot Convergence ---
optimizer.plot_convergence()
plt.savefig("convergence.png")
plt.show()

# --- Save Best Result to Log ---
best_params = optimizer.X[np.argmin(optimizer.Y)]
best_score = -np.min(optimizer.Y)
with open("bayes_opt.txt", "a") as log_file:
    log_file.write(f"\nBest parameters: {best_params}\nBest validation accuracy: {best_score}\n")

    