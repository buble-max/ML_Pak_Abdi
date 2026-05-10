"""
Arsitektur Deep Learning untuk BISINDO:

Input : (batch, T=SEQUENCE_LENGTH, F=FEATURES_PER_FRAME)
        Kita bentuk ulang menjadi (batch, T, NUM_LANDMARKS*MAX_HANDS, 3)
        → setiap "frame" adalah sekuens L=42 titik dengan 3 channel (x,y,z).

Pipeline:
  TimeDistributed(
      Conv1D(64) → BN → ReLU → Conv1D(128) → BN → ReLU → GlobalAvgPool1D
  )                         # ekstraksi fitur spasial per frame
  LSTM(128, return_sequences=True) + Dropout + BN
  LSTM(64)                         # hubungan temporal antar frame
  Dense(128, relu) + Dropout
  Dense(NUM_CLASSES, softmax)

Jumlah parameter relatif kecil (~200K) agar aman di Colab & real-time.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import (
    FEATURES_PER_FRAME,
    LEARNING_RATE,
    MAX_HANDS,
    NUM_CLASSES,
    NUM_LANDMARKS,
    SEQUENCE_LENGTH,
)


def build_model(
    seq_len: int = SEQUENCE_LENGTH,
    feat_dim: int = FEATURES_PER_FRAME,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Bangun dan compile model CNN + LSTM BISINDO."""
    L = NUM_LANDMARKS * MAX_HANDS  # 42
    assert feat_dim == L * 3, \
        f"feat_dim {feat_dim} harus sama dengan {L}*3 = {L*3}"

    inputs = layers.Input(shape=(seq_len, feat_dim), name="landmark_seq")

    # (B, T, F) -> (B, T, L, 3)  supaya bisa Conv1D di dimensi landmark
    x = layers.Reshape((seq_len, L, 3))(inputs)

    # --- Per-frame CNN feature extractor ---
    conv_block = models.Sequential(
        [
            layers.Conv1D(64, 3, padding="same",
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv1D(128, 3, padding="same",
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling1D(),  # (L, C) -> (C,)
        ],
        name="frame_cnn",
    )
    x = layers.TimeDistributed(conv_block, name="td_cnn")(x)  # (B, T, 128)

    # --- Temporal LSTM ---
    x = layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, dropout=0.2)(x)
    x = layers.BatchNormalization()(x)

    # --- Classifier head ---
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = models.Model(inputs, outputs, name="BISINDO_CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
