import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Train a tiny demo SER model on synthetic data.
# Usage: python train_ser_small.py --out_dir ./saved_model
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./saved_model')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Create synthetic mel-spectrogram-like inputs: shape (num_samples, 128, 64, 1)
num_classes = 4
num_samples = 2000
X = np.random.randn(num_samples, 128, 64, 1).astype('float32')
y = np.random.randint(0, num_classes, size=(num_samples,))

model = keras.Sequential([
    layers.Input(shape=(128,64,1)),
    layers.Conv2D(16, 3, activation='relu', padding='same'),
    layers.MaxPool2D(2,2),
    layers.Conv2D(32,3,activation='relu', padding='same'),
    layers.MaxPool2D(2,2),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=6, batch_size=32, validation_split=0.1)

# Save as SavedModel
model.save(args.out_dir)
print('Saved model to', args.out_dir)
