import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

data = load_digits()
X = data.images.reshape(-1, 8, 8, 1) / 16.0
y = to_categorical(data.target, 10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

configs = [(3, 16), (5, 16), (3, 32)]

for size, filters in configs:
    model = Sequential([
        Conv2D(filters, (size, size), activation='relu',
               input_shape=(8, 8, 1), padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Filter Size: {size}x{size}, Filters: {filters}")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
