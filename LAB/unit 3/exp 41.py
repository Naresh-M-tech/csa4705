import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.model_selection import train_test_split

X = np.random.rand(1000, 16, 16, 1) * 0.2
y = np.array([0, 1] * 500)

for i in range(1000):
    if y[i] == 0:
        X[i, :, 4:12, 0] += 0.8
    else:
        X[i, 4:12, :, 0] += 0.8

X = np.clip(X, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Input(shape=(16, 16, 1)),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")

predictions = model.predict(X_test[:5], verbose=0)
labels = ["Cat" if p < 0.5 else "Dog" for p in predictions]

print("Predicted Labels:", labels)
print("Actual Labels:", ["Cat" if v == 0 else "Dog" for v in y_test[:5]])
