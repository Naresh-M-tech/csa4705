import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.random.rand(1000, 10, 1)
y = X[:, -1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

prediction = model.predict(X_test[:5], verbose=0)

print("Test Loss:", round(loss, 4))
print("Test MAE:", round(mae, 4))
print("Predicted Values:", np.round(prediction.flatten(), 3))
print("Actual Values:", np.round(y_test[:5], 3))
