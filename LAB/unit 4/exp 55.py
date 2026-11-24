import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

prices = 100 + np.cumsum(np.random.randn(500) * 0.5)

X = []
y = []

for i in range(10, len(prices)):
    X.append(prices[i-10:i])
    y.append(prices[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    LSTM(32, input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

predictions = model.predict(X_test[:5], verbose=0)

print("Test Loss:", round(loss, 4))
print("Test MAE:", round(mae, 4))
print("Predicted Prices:", np.round(predictions.flatten(), 2))
print("Actual Prices:", np.round(y_test[:5], 2))
