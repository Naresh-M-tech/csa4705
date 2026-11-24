from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_actual = np.array([3, 5, 7, 9, 11])
y_predicted = np.array([2.5, 5.5, 6, 10, 12])

mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
