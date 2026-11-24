from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_actual = [1, 1, 0, 0, 1, 0, 1, 0]
y_predicted = [1, 1, 0, 1, 1, 0, 0, 0]

accuracy = accuracy_score(y_actual, y_predicted)
precision = precision_score(y_actual, y_predicted)
recall = recall_score(y_actual, y_predicted)
f1 = f1_score(y_actual, y_predicted)

print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1, 2))
