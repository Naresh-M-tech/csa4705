from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_pca = PCA(n_components=10).fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_pca, y_train)
pred2 = model2.predict(X_test_pca)

print("Before PCA Accuracy:",
      round(accuracy_score(y_test, pred1) * 100, 2), "%")

print("After PCA Accuracy:",
      round(accuracy_score(y_test, pred2) * 100, 2), "%")
