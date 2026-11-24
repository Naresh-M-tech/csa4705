import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

data = {
    "Marks": [85, 78, 90, 65, 88, 72, 95, 80],
    "Attendance": [95, 88, 98, 70, 92, 75, 99, 85],
    "Assignments": [90, 80, 95, 68, 92, 74, 98, 82],
    "Participation": [85, 75, 90, 60, 88, 70, 96, 78]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X = scaler.fit_transform(df)

fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X)

print("Original Dataset:")
print(df)

print("\nFactor Loadings:")
print(pd.DataFrame(fa.components_.T,
                   index=df.columns,
                   columns=["Factor 1", "Factor 2"]).round(3))

print("\nExtracted Factors Shape:", factors.shape)
print("Factor Analysis Completed")
