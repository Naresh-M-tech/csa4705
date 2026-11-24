import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

data = {
    "Purchase_Frequency": [8, 6, 9, 3, 7, 2, 10, 4],
    "Spending": [900, 700, 1000, 300, 850, 250, 1100, 400],
    "Discount_Usage": [2, 3, 1, 8, 2, 9, 1, 7],
    "Online_Visits": [9, 7, 10, 4, 8, 3, 10, 5]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X = scaler.fit_transform(df)

fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X)

print("Customer Dataset:")
print(df)

print("\nFactor Loadings:")
print(pd.DataFrame(
    fa.components_.T,
    index=df.columns,
    columns=["Factor 1", "Factor 2"]
).round(3))

print("\nExtracted Factors Shape:", factors.shape)
print("Factor Analysis Completed")
