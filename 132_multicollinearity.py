#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()
print("\nMultiple Linear Regression Summary:\n")
print(model.summary())

# Correlation
plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Predictors\n")
plt.show()

#VIF
vif = pd.DataFrame()
vif["Feature"] = X_const.columns
vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("\nVariance Inflation Factor (VIF):\n")
print(vif)

# Ridge Regression remedy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print("Ridge Regression R² Score:\n")
print("R² Score:", r2_score(y_test, y_pred))

# PCA remedy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print("PCA Summary:\n")
print("Original Features:", X.shape[1])
print("Reduced Components (95% variance):", X_pca.shape[1])


# In[ ]:




