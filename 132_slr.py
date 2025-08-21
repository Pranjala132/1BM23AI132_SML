#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data.csv') 

X = df['YearsExperience'].values
y = df['Salary'].values

n = len(X)
mean_x = np.mean(X)
mean_y = np.mean(y)

numerator = np.sum((X - mean_x) * (y - mean_y))
denominator = np.sum((X - mean_x) ** 2)
m = numerator / denominator
b = mean_y - m * mean_x

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

y_pred = m * X + b

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression (Manual)')
plt.legend()
plt.show()


# In[ ]:




