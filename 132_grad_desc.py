#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
X = data['YearsExperience'].values.astype(float)
Y = data['Salary'].values.astype(float)

m = 0.0
b = 0.0
learning_rate = 0.000001
epochs = 20000
n = float(len(X))

for epoch in range(epochs):
    Y_pred = m * X + b
    dm = (-2/n) * sum(X * (Y - Y_pred))
    db = (-2/n) * sum(Y - Y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db

print(f"Trained model: y = {m:.2f}x + {b:.2f}")

plt.scatter(X, Y, color='blue')
plt.plot(X, m*X + b, color='red')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Linear Regression with Gradient Descent')
plt.show()


# In[ ]:




