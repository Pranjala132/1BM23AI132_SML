#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

expenditure = [55, 65, 70, 80, 79, 84, 98, 95, 90, 75, 74, 110, 113, 125, 108, 115, 140, 120, 145, 130, 152, 144, 175, 180, 135, 140, 178, 191, 137, 189]
income = [80, 100, 85, 110, 120, 115, 130, 140, 125, 90, 105, 160, 150, 165, 145, 180, 225, 200, 240, 185, 220, 210, 245, 260, 190, 205, 265, 270, 230, 250]

X = sm.add_constant(income)
y = expenditure

model = sm.OLS(y, X).fit()
beta0, beta1 = model.params
print("Model Parameters")
print("B0 value", beta0)
print("B1 value", beta1)
names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test_result = het_breuschpagan(model.resid, model.model.exog)

for name, value in zip(names, test_result):
    print(f"{name}: {value:.4f}")
    
p_value = test_result[1]
if p_value < 0.08:
    print("Conclusion: Reject the null hypothesis. There is evidence of heteroskedasticity.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No significant evidence of heteroskedasticity.")


# In[ ]:




