'''
Just showing I know how to do a linreg
https://datatofish.com/multiple-linear-regression-python/
'''

### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm


### Read in data as dataframe
df_ypred = pd.read_csv("./input/y_prediction.csv")

### Determine linearity using scatter plot
plt.scatter(df_ypred['y'], df_ypred['x_1'], df_ypred['x_2'], color='green')
plt.grid(True)
plt.show()

### Linear Regression
regr = linear_model.LinearRegression()
y = df_ypred["y"]
x1 = df_ypred[["x_1"]]
x2 = df_ypred[["x_2"]]

lr1 = regr.fit(x1,y)
print('Intercept: \n', lr1.intercept_)
print('Coefficients: \n', lr1.coef_)

lr2 = regr.fit(x2,y)
print('Intercept: \n', lr2.intercept_)
print('Coefficients: \n', lr2.coef_)

### Multilinear Regression
x = df_ypred[['x_1','x_2']]
y = df_ypred['y']

regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


### Stats
x = sm.add_constant(x)
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
 
print_model = model.summary()
print(print_model)

