Python 3.7.2 (v3.7.2:9a3ffc0492, Dec 24 2018, 02:44:43) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('Position_Salaries.csv')

data

x = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

x

y

**Create a Linear Regression and Polynomial Regression to compare**

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(x,y)

**Polynomial Regression**

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4) #degree just means exponent number

x_poly = poly_reg.fit_transform(x)

x_poly

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, y)

**Visualizing Linear Regression results**

plt.scatter(x,y, color='red')
plt.plot(x, linear_regressor.predict(x),color='green')

**Visualizing Polynomial Regression results**

plt.scatter(x,y, color='red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)),color='green')

**Predicting new results with Linear Regression**

linear_regressor.predict(x)

**Predicting new results with Polynomial Regression**

lin_reg2.predict(poly_reg.fit_transform(x))
