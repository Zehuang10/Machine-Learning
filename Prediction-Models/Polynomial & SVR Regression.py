#Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#SVR 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

data

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

x

y

y = y.reshape(-1,1)

**Feature Scaling**

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

**Fitting SVR to the dataset**

from sklearn.svm import SVR

regressor = SVR(kernel='rbf',gamma='auto')

regressor.fit(x,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

y_pred #It is a very close prediction to the actual value

The values are not good here because SVR does not apply feature scaling in this library

**Visualizing the SVR results**

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x))

**Visualizing the SVR results (for higher resolution and smoother curve)**

x_grid = np.arange(min(x),max(x),0.1)

x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid))
