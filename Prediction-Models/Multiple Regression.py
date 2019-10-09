import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')

data.head()

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x[0]

**Creating Dummy Variables**

State = pd.get_dummies(data['State'],drop_first=True)

State.head()

data.head()

data = pd.concat([data,State],axis=1)

data.head()

data.drop('State',axis=1,inplace=True)

data.head()
x = data.drop('Profit',axis=1)
y = data['Profit']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

y_pred

y_test

# Backward Elimination MLR

import statsmodels.formula.api as sm

Based on the MLR formula we have b0(constant) and x0 so we need to add both to our formula, here x0 = 1. Here we are putting the values of x dataframe into our array of np ones based on the parameters arr and values

x = np.append(arr=np.ones((50,1)).astype(int), values=x,axis=1)

#x

x_opt = x[:, [0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y , exog=x_opt).fit()

regressor_OLS.summary()

data.head()

x_opt = x[:, [0,1,2,3,4]]
regressor_OLS = sm.OLS(endog=y , exog=x_opt).fit()
regressor_OLS.summary()

**Lets keep removing variables**

x_opt = x[:, [0,1]]
regressor_OLS = sm.OLS(endog=y , exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)

#x_Modeled

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        adjR_before = regressor_OLS.rsquared_adj
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)
