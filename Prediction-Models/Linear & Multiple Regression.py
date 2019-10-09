>>> # Linear Regression Project

## Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

## Get the Data

**Read in the Ecommerce Customers csv file as a DataFrame called customers.**

customers = pd.read_csv('Ecommerce Customers.csv')

**Check the head of customers, and check out its info() and describe() methods.**

customers.head()

customers.describe()

customers.info()

## Exploratory Data Analysis

sns.jointplot('Time on Website','Yearly Amount Spent',customers,annot_kws=True)

There is no correlation based on the graph

**Do the same but with the Time on App column instead.**

sns.jointplot('Time on App','Yearly Amount Spent',customers)

This new graph illustrates a more comprehensive correlation

**Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

sns.jointplot('Time on App','Length of Membership',customers,kind='hex')

sns.pairplot(customers)

**Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

Its length of membership

**Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.**

sns.lmplot('Yearly Amount Spent','Length of Membership',customers)

## Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
**Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.**

customers.columns

x = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

x.head()

**Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

## Training the Model

Now its time to train our model on our training data!

**Import LinearRegression from sklearn.linear_model **

from sklearn.linear_model import LinearRegression

**Create an instance of a LinearRegression() model named lm.**

lm = LinearRegression()

**Train/fit lm on the training data.**

lm.fit(x_train,y_train)

**Print out the coefficients of the model**

lm.coef_

inputs = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficients'])
inputs

## Predicting Test Data
Now that we have fit our model, let's evaluate its performance by predicting off the test values!

**Use lm.predict() to predict off the X_test set of the data.**

predictions = lm.predict(x_test)
predictions

y_test.head()

**Create a scatterplot of the real test values versus the predicted values.**

sns.scatterplot(y_test, predictions)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Values')

## Evaluating the Model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

**Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)

metrics.mean_squared_error(y_test, predictions)

np.sqrt(metrics.mean_squared_error(y_test, predictions))

print('MAE:', round(metrics.mean_absolute_error(y_test, predictions),2))
print('MSE:', round(metrics.mean_squared_error(y_test, predictions),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))

explained variance score (R^2), this explains the regression fit and the variance that our model explains which is 99% which is a very good model

metrics.explained_variance_score(y_test,predictions)

## Residuals

You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 

**Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

sns.distplot(y_test-predictions,bins=50)

# Logistic Regression with Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
