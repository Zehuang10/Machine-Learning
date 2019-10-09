#K-Nearest Neighbors Algorithm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Get the Data
** Read the 'KNN_Project_Data csv file into a dataframe **

df = pd.read_csv('KNN_Project_Data')

**Check the head of the dataframe.**

df.head()

sns.pairplot(df, hue = 'TARGET CLASS')

# Standardize the Variables
** Import StandardScaler from Scikit learn.**

from sklearn.preprocessing import StandardScaler

** Create a StandardScaler() object called scaler.**

scaler = StandardScaler()

** Fit scaler to the features.**

scaler.fit(df.drop('TARGET CLASS',axis=1))

**Use the .transform() method to transform the features to a scaled version.**

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

**Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

df_features = pd.DataFrame(scaled_features,columns = df.columns[:-1])

df_features.head()

# Train Test Split

**Use train_test_split to split your data into a training set and a testing set.**

from sklearn.model_selection import train_test_split

x = df_features
y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Using KNN

**Import KNeighborsClassifier from scikit learn.**

from sklearn.neighbors import KNeighborsClassifier

**Create a KNN model instance with n_neighbors=1**

knn = KNeighborsClassifier(n_neighbors=1)

**Fit this KNN model to the training data.**

knn.fit(x_train,y_train)

# Predictions and Evaluations
Let's evaluate our KNN model!

**Use the predict method to predict values using your KNN model and X_test.**

pred = knn.predict(x_test)

pred

** Create a confusion matrix and classification report.**

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, pred))
print(confusion_matrix(y_test,pred))

# Choosing a K Value
Let's go ahead and use the elbow method to pick a good K Value!

** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

**Now create the following plot using the information from your for loop.**

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
plt.plot(range(1,40),error_rate,color='green',linestyle='--',marker = 'v', markerfacecolor = 'red',markersize = 10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

## Retrain with new K Value

**Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test,pred))

## Visualizing training results
plt.figure(figsize=(10,5))
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.25,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j, 1],
               c = ListedColormap(('red','green'))(i), label=j)

plt.legend()
plt.show()
