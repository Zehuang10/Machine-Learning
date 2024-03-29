# K-Fold Cross Validation & Grid Search for model optimization

Apply K-Fold to the Kernel SVM model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

**Splitting dataset into training and test set**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

**Feature Scaling**

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

**Fitting classifier to the training set**

from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)

classifier.fit(x_train,y_train)

**Predicting the test results**

y_pred = classifier.predict(x_test)

y_pred

y_test

**Making our confusion matrix**

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

**Advanced model evaluation with k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train,cv=10)

np.round(accuracies,2)

accuracies.mean()

**Look at the variance of the model**

accuracies.std() #Low variance between accuracies for model performance

**Visualizing the Training results**

plt.figure(figsize=(10,5))
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.10,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j, 1],
               c = ListedColormap(('red','green'))(i), label=j)

plt.legend()
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

**Visualizing the Test results**

plt.figure(figsize=(10,5))
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
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
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# Grid Search

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('Social_Network_Ads.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

**Splitting dataset into training and test set**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

**Feature Scaling**

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

**Fitting classifier to the training set**

from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0, gamma=0.5)

classifier.fit(x_train,y_train)

**Predicting the test results**

y_pred = classifier.predict(x_test)

y_pred

y_test

**Making our confusion matrix**

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

**Advanced model evaluation with k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train,cv=10)

np.round(accuracies,2)

accuracies.mean()

accuracies.std() #Low variance between accuracies for model performance

**Applying Grid Search to find the best model and parameters**

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000],'kernel': ['linear']},
              {'C': [1, 10, 100, 1000],'kernel': ['rbf'],'gamma':[0.5,0.1,0.3,0.4,0.2,0.6,0.7,0.8,0.9]}]

grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)

grid_search = grid_search.fit(x_train,y_train)

best_accuracy = grid_search.best_score_

best_accuracy

grid_search.best_params_ #Given an optimal gamma value of 0.5, we can choose different values for grid search
