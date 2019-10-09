## Support Vector Machines
# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.head()

sns.set_style(style='darkgrid')
sns.pairplot(iris, hue='species',palette='Dark2')

**Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

setosa = iris[iris['species']=='setosa']

sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True, shade_lowest=False)

# Train Test Split

** Split your data into a training set and a testing set.**

from sklearn.model_selection import train_test_split

x = iris.drop('species', axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**

from sklearn.svm import SVC

model = SVC(gamma='scale')

model.fit(x_train,y_train)

## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**

predictions = model.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

## Gridsearch Practice

** Import GridsearchCV from SciKit Learn.**

from sklearn.model_selection import GridSearchCV

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**

param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}

** Create a GridSearchCV object and fit it to the training data.**

grid = GridSearchCV(SVC(),param_grid, refit=True, verbose=2,cv=5)
grid.fit(x_train,y_train)

** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

grid.predictions = grid.predict(x_test)

print(classification_report(y_test, grid.predictions))
print(confusion_matrix(y_test, grid.predictions))

grid.best_params_

## Kernel SVM
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

classifier = SVC(kernel='rbf', random_state=0)

classifier.fit(x_train,y_train)

**Predicting the test results**

y_pred = classifier.predict(x_test)

y_pred

y_test

**Making our confusion matrix**

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

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
