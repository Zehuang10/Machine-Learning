## Naive Bayes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('Social_Network_Ads.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

**Fitting classifier to the training set**

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train,y_train)

**Predicting the test results**

y_pred = classifier.predict(x_test)

y_pred

y_test

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred)

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
