**XGBoost**

**Importing the Libraries**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:,3:13].values
y = data.iloc[:, 13].values

**Encoding Categorical Data**

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:,1:]

**Splitting the dataset**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

**Fitting XGBoost to the Training Set**

from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(x_train,y_train)

**Predicting Test Results**

y_pred = classifier.predict(x_test)

**Confusion Matrix**

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

**Applying k-Fold Cross Validation**

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, x=x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
