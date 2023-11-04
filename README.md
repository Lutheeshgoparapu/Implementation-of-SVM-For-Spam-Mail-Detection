# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: G.Lutheesh
RegisterNumber:  212221230029
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Data Head
![data head](https://github.com/Lutheeshgoparapu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94154531/574f67f3-23a1-4082-b7da-f1ab1e0eb546)
## Data Info:
![data info](https://github.com/Lutheeshgoparapu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94154531/98be8f0b-4ed3-4f34-9787-73584b826253)
## Data isnull:
![data isnull](https://github.com/Lutheeshgoparapu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94154531/8614b53b-dbc8-4d3a-9c9d-78ad1b86798d)
## Y_pred:
![y-pred](https://github.com/Lutheeshgoparapu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94154531/41c55487-a822-4ce4-9c65-93253817cdf4)
## Accuracy:
![accuracy](https://github.com/Lutheeshgoparapu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94154531/07c8d623-dd78-42c9-80f8-18b2f6ac65e4)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

