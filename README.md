# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ABBU REHAN
RegisterNumber: 212223240165 
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## 1.Placement Data
![ml 4 1](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/bbb19c3b-2617-4ec6-9803-8ae675dd8e88)
## 2.Salary Data
![ml 4 2](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/38852795-f83a-405e-bb7b-7942c05fe3c4)
## 3.Checking The Null Function()
![ml 4 3](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/7a4cc98d-0c65-4372-b35f-b45d9fe20e6e)
## 4.Data Duplicate
![ml 4 4](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/3fb25f78-cd3c-49fc-b604-8b86350367c3)
## 5.Print Data
![ml 4 5](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/0b6eb93f-ac20-4486-b0d0-475b9a05c94a)
## 6.Data Status
![ml 4 6](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/3ebd0d14-fe2e-4b72-bede-f78a629008fb)
## 7.Y_Prediction Array
![ml 4 7](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/a50add46-0fcd-47b8-a45c-2c75ccb269c4)
## 8.Accuracy Value
![ml 4 8](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/f5f26e56-156c-48c1-adeb-d02049916153)
## 9.Confusion Matrix
![ml 4 9](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/13839b02-f326-44b1-8fe4-e9cfd5b2dc2c)
## 10.Classification Matrix
![ml 4 10](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/744e89d1-0578-4eb0-a4e6-04c162cf6f41)
## 11.Prediction Of LR
![ml 4 11](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/65e34513-61fd-4c46-a841-c7fa2b95fcfc)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
