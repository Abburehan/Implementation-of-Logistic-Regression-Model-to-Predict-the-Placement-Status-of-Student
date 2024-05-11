# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the python library pandas.

2.Read the dataset of Placement_Data.

3.Copy the dataset in data1.

4.Remove the columns which have null values using drop().

5.Import the LabelEncoder for preprocessing of the dataset.

6.Assign x and y as status column values.

7.From sklearn library select the model to perform Logistic Regression.

8.Print the accuracy, confusion matrix and classification report of the dataset.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SYED ABBU REHAN
RegisterNumber:  212223240165
*/
```
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
```
```
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()
```
```
data1.isnull().sum()
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
```
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
```
```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/fec7836a-5ff0-4faa-9d8e-69fd49ee12b6)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/adc12dea-781d-4ac7-9ba8-27240ab26cd0)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/9b20a100-1c72-49da-b084-b678f2cdb636)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/aa78900d-a55d-407a-ab72-90694e683ea4)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/e0748ff1-f383-4381-b2e1-5ac22c45565e)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/e8e2054a-efbd-459b-ae98-fe309be0a837)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/be11d0cb-0a62-4944-bc44-9194c02416d4)
![image](https://github.com/Abburehan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849336/2bdf01f5-d6ee-4a3c-b59f-6cac47ec7710)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
