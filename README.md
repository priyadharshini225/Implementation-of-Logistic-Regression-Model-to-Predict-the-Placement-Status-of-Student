# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRIYADHARSHINI S
RegisterNumber: 212223240129

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
*/
```

## Output:

![Screenshot 2024-09-16 091035](https://github.com/user-attachments/assets/db322afb-e612-45f6-a788-5d355b22b8d1)

![Screenshot 2024-09-16 091052](https://github.com/user-attachments/assets/9e416096-933c-4b41-8906-b80d5b4bd4e6)

![Screenshot 2024-09-16 091101](https://github.com/user-attachments/assets/c58b678f-4af8-4cc1-84f1-832f9b8effa6)

![Screenshot 2024-09-16 091113](https://github.com/user-attachments/assets/19539c32-a6e2-41ea-847f-1e613a1d85ec)

![Screenshot 2024-09-16 091124](https://github.com/user-attachments/assets/84055b33-54b2-4bc8-afb9-e86c92fff959)

![Screenshot 2024-09-16 091133](https://github.com/user-attachments/assets/d77c6c1f-a1f4-4023-9af9-32890cf88a75)

![Screenshot 2024-09-16 091139](https://github.com/user-attachments/assets/3a4d44f2-dcaf-45f3-8893-1156e1de4942)

![Screenshot 2024-09-16 091153](https://github.com/user-attachments/assets/5ab1cc41-d69e-4720-93fe-35f146156214)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
