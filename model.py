import pandas as pd
import numpy as np
from pandas import get_dummies
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

emp_data=pd.read_csv("HR_comma_sep.csv")
print(emp_data.isnull())
print(emp_data.describe())

X=emp_data.columns.drop("left")
y=emp_data["left"]
enc_data=get_dummies(emp_data[X])
X_train,X_test,y_train,y_test=train_test_split(enc_data,y,test_size=0.2,random_state=0)
scaler=StandardScaler()
X_trscaled=scaler.fit_transform(X_train)
X_ttscaled=scaler.fit_transform(X_test)
model=LogisticRegression()
model.fit(X_trscaled,y_train)

train_accu=model.score(X_trscaled,y_train)
test_accu=model.score(X_ttscaled,y_test)
print("train_acc",train_accu)
print("test_acc",test_accu)
train_pred=model.predict(X_trscaled)
test_pred=model.predict(X_ttscaled)
print("classification_report",classification_report(y_test,test_pred))



import pickle

pickle.dump(model,open("trained_model.sav",'wb'))
pickle.dump(model,open("scaler.sav","wb"))

