# classification algorithims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


sal=pd.read_csv("Salary.csv")
x_data=sal.iloc[:,[0]]
y_data=sal.iloc[:,[1]]

train_x,test_y,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=101)
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
sv=SVC(kernel="linear",degree=100)
sv.fit(train_x,train_y)

kn=KNeighborsClassifier(metric="minkowski")
kn.fit(train_x,train_y)

lg=LogisticRegression()
lg.fit(train_x,train_y)
pred_y3=lg.predict(train_x)
pred_y2=kn.predict(train_x)
pred_y1=sv.predict(train_x)
print(pred_y1)
print(train_y)
plt.scatter(x_data,y_data)

plt.show()




















