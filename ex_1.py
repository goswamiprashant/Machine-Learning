import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("IRIS.csv")
#print(data.head(100))
m1=data.iloc[:,4]=='Iris-virginica'
print(data[m1])
real_x=data.iloc[:,[0,1,2,3]].values
real_y=data.iloc[:,4].values
#print(real_x[1])
#m1="Iris-setosa"
#print(data[m1])

train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
sv=SVC()
sv.fit(train_x,train_y)
pred_y=sv.predict(test_x)
cf=confusion_matrix(test_y,pred_y)
print(cf)
print(sv.predict([[2.34, 3.12,3.1,4.2]]))
plt.scatter(real_x,real_x)
plt.show()
