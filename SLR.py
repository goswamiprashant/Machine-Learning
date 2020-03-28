import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("Salary.csv") # loading preprocessed data
print(data.head())
real_x=data.iloc[:,0].values   # accesing input data
real_y=data.iloc[:,1].values   # accesing output data

real_x=real_x.reshape(-1,1)    # 1d to 2d
real_x=real_x.reshape(-1,1)

# divison of data into training and testing data
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=.3,random_state=0)

# building up a model
Lin=LinearRegression()
Lin.fit(train_x,train_y)

# prdection and testing
pred_y=Lin.predict(test_x)

# plotting of regression line  of  data
print(test_y[1],pred_y[1])
print(test_x,test_y)
plt.scatter(real_x,real_y,color="green")
plt.plot(train_x,Lin.predict(train_x),color="red")
plt.title("Salary Exp Plot")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()

# plotting of regression line of testing data
plt.scatter(test_x,test_y,color="green")
plt.plot(train_x,Lin.predict(train_x),color="red")
plt.title("Salary Exp Plot")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()

# further we can check
m=Lin.coef_
c=Lin.intercept_
x=2.0
pred=m*x+c
print(pred)