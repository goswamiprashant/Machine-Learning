# Regression algos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

sal=pd.read_csv("Salary.csv")
print(sal.head())
x_data=sal.iloc[:,[0]].values
y_data=sal.iloc[:,[1]].values
#print(x_data.head())
poly=PolynomialFeatures(degree=4)
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=101)
lin=LinearRegression()
lin.fit(train_x,train_y)
pred_y=lin.predict(test_x)
print(pred_y[1])
#plt.scatter(test_x,test_y,color="red")
#plt.scatter(test_x,pred_y)
lin2=LinearRegression()
train_x_poly=poly.fit_transform(train_x)
lin2.fit(train_x_poly,train_y)
pred_y_poly=lin2.predict(train_x_poly)


rd=RandomForestRegressor(n_estimators=500)
rd.fit(train_x,train_y)
pred_y_rand=rd.predict(train_x)

x_grid=np.arange(min(x_data),max(x_data),0.01)
x_grid=x_grid.reshape(len(x_grid),1)

des=DecisionTreeRegressor()
des.fit(train_x,train_y)
des_y=des.predict(train_x)
plt.scatter(x_data,y_data)
plt.plot(x_grid,des.predict(x_grid),color="blue")
plt.plot(x_data,lin2.predict(poly.fit_transform(x_data)),color="black")
plt.plot(x_data,lin.predict(x_data),color="red")
plt.plot(x_grid,rd.predict(x_grid),color="green")
plt.show()











