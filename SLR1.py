import numpy as np
import Randomforest as rf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import  RandomForestRegressor


data = pd.read_csv("l1.csv")  # loading preprocessed data
print(data.head())
train_x = data.iloc[:, 0:1].values  # accesing input data
train_y = data.iloc[:,1].values  # accesing output data

#train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
lin = LinearRegression()
lin.fit(train_x, train_y)
y_pred = lin.predict(train_x)
lin_p = LinearRegression()
poly = PolynomialFeatures(degree=4)
train_x_poly = poly.fit_transform(train_x)
poly.fit(train_x_poly, train_y)
lin_p = lin_p.fit(train_x_poly, train_y)
y_pred_poly = lin_p.predict(train_x_poly)

reg=RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(train_x,train_y)
pred_y=reg.predict(train_x)
print(reg.predict([[6]]))

x_grid=np.arange(min(train_x),max(train_x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
# plotting the graph
plt.scatter(train_x,pred_y,color='red')
plt.plot(x_grid,reg.predict(x_grid))

#plt.scatter(train_x, train_y)
plt.plot(train_x, y_pred, 'red')
#plt.scatter(train_x, train_y, "green")
#plt.scatter(train_x,train_y,color="green")
plt.scatter(train_x,y_pred_poly)

plt.show()
