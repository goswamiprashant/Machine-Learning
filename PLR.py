import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("pos_sal.csv")
#print(data.head())
# independent and depedent variables
train_x=data.iloc[:,1:2] # array of arrays
train_y=data.iloc[:,2]
#print(real_x,real_y)

#division of data
#train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)

# buliding of model
lin=LinearRegression()
lin.fit(train_x,train_y)
pred_y=lin.predict(train_x)
#print(test_y,pred_y) # error margin is high

# introducing polynomial  lin  reg
poly=PolynomialFeatures(degree=4)
train_x_poly=poly.fit_transform(train_x)
poly.fit(train_x_poly,train_y)
#print(train_x_poly)
lin2=LinearRegression()
lin2=lin2.fit(train_x_poly,train_y)
pred_y_poly=lin2.predict(train_x_poly)
#print(train_y,pred_y_poly)
#plt.scatter(real_x,real_y,color="yellow")
#plt.scatter(train_x,lin2.predict(poly.fit_transform(train_x)),color="red")# yellow =predicted output red=real output
#plt.scatter(train_x,train_y,color="yellow")
#plt.plot(train_x,pred_y_poly,color="red")
plt.show()
print(data.iloc[1])
print(lin2.predict(poly.fit_transform([[2.5]])))