import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data=pd.read_csv("pos_sal.csv")
#print(data.head(10))
real_x=data.iloc[:,1:2].values
real_y=data.iloc[:,2].values
# building of model
reg=DecisionTreeRegressor()
reg.fit(real_x,real_y) #training of model
pred_y=reg.predict(real_x)

#graph plotting
x_grid=np.arange(min(real_x),max(real_x),.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.plot(x_grid,reg.predict(x_grid))
plt.scatter(real_x,real_y)
plt.xlabel("Exp in years")
plt.ylabel("Salary")
plt.title("'Dec tree plot")
plt.show()