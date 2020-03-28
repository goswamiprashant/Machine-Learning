import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestRegressor

data=pd.read_csv("pos_sal.csv")

real_x=data.iloc[:,1:2].values
real_y=data.iloc[:,2].values

reg=RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(real_x,real_y)
pred_y=reg.predict(real_x)
print(reg.predict([[6]]))

x_grid=np.arange(min(real_x),max(real_x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
# plotting the graph
plt.scatter(real_x,pred_y,color='red')
plt.plot(x_grid,reg.predict(x_grid))
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Random Forest")
plt.show()

