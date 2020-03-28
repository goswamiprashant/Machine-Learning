import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("IRIS.csv")

real_x=data.iloc[:30,0:2]
real_y=data.iloc[:30,2]


rfr=RandomForestRegressor(n_estimators=100)
rfr.fit(real_x,real_y)
pred_y=rfr.predict(real_x)
plt.scatter(real_x,pred_y)
plt.show()
