import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import sklearn
from  sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import  statsmodels.api as sm

data=pd.read_csv("startup_company.csv");
#print(data.head())
# creating input and output vars
real_x=data.iloc[:,0:4];
real_y=data.iloc[:,4]
# using label endcoder and dummy variable concept
le=LabelEncoder()
real_x.iloc[:,3]=le.fit_transform(real_x.iloc[:,3])
print(real_x.iloc[:,3])
OneHE=OneHotEncoder(categorical_features=[3])
real_x=OneHE.fit_transform(real_x).toarray() # converts to binary foramt

#print(real_x)
# removng mutlicolinearity
real_x=real_x[:,1:]
# dividding training and testing data
training_x,test_x,training_y,test_y=train_test_split(real_x,real_y,test_size=.2,random_state=0)
# model creation
MLR=LinearRegression()
MLR.fit(training_x,training_y)# training of model
pred_y=MLR.predict((test_x))
#print(pred_y)
#print(test_y)
# margin of error is high so applying orrdinary least square
real_x=np.append(arr=np.ones((50,1)).astype(int),values=real_x,axis=1)
x_opt=real_x[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(exog=x_opt,endog=real_y).fit()
print(reg_OLS.summary())

# now back elimination comes in existence
x_opt=real_x[:,[0,2,3,4,5]]
reg_OLS=sm.OLS(exog=x_opt,endog=real_y).fit()
print(reg_OLS.summary())

x_opt=real_x[:,[0,3,4,5]]
reg_OLS=sm.OLS(exog=x_opt,endog=real_y).fit()
print(reg_OLS.summary())


x_opt=real_x[:,[0,3,5]]
reg_OLS=sm.OLS(exog=x_opt,endog=real_y).fit()
print(reg_OLS.summary())


x_opt=real_x[:,[0,3]]
reg_OLS=sm.OLS(exog=x_opt,endog=real_y).fit()
print(reg_OLS.summary())
