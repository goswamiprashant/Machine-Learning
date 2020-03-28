import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv("IRIS.csv")
print(data.head())
real_x=data.iloc[:,0:4]
real_y=data.iloc[:,4]
le=LabelEncoder()
real_y=le.fit_transform(real_y)
Ohe=OneHotEncoder()
real_x=Ohe.fit_transform(real_x).toarray()
print(real_x[1])