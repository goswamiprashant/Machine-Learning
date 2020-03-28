import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data=pd.read_csv("jamesbond.csv")
print(data.head())
print(data.axes )
data["Budget"]=data['Budget']+6.00
print(data.head())
print(data.notnull())
m1=data['Bond Actor Salary'].notnull()
print(data[m1])
m2=data['Bond Actor Salary'].isnull()
print(data[m2])
print(data.size)
print(data.info())
data.iloc[:,4:7]=data.iloc[:,4:7].fillna(0.0)
print(data.iloc[:,4:7])
#print(data.iloc[:,4:7].dropna());
print(data[data["Actor"].duplicated()])
#data=data.drop_duplicates(subset=["Film","Year","Budget","Actor","Director","Bond Actor Salary","Box Office"],inplace=True)
print(data)
print(data.sort_values(by="Year",ascending=True))
print(data.info())

real_x=data.iloc[:,[1,5]]
real_y=data.iloc[:,[4]]

le=LabelEncoder()

real_x.iloc[:,0]=le.fit_transform(real_x.iloc[:,0])
real_x.iloc[:,1]=le.fit_transform(real_x.iloc[:,1])
#real_x.iloc[:,2]=le.fit_transform(real_x.iloc[:,2])
ohe=OneHotEncoder()
real_x=ohe.fit_transform(real_x).toarray()
#print(real_x)

lin1=LinearRegression()
lin1.fit(real_x,real_y)
pred_y=lin1.predict(real_x)
#plt.scatter(real_x,pred_y)
#plt.show()
