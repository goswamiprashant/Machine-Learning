import pandas as pd
x=[40.0,54.3,78.8,43,2,56.3,45.3,90.7]
y=pd.Series(x) # conversion of list to series object data
print(y)
print(y.index)  # attributes  of series object to find out the index,values and datatype
print(y.values)
print(y.dtype)
#methods parameters
Fruits=["mango","Apple","Orange","Guava","Pineapple","Grapes",'Banana']
days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
z=pd.Series(data=Fruits,index=days) #(Fruits,days),data=Fruits,days)
print(z)
print(y.sum()) # for calculating sum product and mean
print(y.product())
print(y.mean())

