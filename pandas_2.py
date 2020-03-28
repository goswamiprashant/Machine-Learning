import pandas as pd
x=pd.read_csv("info.txt")# to read the data from file
y=pd.read_csv("info.txt",squeeze=True,usecols=[1])# (squeeze)to change the data into pandas series Object,(usecols to use particular columns
print(y)
# after collecting data from files  proprer arrangement of should be done
print(y.sort_values())# used for sorting of data
print(y.sort_values(ascending=False).head(4))# for getting values in descending order
print(y.sort_values(ascending=False,inplace=True).tail(4))# for allowing the current changes in the current series obj
print(y)