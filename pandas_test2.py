import numpy as np
import pandas as pd
nba=pd.read_csv('nba.txt')
print(nba.dtypes)
print(nba.axes)
print(nba.columns)
print(nba.info)
print(nba["Salary"].head())
nba["Sport"]="BasketBall"
print(nba.head())
nba.insert(3,"League","NBA")
print(nba.columns)
d1=nba[["Name","Team",'League', 'Position', 'Age', 'Height',  'Weight', 'College', 'Salary']].notnull()
d2=nba[["Name","Team"]].isnull()
print(d1)
d3=nba[["Name","Team","Position","Height","College"]].isnull()
x=["Name","Team","Position","Height","College"]
#nba[["Name","Team","Position","Height","College"]].fillna("Not Available",inplace=True)
#nba["Name"].fillna("Not Available",inplace=True)
print(nba.tail())
nba.dropna(axis=0,inplace=True)
print(nba.tail())
m1=nba["Age"].between(20,22)
print(nba[m1])
print(nba.info())
#nba['Height']=int(nba["Height"])
print(nba.info())
nba.sort_values("Name",inplace=True,na_position="first")
print(nba)

m2=nba["Team"].isin(["Atlanta Hawks","Denver Nuggets","brooklyn Nets"])
print(nba[m2])
m3=nba["Name"].duplicated()
m4=m3==True
print(nba[m4])

nba.set_index("Name",inplace=True)
print(nba.head())
nba.reset_index(drop=True,inplace=True)
print(nba.sort_values("Team").head())
#print(nba.loc["Name"])
print(nba.iloc[2:5])
print(nba.iloc[2:6,0:3])
#print(nba.ix["Salary"])
print(nba.ix[1:5,2:4])
#print(nba._ix[20:,"Salary"])
m3=nba['Team']=="Atlanta Hawks"
nba.ix[m3,'Team']="Atlantic Hawks"
print(nba.sort_values("Team").head(20))
nba.sort_values("Team",inplace=True)
print(nba.set_index("Team",inplace=True))
y=[]
for i in range (364):
    y.append(i)
nba.insert(0,"Index2",y)
print(nba.info())
print(nba.tail())
nba.reset_index(inplace=True)
print(nba.tail())
nba.rename(columns={"Team":"Name of team","Salary":"Earnings"},inplace=True)
print(nba.columns)
s1=nba.sample(frac=.25)
print(s1)
s2=nba.sample(n=3,axis="columns")
print(s2)