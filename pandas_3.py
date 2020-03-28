import pandas as pd
nb=pd.read_csv("nba.txt",squeeze=True,usecols=[1])
print(nb.head(10))
nb.sort_values()
print(nb.sort_values(ascending=True,inplace=True))# for sorting of values
print(nb.sort_index(ascending=False))# for sortinhg of index
print(nb[2])# way of accessing of values through index slicing
print(nb[[6,7,8,9,0]])
print(nb[10:15])
x=["mon",'tue',"wed",'Thur',"fri",'sat',"sun"]
y=["Mango","Apple","Grapes","Guava","banana","Orange","Strawberry"]
nb1=pd.Series(data=y,index=x)
print(nb1["mon"])# way of accessing of values through position label slicing
print(nb1[["mon","tue"]])
print(nb1["mon":"sun"])
print(nb1.get("mon"))    # accesing values through get method
print(nb1.get(["mon","tue"]))
print(nb1.get(["mon"],"This string  is not found"))
x2=[1,2,3,4,5,6,7,8,9]
y2=[2,4,6,8,10,12,14,16,18]
nb2=pd.Series(data=y2,index=x2)
# math functions and idxmax and idxmin to get index of max and min numbers
print(nb2.sum(),nb2.product(),nb2.max(),nb2.idxmax(),nb2.min(),nb2.idxmax(),nb2.mean(),nb2.median(),nb2.std())
print(nb.value_counts())# to get the numbers of type categories members
def lev_up(n):
    if n>5:
        return "Yes"
    else:
        return "No"
print(nb2.apply(lev_up))# to apply specific functionality to dataset