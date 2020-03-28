import pandas as pd
l=[13,14,15,17,18,19]
n=['a','b','c','d','e','f']
m = pd.Series(l)
print(m)
print(m.index)
print(m.values)
print(m.dtype)
o = pd.Series(data=l,index=n)
print(o)

data = pd.read_csv("Pokemon.txt",squeeze=True)
print(data.head(10))
print(data.index)
print(data.values)


print(data.ndim)
print(data.shape)
print(data.size)

print(data.sort_values(by="Pokemon" ,inplace=False,ascending=False))
print(data.sort_index(inplace=False,ascending=False))
print(data[0:10])
# math functions min(),max(),median(),mean(),std(),sum(),product(),idxmax(),idxmin,value_counts()