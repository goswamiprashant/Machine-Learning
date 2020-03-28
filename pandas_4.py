import pandas as pd
df=pd.read_csv("nba.txt")
print(df.head())
print(df.tail())
print(df.info()) # to get basic details of data frames
print(df.index)
print(df.shape)
print(df.dtypes)
print(df.columns)  # to get the names of columns
print(df.axes)     # to get the names of columns
#print(df.get_dtype_counts() )
sp=pd.read_csv("stock_price.csv")
print(sp.sum()) # to get sum
print(sp.sum())
print(df.Name)          # to access the column elements
print(df["Name"])
print(df[["Name","Salary"]])
x=["Name","Team","Salary"]
print(df[x])            # more efficient way
df["League"]="National basketball Association"# to add more columns in dataframes
print(df.head())
print(df.insert(1,"Sport","BasketBall"))     # to add more columns in dataframes at appropriate location
print(df.head())
print(df["Salary"].add(5),df["Salary"]+500) # broadcasting operations
print(df["Salary"].sub(5),df["Salary"]-500)
print(df["Salary"].mul(5),df["Salary"]*500)

