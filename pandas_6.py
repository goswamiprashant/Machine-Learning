import pandas as pd
bond=pd.read_csv("jamesbond.csv",index_col="Film")
print(bond.head())
print(bond.loc["From Russia with Love"]) # accesing row values from index label
print(bond.loc["From Russia with Love":"The Spy Who Loved Me"])
print(bond.loc[["From Russia with Love","The Spy Who Loved Me"]])
print(bond.iloc[10])   # accessing row values from index locaton
print(bond.iloc[1:10])
print(bond.iloc[[1,10]])
print(bond.ix[1])  # combination of loc and iloc
print(bond.ix["From Russia with Love"])
print(bond.loc["From Russia with Love","Actor"])# accessing row's specific values
print(bond.iloc[1,1])
print(bond.ix[1,"Actor"])
m1=bond["Actor"]=="Sean Connery"  # to set new values in rows
bond.ix[m1,"Actor"]="Sir Sean Connery"
print(bond["Actor"])
bond.ix["From Russia with Love",["Actor","Director"]]=['Sean Connery',"Sir terence Young"]   # to set the multiple values
print(bond.ix[1])
print(bond.head())
print(bond.rename(columns={"Year":"Release Date","Bond Actor Salary":"Salary"},inplace=True))
print(bond.columns)
print(bond.rename(index={"Dr. NO":"Doctor No"},inplace=True))
print(bond.head())
# for deleting the rows an columns
bond.drop("Goldfinger",axis=0,inplace=True) # deleting rows
print(bond.head())
bond.drop("Salary",axis=1,inplace=True)  # deleting columns
print(bond.head())
actor=bond.pop("Actor") # popping out a column
print(actor )
print(bond.head())
del bond["Budget"] # deleting columns directly
print(bond.head())
#print(bond.sample(4)) # for creating a random sample of dataframe
#print(bond.sample(4,axis=1))
# print(bond.sample(frac=.25,axis=0))
print(bond.nsmallest(2,columns=["Box Office"]))# getting smallest and largest values sorting with app no
print(bond.nlargest(2,columns=["Box Office"]))
m3=bond["Director"]=="Terence Young"                            # to apply con with the help of where()
m4=bond["Release Date">"1960"]
#print(bond.where(m4))
# to apply particular functionality to a dataseries
def add_millions(number):# on rows
    str(number)+" Millions!"
clm=["Budget","Box Office"]
for col in clm:
    bond[col]=bond[col].apply(add_millions)
print(bond.head())

def good_movie(row):
    budget=row[4]
    if budget==5.0:
        return "Good"
    elif budget>5.0:
        return "Better"
    else:
        return "Bad"
bond.apply(good_movie,axis="column")

