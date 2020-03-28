import pandas as pd
import  datetime

df=pd.read_csv("nba.txt")
print(df.tail())
print(df.dropna(axis=0,how="all"))# to drop NaN values
print(df.dropna(axis=1,how="all"))
print(df.fillna(0))  # to replace NaN values with other appropriatee values
print(df["Name"].fillna("Not availble"))
print(df.sort_values(by="Name"))        # for sorting values
print(df.sort_values(by=["Name","Team"],ascending=["True","False"],na_position="first"))
#df["Rank"]=df['Salary'].rank().astype(int) #for rank
emp=pd.read_csv("employees.csv")
print(emp.head(),emp.info())
emp["Senior Management"]=emp["Senior Management"].astype(bool)  # filtering of dataseries in dataframe
#emp["Start Date"]=emp["Start Date"].astype(datetime)
#emp["Last Login Time"]=emp["Last Login Time"].astype("datetime")
emp["Gender"]=emp["Gender"].astype("category")
print(emp.info())
m1=emp["Gender"]=="Male"# filtering of dataset with condtions
m2=emp['Team']!="Finance"
m3=emp["Start Date"]<="01/01/1980"
print(emp[(m1 & m2) | m3])
m4=emp["Team"].isin(["Finance","Marketing","Product"])  # multiple values of a condition
print(emp[m4])
m5=emp["Bonus %"].between(2.0,5.0)   # when condition has a range of values
print(emp[m5])
print(emp.sort_values(by="First Name",inplace=True))
print(emp[emp["First Name"].duplicated()])
print(emp.head())
print(emp.drop_duplicates(subset=["First Name","Team"],keep="first")) # for removing duplicate values
print(emp.set_index("First Name"))  # for setting   index
print(emp.reset_index())           # for removing index
