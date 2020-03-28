import pandas as pd

pokemon=pd.read_csv("startup_company.csv",squeeze=True)
print(pokemon.head())
pokemon.sort_values(by="State",ascending=False)
print(pokemon.head())
pokemon.sort_index(ascending=False,inplace=True)
print(pokemon.head())
#print(pokemon.sum(),pokemon.min(),pokemon.max(),pokemon.product())
print(pokemon['Profit'])
print(pokemon[["Administration","Profit"]])
print(pokemon.iloc[2:10,:])

print(pokemon.apply(lambda Profit: Profit*2))


