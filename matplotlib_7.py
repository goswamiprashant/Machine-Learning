import matplotlib.pyplot as plt
activites=["eat","sleep","Repeat","study"]
slices=[3,3,5,2]
plt.pie(slices,labels=activites,startangle=90,shadow=True, autopct='%1.2f%%',explode=(0,0,0.2,0))
plt.legend()
plt.title("Pie_chart")
plt.show()