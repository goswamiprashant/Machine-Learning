import matplotlib.pyplot as plt
ages=[10,7,15,26,34,23,56,43,45,6,77,52,22,78,66,99,55,90,64]
range=(0,100)
bin=10
plt.hist(ages,bin,range,width=2,color='green',histtype='bar')
plt.xlabel("x-axis")
plt.xlabel("y-axis")
plt.title("Histogram")
plt.show()