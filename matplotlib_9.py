# to draw a graph by accessing data from a file
import matplotlib.pyplot as plt
import csv
x=[]
y=[]
with open("info.txt",'r') as csvfile:
  plots=csv.reader(csvfile)
  for col in plots:
     x.append(col[0])
     y.append(col[1])
plt.plot(x,y,label="file")
plt.legend()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("File graph")
plt.grid()
plt.show()