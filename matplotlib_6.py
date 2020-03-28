# to draw a scatter graph
import  matplotlib.pyplot as plt
x=[2,3,5,1,7,4]
y=[2,4,10,13,5,16]
plt.scatter(x,y,marker='*',label="stars",color="red")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Scatter graph")
plt.grid()
plt.legend()
plt.show()