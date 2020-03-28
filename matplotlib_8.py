#merging two plot in a graph
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,2*np.pi,0.1)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,label='Sin')
plt.plot(x,y2,label='Cos')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.title("Merge graph")
plt.show()