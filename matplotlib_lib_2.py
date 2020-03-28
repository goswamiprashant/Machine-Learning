# to show graph of trign0metrical function
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0.0,2.0,0.1)
y=1+np.cos(2*np.pi*x)
plt.plot(x,y,'--')
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Cosine wave")
plt.grid()
plt.show()