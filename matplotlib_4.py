# to draw a bar graph
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6]
y=[12,34,22,10,45,21]

tick_label=['one','two','three','four','five','six']
plt.bar(x,y,tick_label=tick_label,color=['green','blue'],width=.7)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title('bar-graph')
plt.show()