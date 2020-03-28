# Simple Regression Example
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#creation of input data
x_data=np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
#output
y_label=np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
# graph of data points

# slope and intercept(weight and bias )
np.random.rand(2)
m=tf.Variable(.94)
b=tf.Variable(.69)

# cost function
error=0
for x,y in zip(x_data,y_label):
    y_hat=m*x+b
    error+=(y-y_hat)**2

# error optimization
optimizer=tf1.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(error)

init=tf1.global_variables_initializer()
sess=tf1.Session()
sess.run(init)
epochs=10000
for i in range(epochs):
    sess.run(train)
f_slope,f_intercept=sess.run([m,b])
print(f_slope,f_intercept)

x_test=np.linspace(-1,11,10)
y_pred=f_slope* x_test+f_intercept
plt.plot(x_test,y_pred,'green')
plt.scatter(x_data,y_label)
plt.show()

