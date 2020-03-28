import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#----------------first perceptron creation-----------------------------------
np.random.seed(101)
tf1.set_random_seed(101)

# training data creation
rand_a=np.random.uniform(0,100,(5,5))
rand_b=np.random.uniform(0,100,(5,1))
#Storing in placeholder
a=tf1.placeholder(tf.float32)
b=tf1.placeholder(tf.float32)

add=a+b
mul=a*b

sess=tf1.Session()
add_r=sess.run(add,feed_dict={a:rand_a,b:rand_b})
mul_r=sess.run(mul,feed_dict={a:rand_a,b:rand_b})
#print(add_r,"\n",mul_r)






#----------------first neural network creation-----------------------------------

n_f=10
n_d_n=3

# training data
x=tf1.placeholder(tf.float64,shape=(None,n_f))
print(x)
# bias value
b=tf.Variable(np.zeros([n_d_n])) # with tf.zeroes it becomes of type float 32 so np is used
#weights
w=tf.Variable(np.random.random([n_f,n_d_n]))
init=tf.compat.v1.global_variables_initializer()
sess=tf1.Session()
sess.run(init)
print(sess.run(b))

xw=tf.matmul(x,w)
z=tf.add(xw,b)

#Activation function
a=tf.sigmoid(z)
print((w))
out=sess.run(a,feed_dict={x:np.random.random([1,n_f])})
print(out)