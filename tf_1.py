import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # from removing he avx2 error
h=tf.constant("hdf") # dealing with constants
sess=tf.compat.v1.Session()  # creation ogf session
print(sess.run(h)) # for executing operation
# addition
x=tf.constant(2)
y=tf.constant(3)
print(sess.run(x+y))

# matrix rel functions
mat1=tf.fill((3,4),2) #matrix having asame value of all elements
print(sess.run(mat1))

mat2=tf.constant([[1,2,3],[3,4,5]]) #matrix of required values
print(sess.run(mat2))

mat3=tf.zeros((3,4))  # zero matrix
mat4=tf.ones((3,4))   # one matrix
print(sess.run(mat3))
print(sess.run(mat4))

#mat5=tf.random_normal_initliazer((4,4),mean=0,stdev=1.0)
# mat6=tf.random_uniform_initializer((4,4),minval=0,maxval=1)
#print(sess.run(mat5))
#print(sess.run(mat6))

ops=[mat1,mat2,mat3,mat4]  # for executing multiple fun at a time
#for s in ops:
#print(sess.run(s))

# another method for creating sessions
sess1=tf.compat.v1.InteractiveSession()
for s in ops:
     print(s.eval())
