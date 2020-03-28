import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # from removing he avx2 error
# tensorflow graphs
# every process take place in tf in the form of graph internally
x=tf.constant(20)
y=tf.constant(30)
sess=tf.compat.v1.Session()
print(sess.run(x+y))
print(tf.compat.v1.get_default_graph())
g1=tf.Graph()
print(g1)
g1=tf.compat.v1.get_default_graph() # user defined default graph
print(g1)
g2=tf.Graph()
with g2.as_default():
    print(g2 is tf.compat.v1.get_default_graph()) # another default graph  inside another session
print(g2 is tf.compat.v1.get_default_graph())
#----------------------------------------------------------------------------------------------------
#tensor flow variables and placeholders
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
y=tf.Variable(initial_value=x) # creation of variable to store weight and bias
ini=tf.compat.v1.global_variables_initializer()  # global initialization
sess.run(ini)
print(sess.run(y))
print(y)
# placeholder
ph=tf.compat.v1.placeholder(tf.float32)# creation of placeholder to store training data


