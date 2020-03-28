import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data=np.linspace(0,10,100)
noise=np.random.randn(100)


y_data=.3*x_data+5+noise
x_df=pd.DataFrame(data=x_data,columns=["x"])
y_df=pd.DataFrame(data=y_data,columns=["y"])

my_data=pd.concat([x_df,y_df],axis=1)
print(my_data)
#plt.scatter(my_data["x"],my_data["y"])
#plt.show()
x_d=my_data.iloc[:,0]
y_d=my_data.iloc[:,1]

#x_train,x_test,y_train,y_test=train_test_split(x_d,y_d,test_size=0.2,random_state=0)
xph=tf.compat.v1.placeholder(tf.float32,[len(x_d)])
yph=tf.compat.v1.placeholder(tf.float32,[len(x_d)])
feed={xph:x_d,yph:y_d}
w=tf.compat.v1.Variable(.3)
b=tf.compat.v1.Variable(5.0)


y_true=w*xph+b

error=tf.reduce_sum(tf.square(yph-y_true))

optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)
result=optimizer.minimize(error)
init=tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(init)
sess.run(result,feed_dict=feed)
fw,fb=sess.run([w,b])
print(fw,fb)
y_final=x_d*fw+fb

print(y_final)
print(y_d)
plt.scatter(x_d,y_d)
plt.plot(x_d,y_final,color="red")
plt.show()