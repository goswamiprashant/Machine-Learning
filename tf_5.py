import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# creation of large dataset
x_data=np.linspace(0.0,10,1000000)
noise=np.random.randn(len(x_data))

# y=m*x +b  b=5,m=0.5
y_true=(0.5*x_data)+ 5+noise
# making dataframes
x_df=pd.DataFrame(data=x_data,columns=['X_Data'])
y_df=pd.DataFrame(data=y_true,columns=['Y'])
# data set created
my_data=pd.concat([x_df,y_df],axis=1)
print(my_data.head())
# taking samples of data
my_data.sample(n=250).plot(kind='scatter',x='X_Data',y='Y')

batch_size=8
m=tf.compat.v1.Variable(0.5)  # slope or weight
b=tf.compat.v1.Variable(1.0)  # intercept or bias

xph=tf.compat.v1.placeholder(tf.float32,[batch_size])
yph=tf.compat.v1.placeholder(tf.float32,[batch_size])

# model
y_model=m*xph +b
# cost function
error=tf.reduce_sum(tf.square(yph-y_model))
# optimization
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(error)
init=tf.compat.v1.global_variables_initializer()

sess=tf.compat.v1.Session()
sess.run(init)

batches=10000
for i in range(batches):
    rand_int=np.random.randint(len(x_data),size=batch_size)
    feed={xph:x_data[rand_int],yph:y_true[rand_int]}
    sess.run(train,feed_dict=feed)
    model_m,model_b=sess.run([m,b])  # final m and b
print(model_m,model_b)
y_hat=x_data*model_m+model_b
my_data.sample(n=250).plot(kind='scatter',x='X_Data',y='Y')
plt.plot(x_data,y_hat,'red')
plt.show()