import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data=np.linspace(0.0,10.0,100000)
noise=np.random.randn(len(x_data))

y_true=(0.5*x_data)+5+noise

x_df=pd.DataFrame(data=x_data,columns=['X_data'])
y_df=pd.DataFrame(data=y_true,columns=['Y_data'])

my_data=pd.concat([x_df,y_df],axis=1)
print(my_data.head(10))

d1=my_data.sample(250)
#plt.scatter(d1["X_data"],d1["Y_data"])
#plt.show()

batch_size=8
m=tf.Variable(0.5)
b=tf.Variable(1.0)

x_ph=tf.compat.v1.placeholder(tf.float32,[batch_size])
y_ph=tf.compat.v1.placeholder(tf.float32,[batch_size])

y_model=m*x_ph+b
error=tf.reduce_sum(tf.square(y_ph-y_model))

optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(error)


init=tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(init)

batches=10000
for i in range(batches):
    rand_int=np.random.randint(len(x_data),size=batch_size)
    feed={x_ph:x_data[rand_int],y_ph:y_true[rand_int]}
    sess.run(train,feed_dict=feed)
    model_m,model_b=sess.run([m,b])
print(model_m,model_b)
y_hat=d1["X_data"]*model_m+model_b
plt.scatter(d1["X_data"],d1["Y_data"])
plt.plot(d1["X_data"],y_hat,"red")
plt.show()