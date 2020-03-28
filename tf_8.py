#house pricing prediction
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf1
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

housing=pd.read_csv("housing.csv")
#print(housing.head())
#print(housing.describe().transpose()) #for getting count ,mean min ,max
print(housing.columns)

x_data=housing.drop('medianHouseValue',axis=1)
y_data=housing["medianHouseValue"]
print(x_data.head())
print(y_data.head())

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=101)
#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)

#conversion to dataframe
x_train=pd.DataFrame(data=scaler.transform(x_train),columns=x_train.columns,index=x_train.index)
x_test=pd.DataFrame(data=scaler.transform(x_test),columns=x_test.columns,index=x_test.index)

#print(x_train.head())

# converting into numerical features
housingMedianAge=tf.feature_column.numeric_column("housingMedianAge")
totalRooms=tf.feature_column.numeric_column("totalRooms")
totalBedrooms=tf.feature_column.numeric_column("totalBedrooms")
population=tf.feature_column.numeric_column("population")
households=tf.feature_column.numeric_column("households")
medianIncome=tf.feature_column.numeric_column("medianIncome")

feat_cols=[housingMedianAge, totalRooms, totalBedrooms, population,
       households, medianIncome]

#training of data
input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
#creating a model
model=tf.compat.v1.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)
#traning of model
model.train(input_fn=input_func,steps=25)
pred_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)
pred=model.predict(pred_func)
print(pred)
final_pred=[]
for p in pred:
    final_pred.append(p)
#print(final_pred)
# for calculating root mean square error
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,tf.cast(final_pred,tf.float32)**.5))



feat_cols=[age, workclass, education, education_num, marital_status,occupation, relationship, gender, capital_gain,capital_loss, hours_per_week, native_country]
# input
input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x_train,y_train,num_epochs=None,shuffle=True,batch_size=100)

# creation of model
model=tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols)
# training of model
model.train(input_fn=input_func,steps=10000)
#predictions
pred_input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)
pred=list(model.predict(input_func=pred_input_func))
print(pred)










































