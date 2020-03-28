import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf1
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

diabetes=pd.read_csv("diabetes.csv")
print(diabetes.head());
print(diabetes.columns);
cols_to_norm=["Number_pregnant","Glucose_concentration","Blood_pressure","Triceps","Insulin","BMI","Pedigree"]
diabetes[cols_to_norm]=diabetes[cols_to_norm].apply(lambda x:((x-x.min())/(x.max()-x.min())))
print(diabetes.info())
print(diabetes.head())
import tensorflow as tf
Number_pregnant=tf.feature_column.numeric_column("Number_pregnant")
Glucose_concentration=tf.feature_column.numeric_column("Glucose_concentration")
Blood_pressure=tf.feature_column.numeric_column("Blood_pressure")
Triceps=tf.feature_column.numeric_column("Triceps")
Insulin=tf.feature_column.numeric_column("Insulin")
BMI=tf.feature_column.numeric_column("BMI")
Pedigree=tf.feature_column.numeric_column("Pedigree")
age=tf.feature_column.numeric_column("Age")
assigned_group=tf.feature_column.categorical_column_with_vocabulary_list("Group",['A','B','C','D','E'])
age_bucket=tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

feat_cols=[Number_pregnant,Glucose_concentration,Blood_pressure,Triceps,Insulin,BMI,Pedigree,age_bucket,assigned_group]
x_data=diabetes.drop('Class',axis=1)
labels=diabetes["Class"]
print(x_data.head());

#train test split
x_train,x_test,y_train,y_test=train_test_split(x_data,labels,test_size=.20,random_state=0)

#training model
#input_func=tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model=tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)

model.train(input_fn=input_func,steps=1000)

#evaluation of model
eval_input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10,num_epochs=1000,shuffle=False)
results=model.evaluate(eval_input_func)

pred_input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1000,shuffle=False)
predictions=model.predict(pred_input_func)
my_pred=list(predictions)
























