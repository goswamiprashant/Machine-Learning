#Salary Slab Predcition
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf1
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

census=pd.read_csv("salary_data.csv")
print(census.head())
print(census['income_bracket'].unique())
def label_fix(label):
    if(label==' <=50K'):
        return 0
    else:
        return  1
census["income_bracket"]=census["income_bracket"].apply(label_fix)
print(census["income_bracket"].head())
x_data=census.drop("income_bracket",axis=1)

y_data=census["income_bracket"]

# dividing data
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=101)

# convertin gto feature columns from categorical and neumeric data
print(x_data.columns)
print(x_data.info())
age=tf.feature_column.numeric_column("age")
#education_num =tf.feature_column.numeric_column("education_num ")
#capital_gain =tf.feature_column.numeric_column("capital_gain ")
capital_loss=tf.feature_column.numeric_column("capital_loss")
hours_per_week=tf.feature_column.numeric_column("hours_per_week")

#gender=tf.feature_column.categorical_column_with_vocabulary_list("gender",["Male","Female"])
gender=tf.feature_column.categorical_column_with_hash_bucket("gender",10)
workclass=tf.feature_column.categorical_column_with_hash_bucket("workclass",10)
education =tf.feature_column.categorical_column_with_hash_bucket("education ",10)
marital_status =tf.feature_column.categorical_column_with_hash_bucket("marital_status ",10)
occupation=tf.feature_column.categorical_column_with_hash_bucket("occupation",10)
relationship =tf.feature_column.categorical_column_with_hash_bucket("relationship ",10)
native_country =tf.feature_column.categorical_column_with_hash_bucket("native_country ",10)

feat_cols=[age, workclass]
# input
input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

# creation of model
model=tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=10)
pred_input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)
pred=list(model.predict(pred_input_func))
print(pred[["class_ids"][0]])
final_pred=[]
#for p in pred:
 #   final_pred.append(pred["class_ids"][0])
#from sklearn.metrics import classification_report
#print(classification_report(y_test,final_pred))








