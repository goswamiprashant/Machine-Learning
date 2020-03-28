import numpy as np
from sklearn import preprocessing
x=([2.1, -1.9, 5.5],
       [-1.5, 2.4, 3.5],
       [0.5, -7.9, 5.6],
       [5.9, 2.3, -5.8])
Input_data = np.array(x)
data_binarized=preprocessing.Binarizer(threshold=0.5).transform(Input_data)
print("\nBInarized data:\n",data_binarized)
#-----------------------binarization--------------------------
print("Mean=",Input_data.mean(axis=0))
print("Std deviation=",Input_data.std(axis=0))
#-----------------------Mean Removal---------------------------
data_scaled=preprocessing.scale(Input_data)
print("Mean=",data_scaled.mean(axis=0))
print("Std deviation=",data_scaled.std(axis=0))
#---------min max scaling---------------------------------------
data_scalar_minimax=preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minimax=data_scalar_minimax.fit_transform(Input_data)
print("\nMin max scaled data:\n",data_scaled_minimax)
#-------------Scaling---------------------------------------------
data_normalized_l1=preprocessing.normalize(Input_data,norm='l1')
print("\n L1 normalized data:\n",data_normalized_l1)
#--least absolute deviation----------------------------------
data_normalized_l2=preprocessing.normalize(Input_data,norm='l2')
print("\n L2 normalized data:\n",data_normalized_l2)
#---------least square------------------------------------------
#---------------Normalization-------------------------------------

#-------------------Label encoding Steps--------------------------
input_labels=['red ','black','red','green','black','yellow','white']
encoder=preprocessing.LabelEncoder()
encoder.fit(input_labels)
test_labels=['green','red','black']
encoded_values=encoder.transform(test_labels)
print("\n Labels=",test_labels)
print("Encoded_values",encoded_values)
#-----decoding steps---------------------------------------------------
encoded_values=[3,4,0,1]
decoded_list=encoder.inverse_transform(encoded_values)
print("\nEncoded_values=",encoded_values)
print("\n Decoded values=",decoded_list)
