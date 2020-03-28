import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.examples.tutorials.mnist as input_data
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow_core.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)