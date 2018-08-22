# -*- coding: utf-8 -*-
"""
A simple implementation of resdiual convnet with Keras

Created on Tue Aug 21 22:07:16 2018

@author: Xuejian Li
"""
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras import initializers
from tensorflow.keras.applications.vgg16 import preprocess_input
import pydot
from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot
#import tensorflow.keras.utils.vis_utils.model_to_dot
from tensorflow.keras.utils import plot_model

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def identity_block(X, f, filters, stage, block):
    #names of layers
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut=X
    # First component of main path
    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)
    #second component
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    #3rd component
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    X=layers.add([X,X_shortcut])
    
    X = layers.Activation('relu')(X)
    return X
#test identity_block
tf.reset_default_graph()
with tf.Session() as session:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    print(A_prev)
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    session.run(tf.global_variables_initializer())
    out = session.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0])) 
    
    



