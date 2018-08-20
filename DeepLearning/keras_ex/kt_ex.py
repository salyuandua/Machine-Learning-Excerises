# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:02:00 2018

@author: Xuejian Li
"""
import numpy as np
import h5py
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.applications.vgg16 import preprocess_input
import pydot
from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot
import tensorflow.keras.utils.vis_utils.model_to_dot
from tensorflow.keras.utils import plot_model

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow








def load_data():
    train_data_set=h5py.File("datasets/train_happy.h5","r")
    #get training data set
    train_set_X=np.array(train_data_set["train_set_x"][:])
    train_set_Y=np.array(train_data_set["train_set_y"][:])
    #get test data set
    test_data_set=h5py.File("datasets/test_happy.h5","r")
    test_set_X=np.array(test_data_set["test_set_x"][:])
    test_set_Y=np.array(test_data_set["test_set_y"][:])
    #get class list in test set
    classes=np.array(test_data_set["list_classes"][:])
    #reshape
    train_set_Y=train_set_Y.reshape((1,train_set_X.shape[0]))
    test_set_Y=test_set_Y.reshape((1,test_set_Y.shape[0]))
    return train_set_X,train_set_Y,test_set_X,test_set_Y,classes

#load data
train_set_X,train_set_Y,test_set_X,test_set_Y,classes=load_data()    
#normalize images
train_set_X=train_set_X/255
test_set_X=test_set_X/255
#reshape
train_set_Y=train_set_Y.T
test_set_Y=test_set_Y.T


#construct model

def get_model(input_x_shape):
    #input_shape=train_set_X.shape
    input_X=layers.Input(input_x_shape)
    #padding
    padding=layers.ZeroPadding2D(padding=(1,1))
    X=padding(input_X)
    #-----------------------conv layer1
    layer1=layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),name='layer1')
    X=layer1(X)
    X=layers.BatchNormalization(axis=3,name='bn1')(X)
    X=layers.Activation('relu',name='act1')(X)
    X=layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool1')
    #-----------------------conv layer2
    #padding
    X=layers.ZeroPadding2D(padding=(1,1))(X)
    X=layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),name='layer2')(X)
    X=layers.BatchNormalization(axis=3,name='bn2')(X)
    X=layers.Activation('relu',name='act2')(X)
    X=layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool2')(X)
    #------------------------conv layer3
    X=layers.ZeroPadding2D(padding=(1,1))(X)
    X=layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),name='layer3')(X)
    X=layers.BatchNormalization(axis=3,name='bn3')(X)
    X=layers.Activation('relu',name='act3')(X)
    X=layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool3')(X)
    #------------------------fully connected layer
    X=layers.Flatten()(X)
    X=layers.Dense(units=1,activation='sigmod',name='fully_connected')(X)
    #create model
    my_model=models.Model(inputs=input_X,outputs=X,name='HappyModel')
    return my_model



























    
    
    
    
    
    
    




