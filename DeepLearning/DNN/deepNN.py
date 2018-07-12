# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:26:45 2018

@author: Xuejian Li
"""

import numpy as np
import h5py
from testCases_v4 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

np.random.seed(1)


def initialize_paramters(dims):
    np.random.seed(3)
    L=len(dims)
    paramters={}
    for l in range(1,L):
       W=np.random.randn(dims[l],dims[l-1])*0.01
       b=np.random.randn(dims[l],1)
       paramters['W'+str(l)]=W
       paramters['b'+str(l)]=b
       
    return paramters   

#test initialize_paramters
paramters1=initialize_paramters([4,3,2])    

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    cache=[A,W,b]
    return Z,cache

#test linear_forward
A,W,b=linear_forward_test_case()
Z,cache=linear_forward(A,W,b)    


def linear_act_forward(prev_A,W,b,act_func_name):
    Z,linear_cache=linear_forward(prev_A,W,b)
    A=None
    act_cache=None
    if act_func_name=='relu':
       
        A,act_cache=relu(Z)
    elif act_func_name=='sigmoid':
        A,act_cache=sigmoid(Z)

    cache=(linear_cache,act_cache)
    return A,cache
#test linear_act_forward
prev_A,W,b=linear_activation_forward_test_case()
A,cache=linear_act_forward(prev_A,W,b,'relu')
print('relu A:'+str(A))
A,cache=linear_act_forward(prev_A,W,b,'sigmoid')
print('sogmod A:'+str(A))

def L_model_forward(X,paramters):
    A=X
    L=len(paramters)//2
    caches=[]
    for l in range(1,L):
        pre_A=A
        A,cache=linear_act_forward(pre_A,paramters['W'+str(l)],paramters['b'+str(l)],'relu')
        caches.append(cache)
    AL,cache=linear_act_forward(A,paramters['W'+str(L)],paramters['b'+str(L)],'sigmoid')
    caches.append(cache)    
    return AL,caches


#test L_model_forward
X,paramters=L_model_forward_test_case_2hidden()
AL,caches=L_model_forward(X,paramters)



    #cross entrpy cost
def cost(AL,Y):
    m=Y.shape[1]
    cost=-(1/m)*np.sum(Y*np.log(AL)+(1-Y)*(1-np.log(AL)))
    return cost
#test cost
Y,AL=compute_cost_test_case()    
cost=cost(AL,Y)    
    
def linear_backward(dZ,cache):
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=(1/m)*np.dot(dZ,np.transpose(A_prev))
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(np.transpose(W),dZ)
    return dW,db,dA_prev

#test linear_backward
dZ,linear_cache=linear_backward_test_case()
dW,db,dA_prev=linear_backward(dZ,linear_cache)

def linear_act_backward(dA,cache,act_name):
    linear_cache,act_cache=cache
    dZ,dA_prev,dW,db=None
    if act_name=='relu':
        dZ=relu_backward(dA,act_cache)
        dW,db,dA_prev=linear_backward(dZ,linear_cache)
    elif act_name=='sigmoid':
        dZ=sigmoid_backward(dA,act_cache)
        dW,db,dA_prev=linear_backward(dZ,linear_cache)
    return dA_prev,dW,db


       
def L_model_backward(AL,Y,caches):
     grads={}
     L=len(caches)/2
     m=AL.shape[1]
     Y=Y.reshape(AL.shape[1])
     
        
        
        
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
