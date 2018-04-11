# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 07:13:31 2018

@author: Xuejian Li
"""
import numpy as np
import matplotlib.pyplot as plot

def load_data(file_path):
    file=open(file_path)
    data_mat=[]
    label_mat=[]
    for line in file:
        paramters=line.split('\t')
        data_mat.append([float(paramters[0]),float(paramters[1])])
        label_mat.append(float(paramters[2]))
    return data_mat,label_mat

def sigmod(thetaX):
    return 1/(1+np.exp(-thetaX))

def gradient_ascent(X,Y,alph,iter_num):
    X_mat=np.mat(X)
    Y_mat=np.mat(Y).transpose()
    m,n=X_mat.shape
    theta=np.ones((n,1))
    for k in range(iter_num):
        h=sigmod(X_mat*theta)
        error=Y_mat-h;
        theta=alph*(X_mat.transpose()*error)
    return theta

def plot_points(X,Y):
    fig=plot.figure()
    axe=fig.add_subplot(1,1,1)
    
    


