# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:04:16 2018

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

def normal_equation(x,y):
    X=np.mat(x)
    Y=np.mat(y).T
    XtX=X.T*X
    if np.linalg.det(XtX)==0.0:
        print('error')
    theta=XtX.I*X.T*Y
    return theta    
def plot_points_line(x,y,theta):
    X=np.mat(x)
    Y=np.mat(y)
    fig=plot.figure()
    ax=fig.add_subplot(1,1,1)
    #plot points
    ax.scatter(X[:,1].flatten().A[0],Y.T[:,0].flatten().A[0])
    #plot line
    X_copy=X.copy()
    X_copy.sort(0)
    Y_hat=X_copy*theta
    ax.plot(X_copy[:,1],Y_hat)
    plot.show()

#locally weighted lr
def lwlr():
        

    
    
    
    

        
    