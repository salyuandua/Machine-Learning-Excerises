# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:26:07 2018

@author: l1876
"""
import numpy as np
def get_data():
    return np.mat([[1,1,1,0,0],
                   [2,2,2,0,0],
                   [1,1,1,0,0],
                   [5,5,5,0,0],
                   [1,1,0,2,3],
                   [0,0,0,3,3],
                   [0,0,0,1,1]]);
    
    
def sim(a,b,alg='pearson'):
    if alg=='cos':
        return 1.0/(1.0+np.linalg.norm(a-b))
    elif alg=='pearson':
        
