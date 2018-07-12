# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:07:06 2018

@author: Xuejian Li
"""
import numpy as np
import pandas as pd

#read data

    file=open('student-por.csv')
    data_mat=[]
    label_mat=[]
    for line in file:
        paramters=line.split(';')
        data_mat.append(paramters)



    
