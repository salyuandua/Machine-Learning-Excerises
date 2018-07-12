# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:51:35 2018

@author: Zhai Xu
"""
import numpy as np
#load secom.data
def load_data(file_name):
    file=open(file_name);
    stringArr=[line.strip().split(' ') for line in file.readlines()]
    float_arr=np.array(stringArr).astype(np.float)
    return np.mat(float_arr)

#replace nan to mean
def remove_nan(data_mat):
    row_num,col_num=data_mat.shape
    for i in range(col_num):
        #non_nan=~np.isnan(data_mat[:,i].A)
        col_mean=np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:,i].A))[0],i])
        #replace nan value as mean
        print(col_mean)
        data_mat[np.nonzero(np.isnan(data_mat[:,i].A))[0],i]=col_mean
    return data_mat    

def pca(data_mx,top_n):
    mx_mean=np.mean(data_mx)
    
    
    