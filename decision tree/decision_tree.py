# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:54:41 2018

@author: Xuejian Li
"""
import numpy as np
from sklearn import tree
X=np.array([['sales','31-35','46k-50k'],['sales','26-30','26k-30k'],['sales','31-35','31k-35k'],
['systems','21-25','46k-50k'],['systems','31-35','66k-70k'],['systems','26-30','46k-50k'],
['systems','41-45','66k-70k'],['marketing','36-40','46k-50k'],['marketing','31-35','41k-45k'],
['secretary','46-50','36k-40k'],['secretary','26-30','26k-30k']]);
Y=np.array(['senior','junior','junior','junior','senior','senior','senior','senior','junior',
'senior','junior']);

#question 1
def entropy(X,target_attr_indx):
    X_dict={};
    for entry in X:
        if entry[target_attr_indx] in X_dict:
            X_dict[entry[target_attr_indx]]+=1
        else:
            X_dict[entry[target_attr_indx]]=1
            
    #compute entropy
    entropy=0
    m=len(X)
    for entry in X_dict:
        entropy+=(X_dict[entry]/m*np.log2(X_dict[entry]/m))
        
         
    return -1*entropy
#compute information gain
def info_gain(X,target_attr_indx):
    #compute entropy
    sub_entropy=0
    X_dict={}
    for entry in X:
        if entry[target_attr_indx] in X_dict:
            X_dict[entry[target_attr_indx]]+=1
        else:
            X_dict[entry[target_attr_indx]]=1
            
    for entry in X_dict:
        entry_probab=X_dict[entry]/sum(X_dict.values())
        subdata=[i for i in X if i[target_attr_indx] == entry]
        sub_entropy+=entry_probab*entropy(subdata,target_attr_indx)
    X_entropy=entropy(X,target_attr_indx) 
    return X_entropy-sub_entropy




info_gain=info_gain(X,2)
#ent=entropy(X,0)






