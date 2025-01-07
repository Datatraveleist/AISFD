# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:47:14 2024

@author: 1
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hyperparameter import estimator
from input import data_type

df_train = pd.read_csv('../model_train/data_train.csv')
df_test = pd.read_csv('../model_train/data_test.csv')
    
X_train, Y_c_t_train, Y_isp_train, Y_cstar_train = data_type(df_train)
X_test, Y_c_t_test, Y_isp_test, Y_cstar_test = data_type(df_test)
    
std = StandardScaler()
X_train_s = std.fit_transform(X_train)
X_test_s = std.transform(X_test)

parameters = ['c_t','isp','cstar']
models_names = ['adaboost','rr','knn']

data_train = {'c_t':Y_c_t_train,'isp':Y_isp_train,'cstar':Y_cstar_train}
data_train_X = {'c_t':X_train_s,'isp':X_train_s,'cstar':X_train_s}
data_test = {'c_t':Y_c_t_test,'isp':Y_isp_test,'cstar':Y_cstar_test}
data_test_X = {'c_t':X_test_s,'isp':X_test_s,'cstar':X_test_s}
for parameter in parameters: 
    for models_name in models_names:
    	estimator(models_name,parameter,data_train_X[parameter],data_train[parameter],data_test_X[parameter],data_test[parameter])    


