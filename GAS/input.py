# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:48:50 2024

@author: 1
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:47:14 2024

@author: 1
"""
import numpy as np
import pandas as pd
import time 
import random
from tqdm import tqdm
from time import sleep
from sklearn.model_selection import train_test_split
import pickle
from math import sqrt
import pandas as pd
#import descriptor_generator
from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn import svm
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error#, mean_absolute_percentage_error
from matplotlib import pyplot as plt
#from matplotlib import mlab
# from plot_learning_curve import plot_learning_curve_
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ShuffleSplit
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# import sys
# sys.path.append('E:/python/Optimization and Design/result_display')
# from draw import plot
from sklearn.metrics import r2_score, mean_absolute_percentage_error#, mean_absolute_percentage_error
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
# from model_evaluate import evaluation_MLP,violin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# unused = ['Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2']
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.utils.data as Data

def data_type(data_type):
    data = pd.DataFrame(data_type)
    data_ems = [] 
    data = data.loc[~((data['isp'] == 0) | (data['cstar'] == 0))]
    name = ['nHbondA', 'nHbondD', 'nNH2', 'nAHC', 'nACC', 'nHC', 'nRbond', 'nR', 'nNNO2', 'nONO2', 'nNO2', 'nC(NO2)3', 'nC(NO2)2', 'nC(NO2)', 'nH', 'nC', 'nN', 'nO', 'nOCH3', 'nCH3']
    # data[name] = data[name].mul(data['EMs'], axis=0)
    Y = data[['c_t','isp','cstar']]
    X = data[['Al','EMs','HTPB','NH4CLO4','C_mol','H_mol','O_mol','N_mol','Al_mol','Cl_mol','wt_H','nHbondA', 'nHbondD', 'nNH2', 'nAHC', 'nACC', 'nHC', 'nRbond', 'nR', 'nNNO2', 'nONO2', 'nNO2', 'nC(NO2)3', 'nC(NO2)2', 'nC(NO2)', 'MinPartialCharge', 'MaxPartialCharge', 'MOLvolume', 'nH', 'nC', 'nN', 'nO', 'PBF', 'TPSA', 'ob', 'total energy', 'molecular weight', 'PMI3', 'nOCH3', 'nCH3', 'Eccentricity', 'PMI2', 'PMI1', 'NPR1', 'NPR2', 'ESTATE_0', 'ESTATE_1', 'ESTATE_2', 'ESTATE_3', 'ESTATE_4', 'ESTATE_5', 'ESTATE_6', 'ESTATE_7', 'ESTATE_8', 'ESTATE_9', 'ESTATE_10', 'ESTATE_11', 'ESTATE_12', 'ESTATE_13', 'ESTATE_14', 'ESTATE_15', 'ESTATE_16', 'ESTATE_17', 'ESTATE_18', 'ESTATE_19', 'ESTATE_20', 'ESTATE_21', 'ESTATE_22', 'ESTATE_23', 'ESTATE_24', 'ESTATE_25', 'ESTATE_26', 'ESTATE_27', 'ESTATE_28', 'ESTATE_29', 'ESTATE_30', 'ESTATE_31', 'ESTATE_32', 'ESTATE_33', 'ESTATE_34', 'ESTATE_35', 'ESTATE_36', 'ESTATE_37', 'ESTATE_38', 'ESTATE_39', 'ESTATE_40', 'ESTATE_41', 'ESTATE_42', 'ESTATE_43', 'ESTATE_44', 'ESTATE_45', 'ESTATE_46', 'ESTATE_47', 'ESTATE_48', 'ESTATE_49', 'ESTATE_50', 'ESTATE_51', 'ESTATE_52', 'ESTATE_53', 'ESTATE_54', 'ESTATE_55', 'ESTATE_56', 'ESTATE_57', 'ESTATE_58', 'ESTATE_59', 'ESTATE_60', 'ESTATE_61', 'ESTATE_62', 'ESTATE_63', 'ESTATE_64', 'ESTATE_65', 'ESTATE_66', 'ESTATE_67', 'ESTATE_68',
                           'ESTATE_69']] 
    #MOLvolume,PMI2,PMI1
    unused_isp = ['nC(NO2)3','nC(NO2)2','nACC','nONO2','nOCH3','Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2','MOLvolume']
    # unused_isp = ['nACC','nONO2','nOCH3','Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2']
    # unused_cstar = ['Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2','ESTATE_43','ESTATE_12','ESTATE_19','ESTATE_17','ESTATE_4','nONO2','ESTATE_54','ESTATE_24','nC(NO2)3']
    # unused_c_t = ['Eccentricity','nHC','total energy','PMI1','PMI2','ESTATE_54','ESTATE_43','ESTATE_27','ESTATE_9','ESTATE_8','ESTATE_5','ESTATE_35','ESTATE_23','ESTATE_14','ESTATE_19','ESTATE_6']
    # unused = ['Eccentricity','nHC','total energy','PMI1','PMI2','ESTATE_0','ESTATE_58','ESTATE_54','ESTATE_19','ESTATE_17','ESTATE_16','ESTATE_4','nC(NO2)3']
    all_zero = ['ESTATE_3', 'ESTATE_7', 'ESTATE_13', 'ESTATE_15', 'ESTATE_20',
           'ESTATE_26', 'ESTATE_31', 'ESTATE_32', 'ESTATE_33', 'ESTATE_34',
           'ESTATE_38', 'ESTATE_42', 'ESTATE_48', 'ESTATE_50', 'ESTATE_55',
           'ESTATE_61', 'ESTATE_66', 'ESTATE_67', 'ESTATE_68', 'ESTATE_69']
    #unused_isp1 = ['ESTATE_9','ESTATE_62','ESTATE_45','ESTATE_11','ESTATE_6','ESTATE_19','ESTATE_41',
    #'ESTATE_46','ESTATE_54','ESTATE_39','ESTATE_4','ESTATE_1','ESTATE_43','ESTATE_8',
    #'ESTATE_36','ESTATE_16','ESTATE_51','ESTATE_65']
    unused_isp1 = ['ESTATE_36','ESTATE_16','ESTATE_51','ESTATE_65']
    X=X.drop(all_zero,axis=1)
    # unused = []
    X=X.drop(unused_isp,axis=1)
    X=X.drop(unused_isp1,axis=1)
    # X = np.array(X)
    
    # with open('E:/python/wrh1.0/model_train/ss_X.pkl', 'rb') as ex:
    #     normalizing_x = pickle.load(ex)  
    # scaler_S = StandardScaler()
    # with open('E:/python/wrh1.0/model_train/ss_X.pkl', 'wb') as f_ss_X:
        # pickle.dump(scaler_S, f_ss_X)
    # X_Scaler = ss_X.transform(X)E:\python\wrh1.0\model_train
    # X_scaler = normalizing_x.fit_transform(X)
    Y_c_t = np.array(Y['c_t'])
    Y_isp = np.array(Y['isp'])
    Y_cstar = np.array(Y['cstar'])

    return X,Y_c_t,Y_isp,Y_cstar

def datatotensor(X_Scaler,Y_c_t,Y_isp,Y_cstar):
   
    X_S_t = torch.Tensor(X_Scaler.astype(np.float32))    
    Y_c_t_t = torch.Tensor(Y_c_t.astype(np.float32))
    data_c_t_tensor = Data.TensorDataset(X_S_t, Y_c_t_t)

    # X_S_t = torch.Tensor(X_Scaler.astype(np.float32))
    Y_isp_t = torch.Tensor(Y_isp.astype(np.float32))
    data_isp_tensor = Data.TensorDataset(X_S_t, Y_isp_t)
    
    # X_S_t = torch.Tensor(X_Scaler.astype(np.float32))
    Y_cstar_t = torch.Tensor(Y_cstar.astype(np.float32))
    data_cstar_tensor = Data.TensorDataset(X_S_t, Y_cstar_t)
    return data_c_t_tensor,data_isp_tensor,data_cstar_tensor

def datatotensor_(X_Scaler,Y_c_t,Y_isp,Y_cstar):
    # Y = np.array(Y)
    Y_c_t_t = torch.Tensor(Y_c_t.astype(np.float32))
    Y_isp_t = torch.Tensor(Y_isp.astype(np.float32))
    Y_cstar_t = torch.Tensor(Y_cstar.astype(np.float32))
    X_S_t = torch.Tensor(X_Scaler.astype(np.float32))       
    data_tensor = Data.TensorDataset(X_S_t, Y_c_t_t,Y_isp_t,Y_cstar_t)
    return data_tensor
def datatotensor_1(X_Scaler,Y):
    Y = np.array(Y)
    Y = torch.Tensor(Y.astype(np.float32))
    X_S_t = torch.Tensor(X_Scaler.astype(np.float32))       
    data_tensor = Data.TensorDataset(X_S_t, Y)
    return data_tensor