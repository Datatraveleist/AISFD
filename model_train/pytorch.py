# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:54:58 2024

@author: 1
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
# import optuna
from tqdm import tqdm
from early_stopping import EarlyStopping
from input import data_type, datatotensor
from MLP_frame import Model_isp_all,Model_cstar_all,Model_c_t_all
import pandas as pd


df_train = pd.read_csv('data_train.csv')
df_test = pd.read_csv('data_test.csv')

X_train, Y_c_t_train, Y_isp_train, Y_cstar_train = data_type(df_train)
X_test, Y_c_t_test, Y_isp_test, Y_cstar_test = data_type(df_test)

std = StandardScaler()
X_train_s = std.fit_transform(X_train)
X_test_s = std.transform(X_test)
#with open('isp_X.pkl', 'wb') as ex:
#   pickle.dump(std, ex)
data_c_t_tensor_train, data_isp_tensor_train, data_cstar_tensor_train = datatotensor(X_train_s, Y_c_t_train, Y_isp_train, Y_cstar_train)
data_c_t_tensor_test, data_isp_tensor_test, data_cstar_tensor_test = datatotensor(X_test_s, Y_c_t_test, Y_isp_test, Y_cstar_test)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate(trial, model_class, data_train, data_test, target_name):
    #lr = trial.suggest_loguniform('lr', 1e-9, 1e-7)  
    lr_dict = {'isp':1e-5,'c_t':1e-7,'cstar':1e-6}
    lr = lr_dict[target_name]

    #momentum = trial.suggest_uniform('momentum', 0.1, 0.5)  
    momentum = 0.2
    num_epochs = trial.suggest_int('num_epochs', 500, 1000)  

    layer_sizes = [trial.suggest_int(f"layer_size_{i}", 32, 256, step=32) for i in range(trial.suggest_int('num_layers', 12, 14))]


    model = model_class(layer_sizes).to(device)
    criterion = nn.MSELoss().to(device)  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    min_eval_loss = float('inf')  

    with open(f'hyperparameters_{target_name}.txt', 'a') as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"lr: {lr}\nmomentum: {momentum}\nnum_epochs: {num_epochs}\n")
        f.write(f"num_layers: {len(layer_sizes)}\n")
        for i, layer_size in enumerate(layer_sizes):
            f.write(f"layer_size_{i}: {layer_size}\n")
        f.write("-" * 50 + "\n")


    train_loader = Data.DataLoader(dataset=data_train, batch_size=64, shuffle=True, num_workers=24)
    test_loader = Data.DataLoader(dataset=data_test, batch_size=64, shuffle=True, num_workers=24)

 
    early_stopping = EarlyStopping('/home/shuoxing/wrh/AI_formulation_design', target_name)

    # print(f'{target_name}')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for X, Y_MLP in train_bar:
            out = model(X.to(device))
            loss = criterion(out, Y_MLP.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(train_loss=train_loss / (train_bar.n + 1))

      
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Evaluating', leave=False)
            for X_test, yy in test_bar:
                out = model(X_test.to(device))
                loss = criterion(out, yy.to(device))
                eval_loss += loss.item()
                test_bar.set_postfix(eval_loss=eval_loss / (test_bar.n + 1))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            with open(f'hyperparameters_{target_name}.txt', 'a') as f:
                f.write(f"Epoch {epoch + 1}/{num_epochs} - Min Eval Loss: {min_eval_loss / len(test_loader)}\n")
        early_stopping(eval_loss / len(test_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return eval_loss / len(test_loader)

def objective_isp(trial):
    return train_and_evaluate(trial, Model_isp_all, data_isp_tensor_train, data_isp_tensor_test, 'isp')

def objective_cstar(trial):
    return train_and_evaluate(trial, Model_cstar_all, data_cstar_tensor_train, data_cstar_tensor_test, 'cstar')

def objective_c_t(trial):
    return train_and_evaluate(trial, Model_c_t_all, data_c_t_tensor_train, data_c_t_tensor_test, 'c_t')


#study = optuna.create_study(direction='minimize') 
#study.optimize(objective_cstar, n_trials=5)  
#study.optimize(objective_c_t, n_trials=5) 

#print(f"Best hyperparameters: {study.best_params}")
#print(f"Best value: {study.best_value}")

def train_isp():
    #model_isp = Model_isp_all([160,64,160,160,96,256,64,256,192,96,192,192,224]).to(device) 
    model_isp = Model_isp_all([832,704,1024,288,64,704,384,1024]).to(device) 
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model_isp.parameters(), lr=1e-5, momentum=0.2)
    train_loader = Data.DataLoader(dataset=data_isp_tensor_train, batch_size=64, shuffle=True, num_workers=24)
    test_loader = Data.DataLoader(dataset=data_isp_tensor_test, batch_size=64, shuffle=True, num_workers=24)

    early_stopping = EarlyStopping('../prediction', 'isp')  # 保存路径和模型名称
    num_epochs = 500

    for epoch in range(num_epochs):
        model_isp.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        
        for X, Y_MLP in train_bar:
            #formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]

            #formulation_x, ems_x, Y_MLP = formulation_x.to(device), ems_x.to(device), Y_MLP.to(device)

            out = model_isp(X.to(device))
            loss = criterion(out, Y_MLP.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_bar.set_postfix(train_loss=train_loss / (train_bar.n + 1))

        model_isp.eval()
        eval_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Evaluating', leave=False)
            for X_test, yy in test_bar:
                #formulation_x_test, ems_x_test = X_test[:, 0:10], X_test[:, 10:-1]

                #formulation_x_test, ems_x_test, yy = formulation_x_test.to(device), ems_x_test.to(device), yy.to(device)

                out = model_isp(X_test.to(device))
                loss = criterion(out, yy.to(device))
                eval_loss += loss.item()
                test_bar.set_postfix(eval_loss=eval_loss / (test_bar.n + 1))
        with open('hyperparameters_and_results.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs} - Eval Loss: {eval_loss / len(test_loader)}\n")
        early_stopping(eval_loss / len(test_loader), model_isp)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def train_cstar():
    Model_cstar = Model_cstar_all([160,64,160,160,96,256,64,256,192,96,192,192,224]).to(device) 
    #model_isp = Model_isp_all([2000,1028,256, 128, 64,32, 16, 8]).to(device) 

    criterion = nn.MSELoss().to(device) 
    optimizer = torch.optim.SGD(Model_cstar.parameters(), lr=1e-6, momentum=0.2)

    train_loader = Data.DataLoader(dataset=data_cstar_tensor_train, batch_size=64, shuffle=True, num_workers=24)
    test_loader = Data.DataLoader(dataset=data_cstar_tensor_test, batch_size=64, shuffle=True, num_workers=24)

    early_stopping = EarlyStopping('../prediction', 'cstar')  
    num_epochs = 677
    for epoch in range(num_epochs):
        Model_cstar.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        
        for X, Y_MLP in train_bar:
            #formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]
            #formulation_x, ems_x, Y_MLP = formulation_x.to(device), ems_x.to(device), Y_MLP.to(device)

            out = Model_cstar(X.to(device))
            loss = criterion(out, Y_MLP.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

            train_bar.set_postfix(train_loss=train_loss / (train_bar.n + 1))


        Model_cstar.eval()
        eval_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Evaluating', leave=False)
            for X_test, yy in test_bar:
                #formulation_x_test, ems_x_test = X_test[:, 0:10], X_test[:, 10:-1]

                #formulation_x_test, ems_x_test, yy = formulation_x_test.to(device), ems_x_test.to(device), yy.to(device)

                out = Model_cstar(X_test.to(device))
                loss = criterion(out, yy.to(device))
                eval_loss += loss.item()
                test_bar.set_postfix(eval_loss=eval_loss / (test_bar.n + 1))
        with open('hyperparameters_and_results.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs} - Eval Loss: {eval_loss / len(test_loader)}\n")
        early_stopping(eval_loss / len(test_loader), Model_cstar)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
def train_c_t():
    Model_c_t = Model_c_t_all([160,128,192,256,128,224,192,224,192,128,128,160,32,192]).to(device) 
    #model_isp = Model_isp_all([2000,1028,256, 128, 64,32, 16, 8]).to(device) 

    criterion = nn.MSELoss().to(device)  
    optimizer = torch.optim.SGD(Model_c_t.parameters(), lr=1e-7, momentum=0.2)

    train_loader = Data.DataLoader(dataset=data_c_t_tensor_train, batch_size=64, shuffle=True, num_workers=24)
    test_loader = Data.DataLoader(dataset=data_c_t_tensor_test, batch_size=64, shuffle=True, num_workers=24)

    early_stopping = EarlyStopping('../prediction', 'c_t')  
    num_epochs = 582
    for epoch in range(num_epochs):
        Model_c_t.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        
        for X, Y_MLP in train_bar:
            #formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]

            #formulation_x, ems_x, Y_MLP = formulation_x.to(device), ems_x.to(device), Y_MLP.to(device)

            out = Model_c_t(X.to(device))
            loss = criterion(out, Y_MLP.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(train_loss=train_loss / (train_bar.n + 1))

        Model_c_t.eval()
        eval_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Evaluating', leave=False)
            for X_test, yy in test_bar:
                #formulation_x_test, ems_x_test = X_test[:, 0:10], X_test[:, 10:-1]
                #formulation_x_test, ems_x_test, yy = formulation_x_test.to(device), ems_x_test.to(device), yy.to(device)

                out = Model_c_t(X_test.to(device))
                loss = criterion(out, yy.to(device))
                eval_loss += loss.item()
                test_bar.set_postfix(eval_loss=eval_loss / (test_bar.n + 1))
        with open('hyperparameters_and_results.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs} - Eval Loss: {eval_loss / len(test_loader)}\n")
        early_stopping(eval_loss / len(test_loader), Model_c_t)
        if early_stopping.early_stop:
            print("Early stopping")
            break
     
#train_cstar()
#train_c_t()
#train_isp()