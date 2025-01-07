# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:13:24 2024

@author: 1
"""
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import os

# sys.path.append('E:/python/Optimization and Design/data_generate')

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.scatter(x, y)
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(200, lim + binwidth, binwidth)
    
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def evaluate_model(learner_name, label_name, target_train, predict_train, target_test, predict_test, save_dir):
    
    target_train = target_train.detach().cpu().numpy()
    #target_train = target_train.detach().numpy()
    predict_train = predict_train.detach().cpu().numpy()
    target_test = target_test.detach().cpu().numpy()
    predict_test = predict_test.detach().cpu().numpy()
    print(target_test[1]-predict_test[1])

    for data, name in zip([(predict_test, 'test'), (target_test, 'test')], ['predict', 'target']):
        pd.DataFrame(data[0]).to_csv(os.path.join(save_dir, f'{learner_name}_{label_name}_{name}.csv'), index=False)
    

    metrics = {
        'mean_absolute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score
    }

    with open(os.path.join(save_dir, f'{learner_name}_{label_name}.txt'), 'w', encoding='utf-8') as f:
        for metric_name, metric_func in metrics.items():
        
            f.write(f"{metric_name}_训练集: {metric_func(target_train, predict_train)}\n")
     
            f.write(f"{metric_name}_测试集: {metric_func(target_test, predict_test)}\n")


def evaluation_MLP(learner_name, label_name, target_train, predict_train, target_test, predict_test, save_dir):
    evaluate_model(learner_name, label_name, target_train, predict_train, target_test, predict_test, save_dir)


def violin_plot(label_name, target_train, target_val, target_test, save_dir):

    target_train = target_train.detach().numpy()
    target_val = target_val.detach().numpy()
    target_test = target_test.detach().numpy()

    for data, name in zip([(target_train, 'train'), (target_val, 'val'), (target_test, 'test')], ['train', 'val', 'test']):
        pd.DataFrame(data[0]).to_csv(os.path.join(save_dir, f'{label_name}_{name}.csv'))

def hexbin_plot(learner_name, label_name, y_train, predict_train, y_test, predict_test, save_dir):
    font = {'family': 'Arial', 'weight': 'bold', 'size': 20}
    plt.rcParams["figure.figsize"] = [3, 3]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    ax.hexbin(y_train, predict_train, gridsize=1)
    ax.hexbin(y_test, predict_test, gridsize=1)
    
    plt.grid(linestyle='--', linewidth=2, color='gray', alpha=0.4)
    
    x = np.linspace(min(predict_train), max(predict_train), 5)
    plt.plot(x, x, 'r--', linewidth=2.0)
    
    plt.xlabel('Observation', fontdict=font)
    plt.ylabel('Prediction', fontdict=font)
    plt.tick_params(width=3, length=5)
    plt.xticks(family='Arial', size=20, weight='bold')
    plt.yticks(family='Arial', size=20, weight='bold')
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label_name}_{learner_name}.tif'), dpi=300)
    plt.show()
