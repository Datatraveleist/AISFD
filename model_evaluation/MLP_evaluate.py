import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from input import data_type
import shap
from MLP_frame import Model_isp_all,Model_c_t_all,Model_cstar_all
import seaborn as sns
from model_evaluate import evaluation_MLP
from draw import plot
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
sns.set_theme(context='paper', style='whitegrid', palette='deep', 
              font='Arial', font_scale=1.8, color_codes=True, 
              rc={'lines.linewidth': 2, 'axes.grid': True,
                  'ytick.left': True, 'xtick.bottom': True, 
                  'font.weight': 'bold', 'axes.labelweight': 'bold'})
def load_data():
    df_train = pd.read_csv('../model_train/data_train.csv')
    df_test = pd.read_csv('../model_train/data_test.csv')
    
    X_train, Y_c_t_train, Y_isp_train, Y_cstar_train = data_type(df_train)
    X_test, Y_c_t_test, Y_isp_test, Y_cstar_test = data_type(df_test)
    feature_names = X_train.columns
  
    std = StandardScaler()
    X_train_s = std.fit_transform(X_train)
    X_test_s = std.transform(X_test)
    

    X_train_s_t = torch.Tensor(X_train_s.astype(np.float32))
    X_test_s_t = torch.Tensor(X_test_s.astype(np.float32))
    Y_c_t_train_t = torch.Tensor(Y_c_t_train.astype(np.float32))
    Y_c_t_test_t = torch.Tensor(Y_c_t_test.astype(np.float32))
    Y_isp_train_t = torch.Tensor(Y_isp_train.astype(np.float32))
    Y_isp_test_t = torch.Tensor(Y_isp_test.astype(np.float32))
    Y_cstar_train_t = torch.Tensor(Y_cstar_train.astype(np.float32))
    Y_cstar_test_t = torch.Tensor(Y_cstar_test.astype(np.float32))
    
    return X_train_s_t, X_test_s_t, Y_c_t_train_t, Y_c_t_test_t, Y_isp_train_t, Y_isp_test_t, Y_cstar_train_t, Y_cstar_test_t,feature_names


def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_isp = Model_isp_all([832,704,1024,288,64,704,384,1024])
    model_isp.load_state_dict(torch.load('../prediction/isp_best_network.pth'))
    
    model_c_t = Model_c_t_all([160,128,192,256,128,224,192,224,192,128,128,160,32,192])
    model_c_t.load_state_dict(torch.load('../prediction/c_t_best_network.pth'))
    
    model_cstar = Model_cstar_all([160,64,160,160,96,256,64,256,192,96,192,192,224])
    model_cstar.load_state_dict(torch.load('../prediction/cstar_best_network.pth'))

    return model_c_t, model_isp, model_cstar


def predict(models, X_train_s_t, X_test_s_t):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_c_t, model_isp, model_cstar = models
    model_c_t = model_c_t.to(device)
    c_t_train_p = model_c_t(X_train_s_t.to(device))
    c_t_test_p = model_c_t(X_test_s_t.to(device))
    print('--------------------------')
    
    model_isp = model_isp.to(device)
    isp_train_p = model_isp(X_train_s_t.to(device))
    isp_test_p = model_isp(X_test_s_t.to(device))
    print('--------------------------')
    
    model_cstar = model_cstar.to(device)
    cstar_train_p = model_cstar(X_train_s_t.to(device))
    cstar_test_p = model_cstar(X_test_s_t.to(device))
    print('--------------------------')
    return c_t_train_p, c_t_test_p, isp_train_p,isp_test_p,cstar_train_p, cstar_test_p 
def shap_explain(model_isp,X_train_s_t,X_test_s_t,feature_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_s_t = torch.Tensor(X_train_s_t).to(device)
    X_test_s_t = torch.Tensor(X_test_s_t).to(device)
    X_train_s_t = X_train_s_t # or X_train_s_t.reshape(X_train_s_t.size(0), 1)
    X_test_s_t = X_test_s_t 
    explainer = shap.GradientExplainer(model_isp, X_train_s_t[0:50000])
    shap_values = explainer.shap_values(X_test_s_t)
    shap.summary_plot(shap_values, X_test_s_t.cpu().numpy(), feature_names=feature_names)
    ax = plt.gca() 
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(2)
        
    plt.tight_layout()
    plt.savefig('shap_summary_plot_seaborn_style.tif', bbox_inches='tight', dpi=600)
    print('shap_summary_plot_seaborn_style.png')

    #plt.savefig('shap_summary_plot.png', bbox_inches='tight', dpi=300)
    #print('shap_summary_plot.png')


    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    

    sorted_idx = np.argsort(mean_shap_values)[::-1]
    

    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    sorted_shap_values = mean_shap_values[sorted_idx]
    

    with open('shap_feature_importance.txt', 'w') as f:
        for i in range(len(sorted_feature_names)):
            f.write(f"{sorted_feature_names[i]}: {sorted_shap_values[i]:.4f}\n")
    

    print('shap_feature_importance.txt')
    

    top_n = 20  
    top_feature_names = sorted_feature_names[:top_n]
    top_shap_values = sorted_shap_values[:top_n]    
    plt.figure(figsize=(10, 6))   
    plt.barh(top_feature_names, top_shap_values, color='skyblue')    
    plt.xlabel('Mean Absolute SHAP Value')   
    plt.title(f'Top {top_n} Feature Importance Based on SHAP Values')    
    plt.gca().invert_yaxis()    
    plt.tight_layout()
    plt.savefig('shap_feature_importance_top20.png', bbox_inches='tight', dpi=300)
 
    print('shap_feature_importance_top20.png')
if __name__ == '__main__':
    X_train_s_t, X_test_s_t, Y_c_t_train_t, Y_c_t_test_t, Y_isp_train_t, Y_isp_test_t, Y_cstar_train_t, Y_cstar_test_t,feature_names = load_data()
    models = load_models()   
    c_t_train_p, c_t_test_p, isp_train_p, isp_test_p,cstar_train_p, cstar_test_p = predict(models,X_train_s_t,X_test_s_t)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model_isp = Model_isp_all([832,704,1024,288,64,704,384,1024])
    # model_isp.load_state_dict(torch.load('/home/shuoxing/wrh/AI_formulation_design/model_evaluation/isp_best_network.pth'))
    
    # shap_explain(model_isp.to(device),X_train_s_t.to(device),X_test_s_t.to(device),feature_names)

    save_dir = '/home/shuoxing/wrh/AI_formulation_design/model_evaluation'
    evaluation_MLP('MLP', 'c_t', Y_c_t_train_t.to(device), c_t_train_p.to(device), Y_c_t_test_t.to(device), c_t_test_p.to(device),save_dir)
    evaluation_MLP('MLP', 'isp', Y_isp_train_t.to(device), isp_train_p.to(device), Y_isp_test_t.to(device), isp_test_p.to(device),save_dir)
    evaluation_MLP('MLP', 'cstar', Y_cstar_train_t.to(device), cstar_train_p.to(device), Y_cstar_test_t.to(device), cstar_test_p.to(device),save_dir)

    plot('MLP', 'c_t', Y_c_t_train_t, c_t_train_p, Y_c_t_test_t, c_t_test_p)
    plot('MLP', 'isp', Y_isp_train_t, isp_train_p, Y_isp_test_t, isp_test_p)
    plot('MLP', 'cstar', Y_cstar_train_t, cstar_train_p, Y_cstar_test_t, cstar_test_p)