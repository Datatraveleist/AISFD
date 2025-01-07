from sklearn import linear_model
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error#, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle

def estimator(learner_name,label_name,descriptor_train_list,target_train_list,descriptor_test_list,target_test_list):
    krr = KernelRidge()
    rr = linear_model.Ridge()
    nn = MLPRegressor()
    svr = SVR()
    dectree = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    adaboost = AdaBoostRegressor()
    gbr = GradientBoostingRegressor()
    knn = KNeighborsRegressor()

   # dectree = 
    # path = 'hyperparameter_searching/'+label_name+'_'+learner_name+'_best.pkl'
    # with open(path,'rb') as tf:
    #     learner_parameters = pickle.load(tf)
    # print(learner_parameters)
    parameters_krr = {'alpha':[1,1e-1,1e-2,1e-3], 
                      'gamma':[1,1e-1,1e-2,1e-3],
                      'kernel':['rbf']}#,'gamma':[1,1e-1,1e-2,1e-3],
    #print(parameters_krr)
    parameters_rr = {'alpha':[10,1,0.1,0.01,0.001]}
    parameters_nn = {'hidden_layer_sizes':([300,100,50],[512,64,64,64,12],[1200,64,32,16,8])
                     ,'alpha' : (0.0001,), 
                     'activation' : ('relu',), 
                     'solver': (['adam','sgd','lbfgs']),
                     'max_iter':[1000],
                     'learning_rate_init':([0.001,0.0001,0.00001])}
    
    parameters_svr = {'kernel':('rbf','poly'), 'gamma':['scale'],'degree':[10],'epsilon':[0.1,0.01],'C':[100]}
    parameters_dectree = {'criterion':['mae'], 'max_depth' :[2,3,4,5] }
    parameters_rf = {'criterion':['mae'], 'n_estimators':[200], 'max_features':['auto'],'n_jobs':[-1]}
    parameters_adaboost = {'n_estimators':[100,50],'loss':['exponential']}
    parameters_gbr = {'loss':['lad','squared_error'],'learning_rate':[0.01],'min_samples_leaf':[1], 'max_depth':[12],'min_samples_split' :[2], 'max_features':['sqrt'], 'n_estimators':[1000],'criterion':['friedman_mse']}
    parameters_knn = {'n_neighbors':[2,3,5,8,10], 'weights':['uniform']}
    
    # print(learner_parameters)

    learner = vars()[learner_name]
    learner_parameters = vars()['parameters_'+learner_name]
    # print(learner_parameters)
    #rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    scoring = {'r2':make_scorer(r2_score),'variance': make_scorer(explained_variance_score),
               'mae':make_scorer(mean_absolute_error), 'mse':make_scorer(mean_squared_error),
               'mape':make_scorer(mean_absolute_percentage_error)}

    grid = GridSearchCV(learner, learner_parameters, cv = 5, scoring = scoring, refit = 'r2', return_train_score=True)
    # print(clf.fit(descriptor_train_list, target_train_list).best_score_)
   #     predict_list = []
    #MAE_TEST_LIST = []
    
    grid.fit(descriptor_train_list, target_train_list)
    print(grid.fit(descriptor_train_list, target_train_list).best_score_)
    #print(clf.scorer_)  
    # grid.best_score_   
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    test_predict_value_list = grid.predict(descriptor_test_list)
    test_r2_value = r2_score(target_test_list, test_predict_value_list)
    test_rmse_value = np.sqrt(mean_squared_error(target_test_list, test_predict_value_list))
    test_mae_value = mean_absolute_error(target_test_list, test_predict_value_list)
    test_variance = explained_variance_score(target_test_list, test_predict_value_list)
    test_mape = mean_absolute_percentage_error(target_test_list, test_predict_value_list)
    test_pe = test_mae_value/(np.abs(target_test_list).sum()/len(target_test_list))
    print(grid.cv_results_)
    
    train_mae  = grid.cv_results_['mean_train_mae'][grid.best_index_]
    train_r2  = grid.cv_results_['mean_train_r2'][grid.best_index_]
    train_rmse  = np.sqrt(grid.cv_results_['mean_train_mse'][grid.best_index_])
    train_mape  = grid.cv_results_['mean_train_mape'][grid.best_index_]
    
    validation_mae  = grid.cv_results_['mean_test_mae'][grid.best_index_]
    validation_r2  = grid.cv_results_['mean_test_r2'][grid.best_index_]
    validation_rmse  = np.sqrt(grid.cv_results_['mean_test_mse'][grid.best_index_])
    validation_mape  = grid.cv_results_['mean_test_mape'][grid.best_index_]

    with open(str(learner_name)+str(label_name)+'.pkl', 'wb') as fm:
      	pickle.dump(best_model,fm)  

    print('###########'+str(learner_name)+str(label_name)+'#########')
    # E:\python\wrh1.0\results\model_save\compare
    
    print('######################')
    # f = open('E:/python/Optimization and Design/model_train/model_evaluation/'+str(learner_name)+str(label_name)+'.txt', "w")
    # print(str(best_model)+'best_params is {}'.format(best_params),file=f)
    # # print('r2 of train set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'
    # #       .format(train_r2,  train_rmse, train_mae, train_mape))
    # # print(24*'-')
    # # print('r2 of validation set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'
    # #       .format(validation_r2,  validation_rmse, validation_mae, validation_mape),file=f)
    # # print('r2 of validation set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'
    # #       .format(validation_r2,  validation_rmse, validation_mae, validation_mape))
    # # #print()
    # # print(24*'-')
    # # print('r2 of test set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}, pe is {:.5f}'
    # #       .format(test_r2_value,test_rmse_value,test_mae_value,test_mape, test_pe),file=f)
    # # print('r2 of test set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}, pe is {:.5f}'
    # #       .format(test_r2_value,test_rmse_value,test_mae_value,test_mape, test_pe))
    # # print(24*'-')
    # # print('best_params_{:.3f}'.format(clf.best_params_),file=f)
    # predict_test_results = best_model.predict(descriptor_test_list)
    # predict_train_results = best_model.predict(descriptor_train_list)
    # predict_test_results = np.array([predict_test_results]).reshape(-1,1)    
    # predict_train_results = np.array([predict_train_results]).reshape(-1,1) 

    # model_score_train=best_model.score(descriptor_train_list,target_train_list)
    # model_score_test=best_model.score(descriptor_test_list,target_test_list)
     
    with open('/home/shuoxing/wrh/AI_formulation_design/model_evaluation/'+str(learner_name)+str(label_name)+'.txt', "w", encoding='utf-8') as f:  # 使用 'with' 语句确保文件正确关闭
        print("best_params is ",best_params , file=f)
        print('r2 of train set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'.format(train_r2,  train_rmse, train_mae, train_mape),file=f)
        print('r2 of train set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'.format(train_r2,  train_rmse, train_mae, train_mape))
        print(24*'-')
        print('r2 of validation set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'.format(validation_r2,  validation_rmse, validation_mae, validation_mape),file=f)
        print('r2 of validation set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}'.format(validation_r2,  validation_rmse, validation_mae, validation_mape))
        print(24*'-')
        print('r2 of test set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}, pe is {:.5f}'.format(test_r2_value,test_rmse_value,test_mae_value,test_mape, test_pe),file=f)
        print('r2 of test set is {:.3f}, rmse is {:.3f}, mae is {:.3f}, mape is {:.3f}, pe is {:.5f}'.format(test_r2_value,test_rmse_value,test_mae_value,test_mape, test_pe))
    f.close()



'''    
    try:
        rfecv = RFECV(estimator=clf.best_estimator_, step=1, cv=5,
                  scoring='r2')
        rfecv.fit(descriptor_train_list, target_train_list)
       
        print("Optimal number of features : %d" % rfecv.n_features_)
        
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (r2)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.savefig('feature_selection/' + label_name + '_' + learner_name + '.jpg',dpi = 300)
        plt.show()
    except:
        print('The estimator has no feature importance attributes!')
'''
