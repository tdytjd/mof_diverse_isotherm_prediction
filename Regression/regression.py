import numpy as np
import pandas as pd
import os
import sys

import warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
def transfer_predicted_y(y_predict):
    y_predict_actual=[]
    for i in range(len(y_predict)):
        if y_predict[i]>=0:
            y_predict_actual.append(y_predict[i])
        else:
            y_predict_actual.append(10**y_predict[i])
    return y_predict_actual

def transfer_original_y(y_original):
    y_to_predict=[]
    for i in range(len(y_original)):
        if y_original[i]>0.1:
            y_to_predict.append(y_original[i])
        elif y_original[i]==0:
            y_to_predict.append(-10)
        else:
            y_to_predict.append(np.log10(y_original[i]))
    return y_to_predict


# define functions
def train_model(X, y, test_fold):
    model = XGBRegressor()

    param_grid = {'n_estimators': [200,300,500],
    'max_depth': [4,5,6],
    'learning_rate':[0.1,0.15,0.01,0.2]}

    ps = PredefinedSplit(test_fold)
    #scorer = make_scorer(mean_squared_error, greater_is_better = False)
    rgs = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, refit = True)#, scoring = scorer)#
    rgs.fit(X,y)

    best_model = rgs.best_estimator_

    print('Optimal hyperparameters: {}\n'.format(rgs.best_params_))

    return best_model

# make direcotries
data = 'data'
if not os.path.isdir(data):
    os.makedirs(data)

#read data
set_1=pd.read_csv('Set1_with_predictions.csv')
set_2=pd.read_csv('Set2_with_predictions.csv')
#read 6mof_12mol
set_3=pd.read_csv('Set3_with_predictions.csv')


set_1['transfered_loading']=transfer_original_y(list(set_1['loading']))
set_2['transfered_loading']=transfer_original_y(list(set_2['loading']))
set_3['transfered_loading']=transfer_original_y(list(set_3['loading']))


# descriptors
pore = list(set_1.columns[2:5])
pes = list(set_1.columns[5:33])
critical_mol = list(set_1.columns[33:36])
logK = [set_1.columns[37]]
sphere = list(set_1.columns[38:41])
pressure_des = [set_1.columns[44]]
langmuir_des=[set_1.columns[47]]
xlogp_des = [set_1.columns[48]]
descriptor_list=pore+critical_mol+logK+pressure_des+xlogp_des+sphere+langmuir_des+pes

print(descriptor_list)


# ensemble modeling on set 1
seed = 10
random_seed = np.arange(seed)

set_2_with_predictions = set_2.copy()
set_3_with_predictions = set_3.copy()

column_name_list=[]

for seed in random_seed:
    print('split #{} starts'.format(seed))

    # make subdirecotries
    data_split = '{}/{}'.format(data, seed)

    if not os.path.isdir(data_split):
        os.makedirs(data_split)

    # split dataset
    train_valid, test = train_test_split(set_1, test_size = .2, stratify = set_1.molecule, random_state = seed)
    X_train_valid = train_valid[descriptor_list]
    X_test = test[descriptor_list]
    y_train_valid = train_valid['transfered_loading']
    y_test = test['transfered_loading']

    train, valid = train_test_split(train_valid, test_size = .2 , stratify = train_valid.molecule, random_state = seed)

    test_fold = np.zeros(train_valid.shape[0])
    for j in train.index:
        a = train_valid.index.get_loc(j)
        test_fold[a] = -1

    # predict loading
    best_model_L = train_model(X_train_valid.values, y_train_valid, test_fold)

    L_train_valid_predict = best_model_L.predict(X_train_valid.values)
    L_test_predict = best_model_L.predict(X_test)

    L_train_valid_csv = train_valid.copy()
    L_train_valid_csv['predicted_L'] = L_train_valid_predict
    L_train_valid_csv['transferred_predicted_L'] = transfer_predicted_y(L_train_valid_predict)
    L_train_valid_csv[['MOF', 'molecule', 'pressure', 'loading', 'transferred_predicted_L']].to_csv('{}/{}.tsv'.format(data_split, 'train_valid_L'), sep = '\t', index = False)

    L_test_csv = test.copy()
    L_test_csv['predicted_L'] = L_test_predict
    L_test_csv['transferred_predicted_L'] = transfer_predicted_y(L_test_predict)
    L_test_csv[['MOF', 'molecule', 'pressure', 'loading', 'transferred_predicted_L']].to_csv('{}/{}.tsv'.format(data_split, 'test_L'), sep = '\t', index = False)

    # gather predicted values on the set 2
    L_predict = best_model_L.predict(set_2[descriptor_list].values)
    new_column_name='transferred_predicted_L_'+str(seed)
    set_2_with_predictions[new_column_name]=transfer_predicted_y(L_predict)
    column_name_list.append(new_column_name)
    
    # gather predicted values on the set 3
    L_predict_3 = best_model_L.predict(set_3[descriptor_list].values)
    new_column_name='transferred_predicted_L_'+str(seed)
    set_3_with_predictions[new_column_name]=transfer_predicted_y(L_predict_3)
    column_name_list.append(new_column_name)
    print('r2 training_val',r2_score(L_train_valid_csv['loading'], L_train_valid_csv['transferred_predicted_L']))
    print('r2 test',r2_score(L_test_csv['loading'], L_test_csv['transferred_predicted_L']))
    print('r2 set2',r2_score(set_2['loading'], transfer_predicted_y(L_predict)))
    print('r2 set3',r2_score(set_3['loading'], transfer_predicted_y(L_predict_3)))

    sorted_idx = np.argsort(best_model_L.feature_importances_)
    print('feature importance:\n',np.array(descriptor_list)[sorted_idx][::-1])
    print('feature importance values:\n',np.array(best_model_L.feature_importances_)[sorted_idx][::-1])
    print('split #{} ended\n'.format(seed))

set_2_with_predictions[['MOF', 'molecule', 'pressure', 'loading']+column_name_list].to_csv('{}/{}.tsv'.format(data, 'set_2_predictions'), sep = '\t', index = False)

set_3_with_predictions[['MOF', 'molecule', 'pressure', 'loading']+column_name_list].to_csv('{}/{}.tsv'.format(data, 'set_3_predictions'), sep = '\t', index = False)

