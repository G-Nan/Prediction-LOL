from models import *

import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from torch.optim.adam import Adam

from torch import nn
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

def make_classifier(ml_set_parameter):

    classifiers = {}

    model_LR = LogisticRegression(max_iter = ml_set_parameter.LR.max_iter)
    model_SVC = SVC()
    model_kNN = KNeighborsClassifier()
    model_RF = RandomForestClassifier()
    model_XGB = xgb.XGBClassifier()

    param_LR = {'C' : ml_set_parameter.LR.C}
    param_SVC = {'C' : ml_set_parameter.SVC.C,
                 'gamma' : ml_set_parameter.SVC.gamma,
                 'degree' : ml_set_parameter.SVC.degree}
    param_kNN = {'n_neighbors' : ml_set_parameter.kNN.n_neighbors}
    param_RF = {'n_estimators' : ml_set_parameter.RF.n_estimators,
                'max_depth' : ml_set_parameter.RF.max_depth,
                'min_samples_split' : ml_set_parameter.RF.min_samples_split,
                'min_samples_leaf' : ml_set_parameter.RF.min_samples_leaf}
    param_XGB = {'n_estimators' : ml_set_parameter.XGB.n_estimators,
                 'learning_rate' : ml_set_parameter.XGB.learning_rate,
                 'max_depth' : ml_set_parameter.XGB.max_depth,
                 'subsample' : ml_set_parameter.XGB.subsample,
                 'colsample_bytree' : ml_set_parameter.XGB.colsample_bytree,
                 'reg_alpha' : ml_set_parameter.XGB.reg_alpha}

    SKFold = StratifiedKFold(n_splits = ml_set_parameter.SKFold.n_splits, shuffle = True, random_state = 42)

    gs_LR = GridSearchCV(estimator = model_LR, param_grid = param_LR, 
                         scoring = 'accuracy', cv = SKFold, n_jobs = -1, verbose = 0)
    gs_SVC = GridSearchCV(estimator = model_SVC, param_grid = param_SVC,
                          scoring = 'accuracy', cv = SKFold, n_jobs = -1, verbose = 0)
    gs_kNN = GridSearchCV(estimator = model_kNN, param_grid = param_kNN, 
                          scoring = 'accuracy', cv = SKFold, n_jobs = -1, verbose = 0)
    gs_RF = GridSearchCV(estimator = model_RF, param_grid = param_RF, 
                         scoring = 'accuracy', cv = SKFold, n_jobs = -1, verbose = 0)
    gs_XGB = GridSearchCV(estimator = model_XGB, param_grid = param_XGB, 
                          scoring = 'accuracy', cv = SKFold, n_jobs = -1, verbose = 0)

    classifiers['LR'] = gs_LR
    classifiers['SVC'] = gs_SVC
    classifiers['kNN'] = gs_kNN
    classifiers['RF'] = gs_RF
    classifiers['XGB'] = gs_XGB

    return classifiers

def make_nn_param(nn_set_parameter):

    batch_size_ = nn_set_parameter.batch_size
    patience_ = nn_set_parameter.patience
    num_epochs_ = nn_set_parameter.num_epochs

    param_lr = nn_set_parameter.lr
    param_dropout = nn_set_parameter.dropout
    param_hidden_size = nn_set_parameter.hidden_size
    param_num_layers = nn_set_parameter.num_layers

    list_params = []

    for lr_ in param_lr:
        for dropout_ in param_dropout:
            for hs_ in param_hidden_size:
                for nl_ in param_num_layers:
                    list_params.append({'batch_size' : batch_size_,
                                        'patience' : patience_,
                                        'num_epochs' : num_epochs_,
                                        'lr' : lr_,
                                        'dropout' : dropout_,
                                        'hidden_size' : hs_, 
                                        'num_layers' : nl_})
                                        
    return list_params

def dataset_ml(df, scaler_name):

    x = df.iloc[:, :-2]
    y = np.array(df['win'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    if scaler_name == 'SS':
        scaler = StandardScaler()
    if scaler_name == 'MM':
        scaler = MinMaxScaler()

    if scaler_name in ['SS', 'MM']:
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_scaled = np.array(x_train)
        x_test_scaled = np.array(x_test)
    
    return x_train_scaled, x_test_scaled, y_train, y_test
  
def dataset_nn(df, list_df, scaler_name):

    if scaler_name == 'SS':
        scaler = StandardScaler()
    if scaler_name == 'MM':
        scaler = MinMaxScaler()
        
    x = np.concatenate(list_df, axis=2)
    y = np.array(df['win'])
        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3128)
    
    x_train_scaled = np.zeros_like(x_train)
    x_test_scaled = np.zeros_like(x_test)
    
    if scaler_name in ['SS', 'MM']:
        scalers = [scaler for _ in range(x_train.shape[2])]
        
        for i in range(x_train.shape[2]):
            x_train_scaled[:, :, i] = scalers[i].fit_transform(x_train[:, :, i])
            x_test_scaled[:, :, i] = scalers[i].transform(x_test[:, :, i])
            
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test
    
    x_train_scaled, x_val_scaled, y_train, y_val = train_test_split(x_train_scaled, y_train, test_size = 0.25, random_state = 3128)
    
    x_train_scaled_tensor = torch.FloatTensor(x_train_scaled)
    x_test_scaled_tensor = torch.FloatTensor(x_test_scaled)
    x_val_scaled_tensor = torch.FloatTensor(x_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view([-1, 1])
    y_test_tensor = torch.FloatTensor(y_test).view([-1, 1])
    y_val_tensor = torch.FloatTensor(y_val).view([-1, 1])
    
    return x_train_scaled_tensor, x_test_scaled_tensor, x_val_scaled_tensor, y_train_tensor, y_test_tensor, y_val_tensor
 
def training_ml(x_train, x_test, y_train, y_test, classifiers):

    SKFold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    cv_results = []
    cv_models = []
    scores_accuracy = []
    scores_recall = []
    scores_precision = []
    scores_f1 = []
    model_names = []
    
    pbar_models = tqdm(classifiers.items(), position = 1)
    for model_name, classifier in pbar_models:
        pbar_models.set_description(model_name)
        
        classifier.fit(x_train, y_train)
        best_model = classifier.best_estimator_
        
        cv_results.append(cross_val_score(best_model, 
                                          x_train, 
                                          y = y_train, 
                                          scoring = 'accuracy', 
                                          cv = SKFold, n_jobs = -1, verbose = 0))
        
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        
        cm = confusion_matrix(y_test, y_pred)
        score_accuracy = accuracy_score(y_test, y_pred)
        score_recall = recall_score(y_test, y_pred)
        score_precision = precision_score(y_test, y_pred)
        score_f1 = f1_score(y_test, y_pred)
        
        cv_models.append(best_model)
        scores_accuracy.append(score_accuracy)
        scores_recall.append(score_recall)
        scores_precision.append(score_precision)
        scores_f1.append(score_f1)
        model_names.append(model_name)
        
    pbar_models.close()
    
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    
    cv_res = pd.DataFrame({'CrossValMeans':cv_means,
                           'CrossValerrors': cv_std,
                           'Accuracy' : scores_accuracy,
                           'Recall' : scores_recall,
                           'Precision' : scores_precision,
                           'F1 Score' : scores_f1,
                           'Algorithm': model_names})

    return cv_res, cv_models

def Training_nn(x_train, x_test, x_val, y_train, y_test, y_val, list_params, model_name, device):

    input_size = x_train.shape[2]
    sequence_length = x_train.shape[1]
    min_loss_gshp = np.inf
    
    pbar_param = tqdm(list_params, position = 1)
    for params in pbar_param:
        
        batch_size = params['batch_size']
        patience = params['patience']
        num_epochs = params['num_epochs']
        lr = params['lr']
        dropout = params['dropout']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        
        pbar_param.set_description("(lr=%.4f) (dropout=%.2f) (hidden_size=%.0f) (num_layers=%.0f)" % 
                                    (lr, dropout, hidden_size, num_layers))

        if model_name == 'RNN':
            model = RNN(input_size = input_size,
                        hidden_size = hidden_size,
                        sequence_length = sequence_length,
                        num_layers = num_layers,
                        dropout = dropout,
                        device = device).to(device)

        if model_name == 'LSTM':
            model = LSTM(input_size = input_size,
                         hidden_size = hidden_size,
                         sequence_length = sequence_length,
                         num_layers = num_layers,
                         dropout = dropout,
                         device = device).to(device) 
                 
        if model_name == 'CNN_LSTM':                
            kernel_size = 3
            model = CNN_LSTM(input_size = input_size,
                             hidden_size = hidden_size,
                             kernel_size = kernel_size,
                             num_layers = num_layers,
                             dropout = dropout,
                             device = device).to(device)
                     
        optimizer = Adam(model.parameters(), lr = lr)
        criterion = nn.BCELoss()
        
        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)  
        val = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = batch_size, shuffle = True)

        n = len(train_loader)
        loss_list = []
        min_val_loss = np.inf
        x_val_acc = x_val.to(device)

        #epoch_iterator = tqdm(range(num_epochs), position = 1)
        #for epoch in epoch_iterator:
        for epoch in range(num_epochs):
            running_loss = 0.0

            for seq, target in train_loader:

                seq = seq.to(device)
                out = model(seq)
                
                loss = criterion(out.cpu(), target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            loss_list.append(running_loss/n)

            out_val = model(x_val.to(device)) 
            val_loss = criterion(out_val, y_val.to(device)).item()

            if (val_loss) < min_val_loss:
                min_val_loss = (val_loss)
                k = 0
                min_loss_model = model
                
            else:
                k += 1

            #epoch_iterator.set_description("(loss=%2.4f) (val_loss=%2.4f) (lr=%.4f) (dropout=%.2f) (hidden_size=%.0f) (num_layers=%.0f)" % 
            #                               (running_loss/n, val_loss, lr, dropout, hidden_size, num_layers))
            if k > patience:
                #print('\n Early Stopping / epoch: %d loss: %.4f'%(epoch+1, running_loss/n))
                break

        #epoch_iterator.close()
        
        if min_val_loss < min_loss_gshp:
            min_loss_gshp = min_val_loss
            min_loss_model_gshp = model
            best_epoch = epoch
            best_lr = lr
            best_patience = patience
            best_num_layers = num_layers
            best_hidden_size = hidden_size
            best_dropout = dropout

    x_test_acc = x_test.to(device)
    y_test_acc = y_test.detach().numpy()
    y_pred = np.array([1 if i[0] > 0.5 else 0 for i in min_loss_model_gshp(x_test_acc)]).reshape(-1, 1)

    list_hp = [best_epoch, best_lr, best_patience, best_num_layers, best_hidden_size, best_dropout]

    accuracy = accuracy_score(y_test_acc, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    list_score = [accuracy, recall, precision, f1]
    
    return min_loss_model_gshp, list_hp, list_score
            
def Training_mean(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers):
    
    data_type = 'mean'
    
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        
        if not os.path.exists(f'scores/{data_type}/{tier}/'):
            os.makedirs(f'scores/{data_type}/{tier}/')
         
        if not os.path.exists(f'models/{data_type}/{tier}/'):
            os.makedirs(f'models/{data_type}/{tier}/')
            
        for time_len in list_time_len:
            pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len}')
            
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
                
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
            x_train, x_test, y_train, y_test = dataset_ml(df, scaler_name)
            
            cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

            for row in range(len(cv_res)):
                model_name = cv_res.iloc[row, 6]
                score = list(cv_res.iloc[row, 0:6])
                path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(score, open(path_score, 'wb'))
            
            for i, model in enumerate(cv_models):
                model_name = list(classifiers.keys())[i]
                path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(model, open(path_model, "wb"))

    pbar_list_tier.close()
    
def Training_weightedmean(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers):
    
    data_type = 'weightedmean'
    
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        
        if not os.path.exists(f'scores/{data_type}/{tier}/'):
            os.makedirs(f'scores/{data_type}/{tier}/')
         
        if not os.path.exists(f'models/{data_type}/{tier}/'):
            os.makedirs(f'models/{data_type}/{tier}/')
            
        for time_len in list_time_len:
            pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len}')
            
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
                
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
            x_train, x_test, y_train, y_test = dataset_ml(df, scaler_name)
            
            cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

            for row in range(len(cv_res)):
                model_name = cv_res.iloc[row, 6]
                score = list(cv_res.iloc[row, 0:6])
                path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(score, open(path_score, 'wb'))
            
            for i, model in enumerate(cv_models):
                model_name = list(classifiers.keys())[i]
                path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(model, open(path_model, "wb"))
    
    pbar_list_tier.close()
    
def Training_point(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers):
    
    data_type = 'point'
    
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        
        if not os.path.exists(f'scores/{data_type}/{tier}/'):
            os.makedirs(f'scores/{data_type}/{tier}/')
         
        if not os.path.exists(f'models/{data_type}/{tier}/'):
            os.makedirs(f'models/{data_type}/{tier}/')
            
        for time_len in list_time_len:
            pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len}')
            
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
                
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
            x_train, x_test, y_train, y_test = dataset_ml(df, scaler_name)
            
            cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

            for row in range(len(cv_res)):
                model_name = cv_res.iloc[row, 6]
                score = list(cv_res.iloc[row, 0:6])
                path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(score, open(path_score, 'wb'))
            
            for i, model in enumerate(cv_models):
                model_name = list(classifiers.keys())[i]
                path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                pickle.dump(model, open(path_model, "wb"))

    pbar_list_tier.close()
    
def Training_timeseries(list_tier, list_time_len, list_col, col_name, scaler_name, list_params, model_name, device):
    
    data_type = 'timeseries'
    
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        
        if not os.path.exists(f'scores/{data_type}/{tier}/'):
            os.makedirs(f'scores/{data_type}/{tier}/')
         
        if not os.path.exists(f'models/{data_type}/{tier}/'):
            os.makedirs(f'models/{data_type}/{tier}/')
        
        if not os.path.exists(f'hyperparameters/{data_type}/{tier}/'):
            os.makedirs(f'hyperparameters/{data_type}/{tier}/')
            
        for time_len in list_time_len:
            pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len}')
            
            list_df = []
            for col in list_col:
                if scaler_name in ['SS', 'MM']:
                    df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}_{col}.csv').head(8000)
                if scaler_name == 'RT':
                    df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}_{col}.csv').head(8000)
                list_df.append(df.iloc[:, :-2].values.reshape(-1, time_len + 1, 1))
                
            x_train, x_test, x_val, y_train, y_test, y_val = dataset_nn(df, list_df, scaler_name)
            min_loss_model_gshp, list_hp, list_score = Training_nn(x_train, x_test, x_val, y_train, y_test, y_val, list_params, model_name, device)

            path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(list_score, open(path_score, 'wb'))

            path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pth'
            torch.save(min_loss_model_gshp.state_dict(), path_model)
            
            path_hp = f'hyperparameters/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(list_hp, open(path_hp, 'wb')) 
    
    pbar_list_tier.close()
    
def Training_lanchester(list_tier, list_time_len, list_lan_type, list_col, col_name, scaler_name, classifiers):
    
    data_type = 'lanchester'
    
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        
        if not os.path.exists(f'scores/{data_type}/{tier}/'):
            os.makedirs(f'scores/{data_type}/{tier}/')
         
        if not os.path.exists(f'models/{data_type}/{tier}/'):
            os.makedirs(f'models/{data_type}/{tier}/')
            
        for time_len in list_time_len:
            
            if col_name == '1col':
                
                for i, lan_type in enumerate(list_lan_type):
                    
                    for col in list_col:
                        pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len} / {col} / {lan_type}')
                        
                        df = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
                        df = df.head(8000)
                        
                        x_train, x_test, y_train, y_test = dataset_ml(df, scaler_name)
                        cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)
                
                        for row in range(len(cv_res)):
                            model_name = cv_res.iloc[row, 6]
                            score = list(cv_res.iloc[row, 0:6])
                            path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col}_{lan_type}.pkl'
                            pickle.dump(score, open(path_score, 'wb'))
                    
                        for j, model in enumerate(cv_models):
                            model_name = list(classifiers.keys())[j]
                            path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col}_{lan_type}.pkl'
                            pickle.dump(model, open(path_model, "wb"))
                    
            if col_name == '4col':
            
                for i, lan_type in enumerate(list_lan_type):
                
                    col_name = '4col'
                    df = pd.DataFrame()
                    
                    for col in list_col:
                        pbar_list_tier.set_description(f'{data_type} / {tier} / {time_len} / {col_name} / {lan_type}')
                        
                        df_lanchester = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
                        df_lanchester = df_lanchester.head(8000)
                        
                        df = pd.concat([df, df_lanchester.iloc[:, :-2]], axis = 1)
                        
                    df['win'] = df_lanchester['win']
                    df['res'] = df_lanchester['Residuals']
                   
                    x_train, x_test, y_train, y_test = dataset_ml(df, scaler_name)
                    cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)
                    
                    for row in range(len(cv_res)):
                        model_name = cv_res.iloc[row, 6]
                        score = list(cv_res.iloc[row, 0:6])
                        path_score = f'scores/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{lan_type}.pkl'
                        pickle.dump(score, open(path_score, 'wb'))
                
                    for j, model in enumerate(cv_models):
                        model_name = list(classifiers.keys())[j]
                        path_model = f'models/{data_type}/{tier}/{model_name}_{time_len}_{col_name}_{lan_type}.pkl'
                        pickle.dump(model, open(path_model, "wb"))
                
    pbar_list_tier.close()
    
def Training_mean_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size):

    data_type = 'mean'
    
    pbar_list_timelen = tqdm(list_time_len, position = 0)
    for time_len in pbar_list_timelen:
        pbar_list_timelen.set_description(f'{data_type} / All / {time_len}')
        
        if col_name == '4col':
            x_train, x_test, y_train, y_test = np.empty([0, 4]), np.empty([0, 4]), np.empty(0), np.empty(0)
        else:
            x_train, x_test, y_train, y_test = np.empty([0, 27]), np.empty([0, 27]), np.empty(0), np.empty(0)
        
        for tier in list_tier:
            
            if not os.path.exists(f'scores/{data_type}/ALL/'):
                os.makedirs(f'scores/{data_type}/ALL/')
             
            if not os.path.exists(f'models/{data_type}/ALL/'):
                os.makedirs(f'models/{data_type}/ALL/')
            
            
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
            
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
                
            x_train_sub, x_test_sub, y_train_sub, y_test_sub = dataset_ml(df, scaler_name)
        
            rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.8))
            rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
            
            x_train_sub = x_train_sub[rand_idx_train]
            x_test_sub = x_test_sub[rand_idx_test]
            y_train_sub = y_train_sub[rand_idx_train]
            y_test_sub = y_test_sub[rand_idx_test]
            
            x_train = np.concatenate((x_train, x_train_sub), axis = 0)
            x_test = np.concatenate((x_test, x_test_sub), axis = 0)
            y_train = np.concatenate((y_train, y_train_sub), axis = 0)
            y_test = np.concatenate((y_test, y_test_sub), axis = 0)
            
        cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

        for row in range(len(cv_res)):
            model_name = cv_res.iloc[row, 6]
            score = list(cv_res.iloc[row, 0:6])
            path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(score, open(path_score, 'wb'))
        
        for i, model in enumerate(cv_models):
            model_name = list(classifiers.keys())[i]
            path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(model, open(path_model, "wb"))
    
    pbar_list_timelen.close()

def Training_weightedmean_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size):
     
    data_type = 'weightedmean'
        
    pbar_list_timelen = tqdm(list_time_len, position = 0)
    for time_len in pbar_list_timelen:
        pbar_list_timelen.set_description(f'{data_type} / All / {time_len}')
        
        if col_name == '4col':
            x_train, x_test, y_train, y_test = np.empty([0, 4]), np.empty([0, 4]), np.empty(0), np.empty(0)
        else:
            x_train, x_test, y_train, y_test = np.empty([0, 27]), np.empty([0, 27]), np.empty(0), np.empty(0)
        
        for tier in list_tier:
            
            if not os.path.exists(f'scores/{data_type}/ALL/'):
                os.makedirs(f'scores/{data_type}/ALL/')
             
            if not os.path.exists(f'models/{data_type}/ALL/'):
                os.makedirs(f'models/{data_type}/ALL/')
                
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
            
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
                
            x_train_sub, x_test_sub, y_train_sub, y_test_sub = dataset_ml(df, scaler_name)
        
            rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.8))
            rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
            
            x_train_sub = x_train_sub[rand_idx_train]
            x_test_sub = x_test_sub[rand_idx_test]
            y_train_sub = y_train_sub[rand_idx_train]
            y_test_sub = y_test_sub[rand_idx_test]
            
            x_train = np.concatenate((x_train, x_train_sub), axis = 0)
            x_test = np.concatenate((x_test, x_test_sub), axis = 0)
            y_train = np.concatenate((y_train, y_train_sub), axis = 0)
            y_test = np.concatenate((y_test, y_test_sub), axis = 0)
            
        cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

        for row in range(len(cv_res)):
            model_name = cv_res.iloc[row, 6]
            score = list(cv_res.iloc[row, 0:6])
            path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(score, open(path_score, 'wb'))
        
        for i, model in enumerate(cv_models):
            model_name = list(classifiers.keys())[i]
            path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(model, open(path_model, "wb"))
    
    pbar_list_timelen.close()
    
def Training_point_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size):
     
    data_type = 'point'
        
    pbar_list_timelen = tqdm(list_time_len, position = 0)
    for time_len in pbar_list_timelen:
        pbar_list_timelen.set_description(f'{data_type} / All / {time_len}')
        
        if col_name == '4col':
            x_train, x_test, y_train, y_test = np.empty([0, 4]), np.empty([0, 4]), np.empty(0), np.empty(0)
        else:
            x_train, x_test, y_train, y_test = np.empty([0, 27]), np.empty([0, 27]), np.empty(0), np.empty(0)
        
        for tier in list_tier:
            
            if not os.path.exists(f'scores/{data_type}/ALL/'):
                os.makedirs(f'scores/{data_type}/ALL/')
             
            if not os.path.exists(f'models/{data_type}/ALL/'):
                os.makedirs(f'models/{data_type}/ALL/')
                
            if scaler_name in ['SS', 'MM']:
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}.csv').head(8000)
            if scaler_name == 'RT':
                df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}.csv').head(8000)
            
            if col_name == '4col':
                df = df[list_col + ['win', 'match_id']]
                
            x_train_sub, x_test_sub, y_train_sub, y_test_sub = dataset_ml(df, scaler_name)
        
            rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.8))
            rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
            
            x_train_sub = x_train_sub[rand_idx_train]
            x_test_sub = x_test_sub[rand_idx_test]
            y_train_sub = y_train_sub[rand_idx_train]
            y_test_sub = y_test_sub[rand_idx_test]
            
            x_train = np.concatenate((x_train, x_train_sub), axis = 0)
            x_test = np.concatenate((x_test, x_test_sub), axis = 0)
            y_train = np.concatenate((y_train, y_train_sub), axis = 0)
            y_test = np.concatenate((y_test, y_test_sub), axis = 0)
            
        cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)

        for row in range(len(cv_res)):
            model_name = cv_res.iloc[row, 6]
            score = list(cv_res.iloc[row, 0:6])
            path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(score, open(path_score, 'wb'))
        
        for i, model in enumerate(cv_models):
            model_name = list(classifiers.keys())[i]
            path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
            pickle.dump(model, open(path_model, "wb"))
    
    pbar_list_timelen.close()
    
def Training_timeseries_All(list_tier, list_time_len, list_col, col_name, scaler_name, list_params, model_name, device, rand_size):
     
    data_type = 'timeseries'
        
    pbar_list_timelen = tqdm(list_time_len, position = 0)
    for time_len in pbar_list_timelen:
        pbar_list_timelen.set_description(f'{data_type} / All / {time_len}')
        
        x_train, x_test, x_val = np.empty([0, time_len + 1, 4]), np.empty([0, time_len + 1, 4]), np.empty([0, time_len + 1, 4])
        y_train, y_test, y_val = np.empty([0, 1]), np.empty([0, 1]), np.empty([0, 1])

        for tier in list_tier:
            
            if not os.path.exists(f'scores/{data_type}/ALL/'):
                os.makedirs(f'scores/{data_type}/ALL/')
             
            if not os.path.exists(f'models/{data_type}/ALL/'):
                os.makedirs(f'models/{data_type}/ALL/')
                
            if not os.path.exists(f'hyperparameters/{data_type}/ALL/'):
                os.makedirs(f'hyperparameters/{data_type}/ALL/')
                
            list_df = []
            for col in list_col:
                if scaler_name in ['SS', 'MM']:
                    df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_{time_len}_{col}.csv').head(8000)
                if scaler_name == 'RT':
                    df = pd.read_csv(f'Data/{data_type}/{tier}/df_{data_type}_ratio_{time_len}_{col}.csv').head(8000)
                    
                list_df.append(df.iloc[:, :-2].values.reshape(-1, time_len + 1, 1))
                
            x_train_sub, x_test_sub, x_val_sub, y_train_sub, y_test_sub, y_val_sub = dataset_nn(df, list_df, scaler_name)
        
            rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.6))
            rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
            rand_idx_val = np.random.choice(len(x_val_sub), size = int(rand_size * 0.2))
        
            x_train_sub = x_train_sub[rand_idx_train]
            x_test_sub = x_test_sub[rand_idx_test]
            x_val_sub = x_val_sub[rand_idx_val]
            y_train_sub = y_train_sub[rand_idx_train]
            y_test_sub = y_test_sub[rand_idx_test]
            y_val_sub = y_val_sub[rand_idx_val]
            
            x_train = np.concatenate((x_train, x_train_sub), axis = 0)
            x_test = np.concatenate((x_test, x_test_sub), axis = 0)
            x_val = np.concatenate((x_val, x_val_sub), axis = 0)
            y_train = np.concatenate((y_train, y_train_sub), axis = 0)
            y_test = np.concatenate((y_test, y_test_sub), axis = 0)
            y_val = np.concatenate((y_val, y_val_sub), axis = 0)
            
        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        x_val = torch.FloatTensor(x_val)
        y_train = torch.FloatTensor(y_train).view([-1, 1])
        y_test = torch.FloatTensor(y_test).view([-1, 1])
        y_val = torch.FloatTensor(y_val)
    
        min_loss_model_gshp, list_hp, list_score = Training_nn(x_train, x_test, x_val, y_train, y_test, y_val, list_params, model_name, device)

        path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
        pickle.dump(list_score, open(path_score, 'wb'))

        path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pth'
        torch.save(min_loss_model_gshp.state_dict(), path_model)

        path_hp = f'hyperparameters/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
        pickle.dump(list_hp, open(path_hp, 'wb')) 
    
    pbar_list_timelen.close()
    
def Training_lanchester_All(list_tier, list_time_len, list_lan_type, list_col, col_name, scaler_name, classifiers, rand_size):
          
    data_type = 'lanchester'
    
    pbar_list_timelen = tqdm(list_time_len, position = 0)
    for time_len in pbar_list_timelen:
        
        if col_name == '1col':
                
            for i, lan_type in enumerate(list_lan_type):
                for col in list_col:
                    pbar_list_timelen.set_description(f'{data_type} / All / {time_len} / {col} / {lan_type}')
                    
                    if i <= 1:
                        x_train, x_test, y_train, y_test = np.empty([0, 3]), np.empty([0, 3]), np.empty(0), np.empty(0)
                    if i == 2:
                        x_train, x_test, y_train, y_test = np.empty([0, 4]), np.empty([0, 4]), np.empty(0), np.empty(0)
                        
                    for tier in list_tier:
                        
                        if not os.path.exists(f'scores/{data_type}/ALL/'):
                            os.makedirs(f'scores/{data_type}/ALL/')
                         
                        if not os.path.exists(f'models/{data_type}/ALL/'):
                            os.makedirs(f'models/{data_type}/ALL/')
                
                        df = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
                        df = df.head(8000)
                        
                        x_train_sub, x_test_sub, y_train_sub, y_test_sub = dataset_ml(df, scaler_name)
                        
                        rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.8))
                        rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
                        
                        x_train_sub = x_train_sub[rand_idx_train]
                        x_test_sub = x_test_sub[rand_idx_test]
                        y_train_sub = y_train_sub[rand_idx_train]
                        y_test_sub = y_test_sub[rand_idx_test]
                        
                        x_train = np.concatenate((x_train, x_train_sub), axis = 0)
                        x_test = np.concatenate((x_test, x_test_sub), axis = 0)
                        y_train = np.concatenate((y_train, y_train_sub), axis = 0)
                        y_test = np.concatenate((y_test, y_test_sub), axis = 0)
                        
                    cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)
                
                    for row in range(len(cv_res)):
                        model_name = cv_res.iloc[row, 6]
                        score = list(cv_res.iloc[row, 0:6])
                        path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col}_{lan_type}.pkl'
                        pickle.dump(score, open(path_score, 'wb'))
                
                    for j, model in enumerate(cv_models):
                        model_name = list(classifiers.keys())[j]
                        path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col}_{lan_type}.pkl'
                        pickle.dump(model, open(path_model, "wb"))
                    
        if col_name == '4col':
            
            for i, lan_type in enumerate(list_lan_type):
                pbar_list_timelen.set_description(f'{data_type} / All / {time_len} / {col_name} / {lan_type}')
            
                if i <= 1:
                    x_train, x_test, y_train, y_test = np.empty([0, 12]), np.empty([0, 12]), np.empty(0), np.empty(0)
                if i == 2:
                    x_train, x_test, y_train, y_test = np.empty([0, 16]), np.empty([0, 16]), np.empty(0), np.empty(0)
                    
                for tier in list_tier:
                    
                    if not os.path.exists(f'scores/{data_type}/ALL/'):
                        os.makedirs(f'scores/{data_type}/ALL/')
                     
                    if not os.path.exists(f'models/{data_type}/ALL/'):
                        os.makedirs(f'models/{data_type}/ALL/')
                
                    df = pd.DataFrame()
                    for col in list_col:
                        df_lanchester = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
                        df_lanchester = df_lanchester.head(8000)
                        
                        df = pd.concat([df, df_lanchester.iloc[:, :-2]], axis = 1)
                        
                    df['win'] = df_lanchester['win']
                    df['res'] = df_lanchester['Residuals']

                    x_train_sub, x_test_sub, y_train_sub, y_test_sub = dataset_ml(df, scaler_name)
                     
                    rand_idx_train = np.random.choice(len(x_train_sub), size = int(rand_size * 0.8))
                    rand_idx_test = np.random.choice(len(x_test_sub), size = int(rand_size * 0.2))
                    
                    x_train_sub = x_train_sub[rand_idx_train]
                    x_test_sub = x_test_sub[rand_idx_test]
                    y_train_sub = y_train_sub[rand_idx_train]
                    y_test_sub = y_test_sub[rand_idx_test]
                    
                    x_train = np.concatenate((x_train, x_train_sub), axis = 0)
                    x_test = np.concatenate((x_test, x_test_sub), axis = 0)
                    y_train = np.concatenate((y_train, y_train_sub), axis = 0)
                    y_test = np.concatenate((y_test, y_test_sub), axis = 0)
                        
                cv_res, cv_models = training_ml(x_train, x_test, y_train, y_test, classifiers)
                
                for row in range(len(cv_res)):
                    model_name = cv_res.iloc[row, 6]
                    score = list(cv_res.iloc[row, 0:6])
                    path_score = f'scores/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{lan_type}.pkl'
                    pickle.dump(score, open(path_score, 'wb'))
            
                for j, model in enumerate(cv_models):
                    model_name = list(classifiers.keys())[j]
                    path_model = f'models/{data_type}/ALL/{model_name}_{time_len}_{col_name}_{lan_type}.pkl'
                    pickle.dump(model, open(path_model, "wb"))
                    
    pbar_list_timelen.close()