import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from utils import *

sns.set_style('whitegrid')
colors = sns.color_palette("pastel") 
plt.rcParams['axes.unicode_minus'] = False

def result_score(list_dt, list_tier, list_time_len, list_col):
    i = 0
    list_score = []
    for dt in list_dt:
        for tier in list_tier + ['ALL']:
            for time_len in list_time_len:
                if dt in ['mean', 'weightedmean', 'point']:
                    for model_name in ['knn', 'LR', 'RF', 'SVC', 'XGB']:
                        for col_name in ['4col', 'ALL']:
                            for scaler_name in ['SS', 'RT']:
                                i += 1
                                file_path = f'scores/{dt}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                                score = pickle.load(open(file_path, 'rb')) 
                                list_score.append(score + [model_name, tier, dt, time_len, col_name, scaler_name])
                if dt == 'timeseries':
                    for model_name in ['RNN', 'LSTM', 'CNN_LSTM']:
                        for col_name in ['4col']:
                            for scaler_name in ['SS', 'RT']:
                                i += 1
                                file_path = f'scores/{dt}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                                score = pickle.load(open(file_path, 'rb')) 
                                list_score.append([np.nan] * 2 + score + [model_name, tier, dt, time_len, col_name, scaler_name])
                if dt == 'lanchester':
                    for model_name in ['knn', 'LR', 'RF', 'SVC', 'XGB']:
                        for col_name in list_col + ['4col']:
                            for scaler_name in ['Linear', 'Exponential', 'Mixed']:
                                i += 1
                                file_path = f'scores/{dt}/{tier}/{model_name}_{time_len}_{col_name}_{scaler_name}.pkl'
                                score = pickle.load(open(file_path, 'rb')) 
                                list_score.append(score + [model_name, tier, dt, time_len, col_name, scaler_name])
    
    dfs = pd.DataFrame(list_score, columns = ['CrossValMeans', 'CrossValerrors', 'Accuracy', 'Recall', 'Precision', 'F1 Score', 'Algorithm', 'tier', 'data_type', 'time_len', 'col', 'scaler'])
    dfs.to_csv('results/score.csv', index = False)
             
def result_table2(dfs, tier):

    df = dfs[dfs['tier'] == tier]
    df = df.rename(columns = {'data_type' : 'feature_type'})

    df_acc_mean = pd.DataFrame(df.groupby(['feature_type', 'time_len'])['Accuracy'].mean())
    piv_acc_mean = pd.pivot_table(df_acc_mean, index = 'time_len', columns = 'feature_type')
    piv_acc_mean = piv_acc_mean.rename(columns = {'Accuracy':'Average Accuracy'}, level = 0)

    df_acc_max = pd.DataFrame(df.groupby(['feature_type', 'time_len'])['Accuracy'].max())
    piv_acc_max = pd.pivot_table(df_acc_max, index = 'time_len', columns = 'feature_type')
    piv_acc_max = piv_acc_max.rename(columns = {'Accuracy':'Best Accuracy'}, level = 0)

    df_alg_max = df.loc[df.groupby(['feature_type', 'time_len'])['Accuracy'].idxmax()][['feature_type', 'Algorithm', 'time_len']]
    piv_alg_max = pd.pivot(df_alg_max, index = 'time_len', columns = 'feature_type')
    piv_alg_max = piv_alg_max.rename(columns = {'Algorithm' : 'Best Model'}, level = 0)

    df_concat = pd.concat([piv_acc_mean, piv_acc_max, piv_alg_max], axis = 1)
    df_result = df_concat.stack(0)
    df_result = df_result[['mean', 'weightedmean', 'point', 'timeseries', 'lanchester']]
    
    df_result.to_csv('results/table2.csv')
    
def result_table3(dfs, new_names, new_order):
    dfs_table3 = dfs[(dfs["data_type"] == "lanchester") & (dfs["time_len"].isin([5, 10]))].pivot_table(index = 'tier', 
                                                                                                       columns = 'time_len', 
                                                                                                       values = 'Accuracy', 
                                                                                                       aggfunc = 'mean')
    dfs_table3.columns = ['TA(%) 5mins', 'TA(%) 10mins']
    
    dfs_table3 = dfs_table3.rename(index=new_names)
    dfs_table3 = dfs_table3.loc[new_order]
    
    dfs_table3.to_csv("results/table3.csv")
    
def result_figure123(tier, list_coef, col, list_time_len, list_lan_type, list_fig_label):
   
    val = {}
    for i in range(3):
        val[i+1] = []
        for time_len in list_time_len:
            df = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
            val[i+1].append(df)

    for i in range(3):

        for j in range(4):

            if (i != 2) and j == 3:
                continue
                
            for k in range(2):

                df = val[i+1][k]
                
                model_type = list_lan_type[i]
                coef = list_coef[j]
                time_len = list_time_len[k]
                
                fig = plt.figure(figsize = (8, 4))
                sns.histplot(x = coef, data = df[df['win'] == 0], 
                             color = colors[3], label = f'Win Team')
                sns.histplot(x = coef, data = df[df['win'] == 1], 
                             color = colors[0], label = f'Lose Team')
                plt.legend(fontsize = 17)
                
                plt.xlabel(list_fig_label[3*i + j], fontsize = 20)
                plt.ylabel('Count', fontsize = 20)
                plt.xticks(fontsize = 17)
                plt.yticks(fontsize = 17)
                
                name_image = f'{coef}_{model_type}_{tier}_{time_len}'
                fig.figure.savefig(f'results/fig{i+1}/{name_image}.png', dpi = 200)
           
def result_fig4(tier, col, list_time_len, list_scaling):
    
    for time_len in list_time_len:
        for scaling in list_scaling:
            if scaling == 'Standard':
                df = pd.read_csv(f'Data/Mean/{tier}/df_mean_{time_len}.csv')
            if scaling == 'Ratio':
                df = pd.read_csv(f'Data/Mean/{tier}/df_mean_ratio_{time_len}.csv')
        
            fig = plt.figure(figsize = (8, 4))
            sns.histplot(x = col, data = df[df['win'] == 0], 
                         color = colors[3], label = f'Win Team')
            sns.histplot(x = col, data = df[df['win'] == 1], 
                         color = colors[0], label = f'Lose Team')
            plt.legend(fontsize = 15)
            plt.xlabel('totalDamageDone', fontsize = 18)
            plt.ylabel('Count', fontsize = 18)
            plt.xticks(fontsize = 16)
            plt.yticks(fontsize = 16)
            
            name_image = f'{col}_{scaling}_{tier}_{time_len}'
            fig.figure.savefig(f'results/fig4/{name_image}.png', dpi = 200, bbox_inches='tight')
    
def result_fig5(dfs, colors, new_order):
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 15))
    for i in range(6):
        
        sns.lineplot(x='tier', y='CrossValMeans', data=dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'mean')],
                     marker='o', linestyle='-', color=colors[0], ax=axes[i], label='mean')
        
        sns.lineplot(x='tier', y='Accuracy', data=dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'timeseries')],
                     marker='o', linestyle='-', color=colors[6], ax=axes[i], label='time series')
        
        sns.lineplot(x='tier', y='CrossValMeans', data=dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'weightedmean')],
                     marker='o', linestyle='-', color=colors[1], ax=axes[i], label='weighted mean')
        
        sns.lineplot(x='tier', y='CrossValMeans', data=dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester')],
                     marker='o', linestyle='-', color=colors[4], ax=axes[i], label='Lanchester')

        sns.lineplot(x='tier', y='CrossValMeans', data=dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'point')],
                     marker='o', linestyle='-', color=colors[2], ax=axes[i], label='point')

        axes[i].set_title(f'{i*5 + 5} minutes', fontsize = 17)
        axes[i].set_xticklabels(new_order)
        axes[i].set_ylim(0.55, 1.0)
        axes[i].set_yticks([0.6, 0.8, 1.0])

        ax = axes.flat[i]
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.3), ncol = 3, fontsize = 17)
        if i != 5:
            ax.legend([], [], frameon=False)

        ax.set_xlabel('')
        ax.set_ylabel('')
        axes[i].xaxis.set_tick_params(labelsize=17)
        axes[i].yaxis.set_tick_params(labelsize=17)

    plt.subplots_adjust(hspace=0.7)
    fig.figure.savefig('results/fig5/Score_tier_feature_time.png', dpi = 200)
    
def result_fig6(dfs, colors, new_order):
    
    fig, axes = plt.subplots(6, 1, figsize = (12, 15))
    for i in range(6):
        
        sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['col'] == 'totalGold')],
                     marker = 'o', linestyle = '-', color = colors[0], ax = axes[i], label = 'totalGold')
        sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['col'] == 'xp')],
                     marker = 'o', linestyle = '-', color = colors[1], ax = axes[i], label = 'xp')
        sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['col'] == 'totalDamageDone')],
                     marker = 'o', linestyle = '-', color = colors[2], ax = axes[i], label = 'totalDamageDone')
        sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['col'] == 'totalDamageTaken')],
                     marker = 'o', linestyle = '-', color = colors[3], ax = axes[i], label = 'totalDamageTaken')
        sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == i*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['col'] == '4col')],
                     marker = 'o', linestyle = '-', color = colors[4], ax = axes[i], label = 'allFour')

        axes[i].set_title(f'{i*5 + 5} minutes', fontsize = 17)
        axes[i].set_xticklabels(new_order)
        axes[i].set_ylim(0.6, 1.0)
        axes[i].set_yticks([0.6, 0.8, 1.0])
        
        ax = axes.flat[i]
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.3), ncol = 3, fontsize = 17)
        if i != 5:
            ax.legend([], [], frameon=False)
            
        ax.set_xlabel('')
        ax.set_ylabel('')
        axes[i].xaxis.set_tick_params(labelsize=17)
        axes[i].yaxis.set_tick_params(labelsize=17)

    plt.subplots_adjust(hspace = 0.7)
    fig.figure.savefig('results/fig6/Score_tier_variable_time.png', dpi = 200)
    
def result_fig7(dfs, list_col, colors, new_order):
    
    list_col_sub = list_col[:2]

    fig, axes = plt.subplots(6, 2, figsize = (12, 15))

    for i in range(2):
        for j in range(6):
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Linear') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[0], ax = axes[j][i%2], label = 'Lanchester linear')
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Exponential') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[1], ax = axes[j][i%2], label = 'Lanchester power')
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Mixed') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[2], ax = axes[j][i%2], label = 'Lanchester mixed')
            
            axes[j][i%2].set_xticklabels(new_order)
            axes[j][i%2].set_ylim(0.5, 1.0)
            axes[j][i%2].set_yticks([0.6, 0.8, 1.0])
            
            axes[j][i%2].set_title(f'{j*5+5} minutes', fontsize = 15)
            
            axes[j][i%2].legend(loc = 'lower center', bbox_to_anchor = (-0.15, -1), ncol = 3, fontsize = 17)
            
            if (i < 1) or (j != 5):
                axes[j][i%2].legend([], [], frameon=False)
                
            axes[j][i%2].set(xlabel=None, ylabel = None)

            axes[j][i%2].xaxis.set_tick_params(labelsize=15)
            axes[j][i%2].yaxis.set_tick_params(labelsize=15)

    plt.subplots_adjust(hspace = 0.8)
    fig.figure.savefig(f'results/fig7/Score_tier_timelen_lantype_{list_col_sub[i-1]}_{list_col_sub[i]}.png', dpi = 200)

    list_col_sub = list_col[2:]

    fig, axes = plt.subplots(6, 2, figsize = (12, 15))

    for i in range(2):
        for j in range(6):
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Linear') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[0], ax = axes[j][i%2], label = 'Lanchester linear')
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Exponential') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[1], ax = axes[j][i%2], label = 'Lanchester power')
            
            sns.lineplot(x = 'tier', y = 'CrossValMeans', data = dfs[(dfs['time_len'] == j*5 + 5) & 
                     (dfs['data_type'] == 'lanchester') & 
                     (dfs['scaler'] == 'Mixed') & 
                     (dfs['col'] == list_col_sub[i])],
                     marker = 'o', linestyle = '-', color = colors[2], ax = axes[j][i%2], label = 'Lanchester mixed')
            
            axes[j][i%2].set_xticklabels(new_order)
            axes[j][i%2].set_ylim(0.5, 1.0)
            axes[j][i%2].set_yticks([0.6, 0.8, 1.0])
            
            axes[j][i%2].set_title(f'{j*5+5} minutes', fontsize = 15)
            
            axes[j][i%2].legend(loc = 'lower center', bbox_to_anchor = (-0.15, -1), ncol = 3, fontsize = 17)
            
            if (i < 1) or (j != 5):
                axes[j][i%2].legend([], [], frameon=False)
                
            axes[j][i%2].set(xlabel=None, ylabel = None)

            axes[j][i%2].xaxis.set_tick_params(labelsize=15)
            axes[j][i%2].yaxis.set_tick_params(labelsize=15)

    plt.subplots_adjust(hspace = 0.8)
    fig.figure.savefig(f'results/fig7/Score_tier_timelen_lantype_{list_col_sub[i-1]}_{list_col_sub[i]}.png', dpi = 200)
    
def result_predict_time(list_tier, list_time_len, list_col, list_lan_type, ):

    dic_predict_time = {}
    pbar_list_tier = tqdm(list_tier, position = 0)
    for tier in pbar_list_tier:
        dic_predict_time[tier] = {}

        df_all = pd.read_csv(f'Data/ALL/{tier}/df_all.csv')
        list_match_id = list(df_all.loc[df_all['timestamp'] > time_len * 60000]['match_id'].drop_duplicates())
        for time_len in list_time_len:
            dic_predict_time[tier][time_len] = {}
            for col in list_col + ['4col']:
                dic_predict_time[tier][time_len][col] = {}
                for i, lan_type in enumerate(list_lan_type):
                    dic_predict_time[tier][time_len][col][lan_type] = {}

                    if col == '4col':
                        df_lan = pd.DataFrame()
                        for col2 in list_col[:-1]:
                        
                            df_lan_sub = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col2}_{time_len}.csv')
                            df_lan_sub = df_lan_sub.head(8000)
                            
                            df_lan = pd.concat([df_lan, df_lan_sub.iloc[:, :-2]], axis = 1)
                            
                        df_lan['win'] = df_lan_sub['win']
                        df_lan['res'] = df_lan_sub['Residuals']

                    else:
                        df_lan = pd.read_csv(f'Data/Lanchester/{tier}/df_lsq_{i+1}_{col}_{time_len}.csv')
                    
                    x = df_lan.iloc[:, :-2]
                    y = np.array(df_lan['win'])

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
                    scaler = StandardScaler()
                    
                    x_train_scaled = scaler.fit_transform(x_train)
                    
                    for model_name in list_model:
                        dic_predict_time[tier][time_len][col][lan_type][model_name] = {}
                            
                        file_name = f'models/lanchester/{tier}/{model_name}_{time_len}_{col}_{lan_type}.pkl'
                        model = pickle.load(open(file_name, 'rb'))
                        
                        for match_id in list_match_id[:10]:
                            pbar_list_tier.set_description(f'{tier}_{time_len}_{col}_{lan_type}_{model_name}_{match_id}')
                            
                            if col == '4col':
                                list_df = []
                                list_df_diff = []
                                for col2 in list_col[:-1]:
                                    df, df_diff = Make_DataFrame.make_array(df_all, match_id, time_len, col2)
                                    list_df.append(df)
                                    list_df_diff.append(df_diff)
                            else:
                                df, df_diff = Make_DataFrame.make_array(df_all, match_id, time_len, col)
                            
                            if len(df) < 3:
                                continue
                                
                            start_time = time.perf_counter()
                                
                            for outcome in ['win', 'lose']:
                                if col == '4col':
                                    list_coef = []
                                    for df_, df_diff_ in zip(list_df, list_df_diff):
                                        if i == 0:
                                            alpha, beta, gamma, res = Make_DataFrame.least_square_1(df_, df_diff_, outcome)
                                            list_coef += [alpha, beta, gamma]
                                        if i == 1:
                                            alpha, beta, gamma, res = Make_DataFrame.least_square_2(df_, df_diff_, outcome)
                                            list_coef += [alpha, beta, gamma]
                                        if i == 2:
                                            alpha, beta, gamma, delta, res = Make_DataFrame.least_square_3(df_, df_diff_, outcome)
                                            list_coef += [alpha, beta, gamma, delta]
                                else:
                                    if i == 0:
                                        alpha, beta, gamma, res = Make_DataFrame.least_square_1(df, df_diff, outcome)
                                    elif i == 1:
                                        alpha, beta, gamma, res = Make_DataFrame.least_square_2(df, df_diff, outcome)
                                    else:
                                        alpha, beta, gamma, delta, res = Make_DataFrame.least_square_3(df, df_diff, outcome)
                                        
                            if col == '4col':
                                test_val = scaler.transform([list_coef])
                            else:
                                if i == 0:
                                    test_val = scaler.transform([[alpha, beta, gamma]])
                                elif i == 1:
                                    test_val = scaler.transform([[alpha, beta, gamma]])
                                else:
                                    test_val = scaler.transform([[alpha, beta, gamma, delta]])
                                
                            y_pred = model.predict(test_val)
                            end_time = time.perf_counter()
                        
                            dic_predict_time[tier][time_len][col][lan_type][model_name][match_id] = end_time - start_time

    list_predict_time = []

    for tier in pbar_list_tier:
        for time_len in list_time_len:
            for col in list_col + ['4col']:
                for lan_type in list_lan_type:
                    for model_name in list_model:
                        for match_id, predict_time in dic_predict_time[tier][time_len][col][lan_type][model_name].items():
                            list_predict_time += [[tier, time_len, col, lan_type, model_name, match_id, predict_time]]

    df_predict_time = pd.DataFrame(list_predict_time, columns = ['tier', 'time_len', 'col', 'lan_type', 'model_name', 'match_id', 'training_time'])
    
    df_predict_time.to_csv('results/predict_time.csv', index = False)
 
def result_count(list_tier, list_time_len, new_order, new_names):
    list_count = []

    for tier in list_tier:
        for time_len in list_time_len[3:]:
            if dt == 'timeseries':
                df = pd.read_csv(f'Data/{dt}/{tier}/df_{dt}_{time_len}_xp.csv')

            elif dt == 'lanchester':
                df = pd.read_csv(f'Data/{dt}/{tier}/df_lsq_1_xp_{time_len}.csv')

            else:
                df = pd.read_csv(f'Data/{dt}/{tier}/df_{dt}_{time_len}.csv')

            list_count.append([tier, dt, time_len, len(df)])

    df_c = pd.DataFrame(list_count, columns = ['tier', 'data_type', 'time_len', 'count'])
    pivot_c = df_c.pivot_table(values='count', index='tier', columns='time_len', aggfunc='sum')

    pivot_c = pivot_c.rename(index=new_names)
    pivot_c = pivot_c.loc[new_order]

    pivot_c.to_csv('results/count.csv')