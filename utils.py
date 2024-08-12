import os
import zipfile
import json
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor, Lasso

import warnings
warnings.filterwarnings('ignore')

class Load_files:

    def __init__(self, tier, list_division):
    
        self.tier = tier
        self.list_division = list_division
        
    def load_matchid(self):
    
        tier = self.tier 
        list_division = self.list_division
        
        dic_match_id = {div:[] for div in list_division}
        list_path = []

        for division in list_division:
            if division != 'ALL':
                file_path = f'Data/Raw/{tier}/{division}/match_list.txt'
            else:
                file_path = f'Data/Raw/{tier}/match_list.txt'

            match_list = []
            with open(file_path, 'r') as f:
                for match_id in f:
                    match_list.append(match_id[:-1])
                    dic_match_id[division] = match_list
                    
        return dic_match_id
        
    def load_timestep(self):
            
        tier = self.tier 
        list_division = self.list_division
        
        list_path = []
        for division in list_division:
            if division != 'ALL':
                list_path += [f'Data/Raw/{tier}/{division}/timeline_json.json']
            else:
                list_path += [f'Data/Raw/{tier}/timeline_json.json']
                
        dic_timestep = {}

        for file_path in list_path:
            tier = file_path.split('/')[2]
            
            if len(file_path.split('/')) == 5:
                div = file_path.split('/')[3]
            else:
                div = 'ALL'
            
            with open(file_path, 'r') as f:
                file_timestep = json.load(f)
                dic_timestep = {**dic_timestep, **file_timestep}
                
        return dic_timestep

class Make_All:

    def __init__(self, dic_timestep):
        
        self.dic_timestep = dic_timestep
    
    def make_timestep(match_id, timestep):
        
        list_timestep = []

        num_list = ['1', '2', '3', '4', '5',
                    '6', '7', '8', '9', '10'] 

        for num in num_list:

            for i in range(len(timestep['info']['frames'])):

                dic_timestep_sub = {}
                for key1, val1 in timestep['info']['frames'][i]['participantFrames'][num].items():

                    if key1 in ['championStats', 'damageStats']:

                        for key2, val2 in timestep['info']['frames'][i]['participantFrames'][num][key1].items():
                            dic_timestep_sub[key2] = val2

                    elif key1 in ['position']:
                        dic_timestep_sub[key1] = [[val1['x'], val1['y']]]

                    else:
                        dic_timestep_sub[key1] = val1

                    dic_timestep_sub['timestamp'] = timestep['info']['frames'][i]['timestamp']

                list_timestep.append(dic_timestep_sub)

        df_timestep = pd.DataFrame(list_timestep)
        df_timestep['match_id'] = match_id
        
        return df_timestep

    def concat_event(timestep, df_timestep):
        
        df_timestep[['Riftherald', 'Dragon', 'Baron_Nashor', 'Elder_Dragon', 'Ward_Y', 'Ward_B', 'Ward_S', 'Ward_C', 'win']] = 0
        df_timestep['Tower_Plate'] = 15
        df_timestep['Tower'] = 11
        df_timestep['Inhibitor'] = 3

        dic_level = {i:1 for i in range(1, 11)}
        
        for i in range(len(timestep['info']['frames'])):
            
            for dic in timestep['info']['frames'][i]['events']:
                
                if dic['type'] == 'LEVEL_UP':
                    dic_level[dic['participantId']] = dic['level']
                
                if dic['type'] == 'ELITE_MONSTER_KILL':
                    
                    if dic['monsterType'] == 'RIFTHERALD':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == dic['killerTeamId']//100), 
                                        'Riftherald'] += 1

                    if dic['monsterType'] == 'BARON_NASHOR':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        (df_timestep['timestamp'] < dic['timestamp'] + (180 * 1000)) &
                                        ((df_timestep['participantId'] + 4)//5 == dic['killerTeamId']//100), 
                                        'Baron_Nashor'] = 1

                    if dic['monsterType'] == 'DRAGON':
                        
                        if dic['monsterSubType'] == 'ELDER_DRAGON':
                            df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        (df_timestep['timestamp'] < dic['timestamp'] + (150 * 1000)) &
                                        ((df_timestep['participantId'] + 4)//5 == dic['killerTeamId']//100), 
                                        'Elder_Dragon'] = 1

                        else:
                            df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == dic['killerTeamId']//100), 
                                        'Dragon'] += 1

                if dic['type'] == 'BUILDING_KILL':
                    
                    if dic['buildingType'] == 'TOWER_BUILDING':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == dic['teamId']//100), 
                                        'Tower'] -= 1
                        
                    if dic['buildingType'] == 'INHIBITOR_BUILDING':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        (df_timestep['timestamp'] < (dic['timestamp'] + (300 * 1000))) &
                                        ((df_timestep['participantId'] + 4)//5 == dic['teamId']//100), 
                                        'Inhibitor'] -= 1
                        
                if dic['type'] == 'TURRET_PLATE_DESTROYED':
                    df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                    ((df_timestep['participantId'] + 4)//5 == dic['teamId']//100), 
                                    'Tower_Plate'] -= 1
                    
                if dic['type'] == 'WARD_PLACED':
                        
                    if dic['wardType'] == 'YELLOW_TRINKET':
                        level_mean = sum(dic_level.values())/10
                        ward_time = (90 + (1.76 * (level_mean - 1))) * 1000
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        (df_timestep['timestamp'] < (dic['timestamp'] + ward_time)) &
                                        ((df_timestep['participantId'] + 4)//5 ==(dic['creatorId'] + 4)//5), 
                                        'Ward_Y'] += 1
                        
                    if dic['wardType'] == 'SIGHT_WARD':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        (df_timestep['timestamp'] < (dic['timestamp'] + (150 * 1000))) &
                                        ((df_timestep['participantId'] + 4)//5 == (dic['creatorId'] + 4)//5), 
                                        'Ward_S'] += 1                    
                        
                    if dic['wardType'] == 'BLUE_TRINKET':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == (dic['creatorId'] + 4)//5), 
                                        'Ward_B'] += 1
                        
                    if dic['wardType'] == 'CONTROL_WARD':
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == (dic['creatorId'] + 4)//5), 
                                        'Ward_C'] += 1

                if dic['type'] == 'WARD_KILL':
                    
                    if dic['wardType'] == 'BLUE_TRINKET':
                        if dic['killerId'] == 0:
                            continue
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == int(2/((dic['killerId'] + 4)//5))), 
                                        'Ward_B'] -= 1
                        
                    if dic['wardType'] == 'CONTROL_WARD':
                        if dic['killerId'] == 0:
                            continue
                        df_timestep.loc[(df_timestep['timestamp'] >= dic['timestamp']) & 
                                        ((df_timestep['participantId'] + 4)//5 == int(2/((dic['killerId'] + 4)//5))), 
                                        'Ward_C'] -= 1                    
                
                if dic['type'] == 'GAME_END':
                    df_timestep.loc[(df_timestep['participantId'] + 4)//5 == dic['winningTeam']//100, 'win'] = 1
                 
        # 포탑 방패는 15분이 되면 사라짐
        df_timestep.loc[df_timestep['timestamp']/60000 >= 15, 'Tower_Plate'] = 0

        df_timestep.loc[df_timestep['Ward_B'] < 0, 'Ward_B'] = 0
        df_timestep.loc[df_timestep['Ward_B'] > 5, 'Ward_B'] = 5
        df_timestep.loc[df_timestep['Ward_C'] < 0, 'Ward_C'] = 0
        df_timestep.loc[df_timestep['Ward_C'] > 5, 'Ward_C'] = 5
        df_timestep['Ward'] = df_timestep['Ward_Y'] + df_timestep['Ward_S'] + df_timestep['Ward_B'] + df_timestep['Ward_C']
        
        df_timestep = df_timestep.drop(['Ward_Y', 'Ward_S', 'Ward_B', 'Ward_C'], axis = 1)  
        
        return df_timestep

    def edit_dead(timestep, df_timestep):
        
        dic_level = {i:1 for i in range(1, 11)}
        
        for i in range(len(timestep['info']['frames'])):
            for dic in timestep['info']['frames'][i]['events']:
                if dic['type'] == 'LEVEL_UP':
                    dic_level[dic['participantId']] = dic['level']

                if dic['type'] == 'CHAMPION_KILL':

                    if dic_level[dic['victimId']] < 7:
                        BRW = (dic_level[dic['victimId']] * 2 + 4)
                    if dic_level[dic['victimId']] == 7:
                        BRW = (16 + 5)
                    if dic_level[dic['victimId']] > 7:
                        BRW = (dic_level[dic['victimId']] * 2.5 + 7.5) 

                    death_time = dic['timestamp']

                    if dic['timestamp'] < 15 * 60000:
                        TIF = 0

                    elif dic['timestamp'] < 30 * 60000:
                        TIF = (0 + math.ceil(2 * (dic['timestamp']/60000 - 15)) * 0.425) / 100

                    elif dic['timestamp'] < 45 * 60000:
                        TIF = (12.75 + math.ceil(2 * (dic['timestamp']/60000 - 30)) * 0.30) / 100

                    elif dic['timestamp'] < 55 * 60000:
                        TIF = (21.75 + math.ceil(2 * (dic['timestamp']/60000 - 45)) * 1.45) / 100

                    else:
                        TIF = 78.75


                    dead_time = (BRW + BRW * TIF) * 1000
                    res_time = death_time + dead_time
                    
                    df_timestep.loc[(df_timestep['participantId'] == dic['victimId']) & 
                                    (df_timestep['Baron_Nashor'] == 1) & 
                                    (df_timestep['timestamp'] > death_time) & 
                                    (df_timestep['timestamp'] < res_time), 'Baron_Nashor'] = 0

                    df_timestep.loc[(df_timestep['participantId'] == dic['victimId']) & 
                                    (df_timestep['Elder_Dragon'] == 1) & 
                                    (df_timestep['timestamp'] > death_time) & 
                                    (df_timestep['timestamp'] < res_time), 'Elder_Dragon'] = 0
                    
        return df_timestep

    def make_all(self):
        
        dic_timestep = self.dic_timestep
        
        list_all_timestep = []

        pbar_all = tqdm(dic_timestep.items(), position = 1)
        for match_id, timestep in pbar_all:
            pbar_all.set_description('Make DataFrame - ALL')
            
            if timestep['info']['frames'][-1]['events'][-1]['timestamp']/60000 < 16:
                continue
                
            df_timestep = Make_All.make_timestep(match_id, timestep)
            df_timestep = Make_All.concat_event(timestep, df_timestep)
            df_timestep = Make_All.edit_dead(timestep, df_timestep)
            list_all_timestep.append(df_timestep)
            
        df_all = pd.concat(list_all_timestep, axis = 0).reset_index(drop = True)
        df_all['spendGold'] = df_all['totalGold'] - df_all['currentGold']
        
        return df_all
     
class Make_DataFrame():

    def __init__(self, df_all, dic_period, drop_col, use_col):
    
        self.df_all = df_all
        self.df_period = pd.DataFrame(dic_period)
        self.drop_col = drop_col
        self.use_col = use_col

    def win_lose_ratio(df):
        
        list_ratio = []

        for i in range(len(df)):
            if i % 2 == 0:
                list_ratio.append(df.iloc[i]/(df.iloc[i] + df.iloc[i+1]))

            else:
                list_ratio.append(df.iloc[i]/(df.iloc[i] + df.iloc[i-1]))

        df_ratio = pd.DataFrame(list_ratio)
        df_ratio = df_ratio.fillna(0.5)
        df_ratio = df_ratio.map(lambda x: 1 if x > 1 else (0 if x < 0 else x))
        
        return df_ratio

    def make_array(df_all, match_id, time_len, col):

        df_match = df_all.loc[df_all['match_id'] == match_id][[col, 'participantId', 'win']].reset_index(drop = True)

        if df_match['win'][0] == 1:
            T1 = 1
            T2 = 6
        else:
            T1 = 6
            T2 = 1

        for i in range(5):
            if i == 0:
                array_tDD_win = np.array(df_match.loc[df_match['participantId'] == i+T1][col])
                array_tDD_lose = np.array(df_match.loc[df_match['participantId'] == i+T2][col])
            else:
                array_tDD_win += np.array(df_match.loc[df_match['participantId'] == i+T1][col])
                array_tDD_lose += np.array(df_match.loc[df_match['participantId'] == i+T2][col])

        df = pd.DataFrame([array_tDD_win, array_tDD_lose], index = ['win', 'lose']).T[:time_len + 1]

        df_diff = df.diff().fillna(0)
        
        df = df[(df.T != 0).any()]
        df['C'] = 1
        df = df[['C', 'win', 'lose']]
        
        df_diff = df_diff[(df_diff.T != 0).any()]

        inter_row = df.index.intersection(df_diff.index)
        df = df.loc[inter_row]
        df_diff = df_diff.loc[inter_row]        

        return df, df_diff

    def least_square_1(df, df_diff, outcome):
        
        A = np.array(df[['C', 'win', 'lose']])
        B = np.array(df_diff[outcome])      

        lstsq = np.linalg.lstsq(A, B, rcond=None)

        x, res = lstsq[0], lstsq[1].sum()
        
        alpha, beta, gamma = x[0], x[1], x[2]
                 
        return alpha, beta, gamma, res
        
    def least_square_2(df, df_diff, outcome):
        
        df = df[(df != 0).all(axis=1)]
        df_diff = df_diff[(df_diff != 0).all(axis=1)]
        
        inter_row = df.index.intersection(df_diff.index)
        df = df.loc[inter_row]
        df_diff = df_diff.loc[inter_row]
        
        A = np.array(df[['C', 'win', 'lose']])
        B = np.array(df_diff[outcome])

        logA = np.log(A)
        logA[:, 0] = 1
        logB = np.log(B)

        lstsq = np.linalg.lstsq(logA, logB, rcond=None)
        
        x, res = lstsq[0], lstsq[1].sum()
        
        alpha, beta, gamma = x[0], x[1], x[2]
        
        return alpha, beta, gamma, res
        
    def least_square_3(df, df_diff, outcome):
        
        df['wl'] = df['win'] * df['lose']
        
        A = np.array(df[['C', 'win', 'lose', 'wl']])
        B = np.array(df_diff[outcome])      

        lstsq = np.linalg.lstsq(A, B, rcond=None)
        
        x, res = lstsq[0], lstsq[1].sum()
            
        alpha, beta, gamma, delta = x[0], x[1], x[2], x[3]
                 
        return alpha, beta, gamma, delta, res

    def lanchester_lsq(self, time_len, col):
    
        df_all = self.df_all

        list_match_id = list(df_all.loc[df_all['timestamp'] > time_len * 60000]['match_id'].drop_duplicates())

        list_lsq_1 = []
        list_lsq_2 = []
        list_lsq_3 = []
            
        pbar_list_match_id = tqdm(list_match_id, position = 2)
        for match_id in pbar_list_match_id:
            
            df, df_diff = Make_DataFrame.make_array(df_all, match_id, time_len, col)
            
            if len(df) < 3:
                continue
            
            for outcome in ['win', 'lose']:
            
                alpha1, beta1, gamma1, res1 = Make_DataFrame.least_square_1(df, df_diff, outcome)
                list_lsq_1.append([alpha1, beta1, gamma1, res1, 1 if outcome == 'win' else 0])
                
                alpha2, beta2, gamma2, res2 = Make_DataFrame.least_square_2(df, df_diff, outcome)
                list_lsq_2.append([alpha2, beta2, gamma2, res2, 1 if outcome == 'win' else 0])
                
                if len(df) < 4:
                    continue
                    
                alpha3, beta3, gamma3, delta3, res3 = Make_DataFrame.least_square_3(df, df_diff, outcome)
                list_lsq_3.append([alpha3, beta3, gamma3, delta3, res3, 1 if outcome == 'win' else 0])
                
        pbar_list_match_id.close()
        
        df_lsq_1 = pd.DataFrame(list_lsq_1, columns = [['Alpha', 'Beta', 'Gamma', 'Residuals', 'win']])
        df_lsq_2 = pd.DataFrame(list_lsq_2, columns = [['Alpha', 'Beta', 'Gamma', 'Residuals', 'win']])
        df_lsq_3 = pd.DataFrame(list_lsq_3, columns = [['Alpha', 'Beta', 'Gamma', 'Delta', 'Residuals', 'win']])

        return [df_lsq_1, df_lsq_2, df_lsq_3]

    def make_mean(self, time_len):

        df = self.df_all
        drop_col = self.drop_col
        use_col = self.use_col
        
        threshold = time_len * 60000
        
        df = df.drop(drop_col, axis = 1)
        
        list_last_index = df.index[df['participantId'] != df['participantId'].shift(-1)].tolist()
        last_timestamp = df.loc[list_last_index, 'timestamp']

        invalid_match_ids = df.loc[list_last_index, 'match_id'][last_timestamp < threshold]
        df_t = df[~df['match_id'].isin(invalid_match_ids)]

        df_t = df_t[df_t['timestamp'] < (time_len + 1) * 60000]

        df_t = df_t.groupby(['match_id', 'win']).agg('mean').reset_index()
        df_t = df_t.sort_values(by = 'match_id')

        list_match_id_t = list(df_t['match_id'])

        df_mean = df_t[use_col]
        df_mean = df_mean.drop(['match_id', 'participantId', 'timestamp'], axis = 1)
        df_mean_ratio = Make_DataFrame.win_lose_ratio(df_mean)

        df_mean['match_id'] = list_match_id_t
        df_mean_ratio['match_id'] = list_match_id_t
        
        return df_mean, df_mean_ratio
        
    def make_weightedmean(self, time_len):
        
        df = self.df_all
        drop_col = self.drop_col
        use_col = self.use_col
        df_period = self.df_period
     
        threshold = time_len * 60000
        
        df = df.drop(drop_col, axis = 1)

        list_last_index = df.index[df['participantId'] != df['participantId'].shift(-1)].tolist()
        last_timestamp = df.loc[list_last_index, 'timestamp']

        invalid_match_ids = df.loc[list_last_index, 'match_id'][last_timestamp < threshold]
        df_t = df[~df['match_id'].isin(invalid_match_ids)]
        
        df_t['time'] = 0
        df_t.loc[df['timestamp']/60000 >= 15, 'time'] = 1
        df_t.loc[df['timestamp']/60000 >= 25, 'time'] = 2

        df_t = df_t[df_t['timestamp'] < (time_len + 1) * 60000]
        df_t = df_t.groupby(['match_id', 'time', 'participantId']).agg('mean').reset_index()

        df_t = df_t.set_index(['time', (df_t['participantId']-1)%5])

        df_period = df_period.melt(var_name = 'time', value_name = 'weight')
        df_period['participantId'] = df_period.index % 5

        df_period = df_period.set_index(['time', 'participantId'])

        for col in use_col:
            if col in ['win', 'match_id', 'participantId']:
                df_period[col] = 1
            elif col in ['Riftherald', 'Dragon', 'Baron_Nashor', 'Elder_Dragon', 'Ward', 'Tower', 'Tower_Plate', 'Inhibitor', 'timestamp']:
                df_period[col] = 0.2
            else:
                df_period[col] = df_period['weight']

        df_period = df_period.drop(['weight'], axis = 1)

        df_t = df_t * df_period
        df_t = df_t.groupby(['match_id', 'win']).agg('sum').reset_index()
        
        df_t = df_t.sort_values(by = 'match_id')
        list_match_id_t = list(df_t['match_id'])

        df_weightedmean = df_t[use_col]
        df_weightedmean = df_weightedmean.drop(['match_id', 'participantId', 'timestamp'], axis = 1)
        df_weightedmean_ratio = Make_DataFrame.win_lose_ratio(df_weightedmean)

        df_weightedmean['match_id'] = list_match_id_t
        df_weightedmean_ratio['match_id'] = list_match_id_t
        
        return df_weightedmean, df_weightedmean_ratio
        
    def make_point(self, time_len):
    
        df = self.df_all
        drop_col = self.drop_col
        use_col = self.use_col
        
        threshold = time_len * 60000
        
        df = df.drop(drop_col, axis = 1)
        
        list_last_index = df.index[df['participantId'] != df['participantId'].shift(-1)].tolist()
        last_timestamp = df.loc[list_last_index, 'timestamp']

        invalid_match_ids = df.loc[list_last_index, 'match_id'][last_timestamp < threshold]
        df_t = df[~df['match_id'].isin(invalid_match_ids)]

        df_t = df_t.loc[round(df_t['timestamp'] / 60000, 1) == time_len]
            
        df_t = df_t.groupby(['match_id', 'win']).agg('mean').reset_index()
        df_t = df_t.sort_values(by = 'match_id')

        list_match_id_t = list(df_t['match_id'])
        
        df_t = df_t[use_col]
        
        df_point = df_t.drop(['match_id', 'participantId', 'timestamp'], axis = 1)

        df_point_ratio = Make_DataFrame.win_lose_ratio(df_point)

        df_point['match_id'] = list_match_id_t
        df_point_ratio['match_id'] = list_match_id_t
            
        return df_point, df_point_ratio    
        
    def make_timeseries(self, time_len, col):
        
        df = self.df_all
        
        threshold = time_len * 60000
        
        list_last_index = df.index[df['participantId'] != df['participantId'].shift(-1)].tolist()
        last_timestamp = df.loc[list_last_index, 'timestamp']

        invalid_match_ids = df.loc[list_last_index, 'match_id'][last_timestamp < threshold]
        df_t = df[~df['match_id'].isin(invalid_match_ids)]
        
        #list_last_index_t = df_t.index[df_t['participantId'] != df_t['participantId'].shift(-1)].tolist()
        #df_t = df_t.drop(list_last_index_t)

        df_t = df_t.loc[round(df_t['timestamp'] / 60000, 1) <= time_len]
        df_t['time'] = round(df_t['timestamp']/60000, 1)
        
        df_t = df_t.groupby(['match_id', 'win', 'time']).sum()[[col]]
        df_t = df_t.unstack().reset_index()
        
        list_col = ['match_id', 'win'] + [i for i in range(time_len + 1)]
        df_t.columns = list_col
        df_t = df_t[list_col[2:] + list_col[:2]]
                
        df_t = df_t.sort_values(by = 'match_id')
        list_match_id = [match_id for match_id in df_t['match_id']]
        df_timeseries = df_t.drop(['match_id'], axis = 1)
        df_timeseries_ratio = Make_DataFrame.win_lose_ratio(df_timeseries)

        df_timeseries['match_id'] = list_match_id
        df_timeseries_ratio['match_id'] = list_match_id
        
        return df_timeseries, df_timeseries_ratio
        
def zip_directory(dt):

    folder_path = f'Data/{dt}'
    zip_path = f'Data/{dt}.zip' 
    
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    total_files = len(file_list)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        pbar_file = tqdm(enumerate(file_list))
        for i, file in pbar_file:
            pbar_file.set_description(f"Compressing file {i + 1}/{total_files} ({((i + 1) / total_files) * 100:.2f}%)")
            arcname = os.path.relpath(file, folder_path)
            zipf.write(file, arcname)
        pbar_file.close()