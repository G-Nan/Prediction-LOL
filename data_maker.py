from utils import *
        
def data_maker_all(dic_tier):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : All / tier : {tier}')
        
        if not os.path.exists(f'Data/All/{tier}/'):
            os.makedirs(f'Data/All/{tier}/')
            
        LF = Load_files(tier, list_division)
        dic_timestep = LF.load_timestep()

        MA = Make_All(dic_timestep)
        df_all = MA.make_all()
        df_all.to_csv(f'Data/All/{tier}/df_all.csv', index = False)
    pbar_tier.close()
    
def data_maker_mean(dic_tier, drop_col, use_col, dic_period, list_time_len):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : Mean / tier : {tier}')
        
        if not os.path.exists(f'Data/Mean/{tier}/'):
            os.makedirs(f'Data/Mean/{tier}/')
            
        df_all = pd.read_csv(f'Data/All/{tier}/df_all.csv')
        
        MD = Make_DataFrame(df_all, dic_period, drop_col, use_col)
        
        pbar_mean = tqdm(list_time_len)
        for time_len in pbar_mean:
            pbar_mean.set_description(f'Make DataFrame - Mean / {time_len}')
            
            df_mean, df_mean_ratio = MD.make_mean(time_len)

            df_mean.to_csv(f'Data/Mean/{tier}/df_mean_{time_len}.csv', index = False)
            df_mean_ratio.to_csv(f'Data/Mean/{tier}/df_mean_ratio_{time_len}.csv', index = False)
    pbar_tier.close()
    
def data_maker_weightedmean(dic_tier, drop_col, use_col, dic_period, list_time_len):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : Weighted Mean / tier : {tier}')
        
        if not os.path.exists(f'Data/WeightedMean/{tier}/'):
            os.makedirs(f'Data/WeightedMean/{tier}/')
            
        df_all = pd.read_csv(f'Data/All/{tier}/df_all.csv')
        
        MD = Make_DataFrame(df_all, dic_period, drop_col, use_col)

        pbar_time = tqdm(list_time_len)
        for time_len in pbar_time:
            pbar_time.set_description(f'Make DataFrame - WeightedMean/ {time_len}')

            df_weightedmean, df_weightedmean_ratio = MD.make_weightedmean(time_len)

            df_weightedmean.to_csv(f'Data/WeightedMean/{tier}/df_weightedmean_{time_len}.csv', index = False)
            df_weightedmean_ratio.to_csv(f'Data/WeightedMean/{tier}/df_weightedmean_ratio_{time_len}.csv', index = False)
    pbar_tier.close()
    
def data_maker_point(dic_tier, drop_col, use_col, dic_period, list_time_len):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : Point / tier : {tier}')
        
        if not os.path.exists(f'Data/Point/{tier}/'):
            os.makedirs(f'Data/Point/{tier}/')
            
        df_all = pd.read_csv(f'Data/All/{tier}/df_all.csv')
        
        MD = Make_DataFrame(df_all, dic_period, drop_col, use_col)
        
        pbar_point = tqdm(list_time_len)
        for time_len in pbar_point:
            pbar_point.set_description(f'Make DataFrame - Point / {time_len}')
                
            df_point, df_point_ratio = MD.make_point(time_len)
        
            df_point.to_csv(f'Data/Point/{tier}/df_point_{time_len}.csv', index = False)
            df_point_ratio.to_csv(f'Data/Point/{tier}/df_point_ratio_{time_len}.csv', index = False)
    pbar_tier.close()
    
def data_maker_timeseries(dic_tier, drop_col, use_col, dic_period, list_time_col):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : Timeseries / tier : {tier}')
        
        if not os.path.exists(f'Data/Timeseries/{tier}/'):
            os.makedirs(f'Data/Timeseries/{tier}/')
            
        df_all = pd.read_csv(f'Data/All/{tier}/df_all.csv')
        
        MD = Make_DataFrame(df_all, dic_period, drop_col, use_col)
        
        pbar_lanchester = tqdm(list_time_col)
        for i in pbar_lanchester:
            time_len = i[0]
            col = i[1]
            pbar_lanchester.set_description(f'Make DataFrame - Timeseries / {time_len} / {col}')

            df_timeseries, df_timeseries_ratio = MD.make_timeseries(time_len, col)
            
            df_timeseries.to_csv(f'Data/Timeseries/{tier}/df_timeseries_{time_len}_{col}.csv', index = False)
            df_timeseries_ratio.to_csv(f'Data/Timeseries/{tier}/df_timeseries_ratio_{time_len}_{col}.csv', index = False)
    pbar_tier.close()
    
def data_maker_lanchester(dic_tier, drop_col, use_col, dic_period, list_time_col):
    
    pbar_tier = tqdm(dic_tier.items(), position = 0)
    for tier, list_division in pbar_tier:
        pbar_tier.set_description(f' Making dataset - data type : Lanchester / tier : {tier}')

        if not os.path.exists(f'Data/Lanchester/{tier}/'):
            os.makedirs(f'Data/Lanchester/{tier}/')
            
        df_all = pd.read_csv(f'Data/All/{tier}/df_all.csv')
        
        MD = Make_DataFrame(df_all, dic_period, drop_col, use_col)
        
        pbar_lanchester = tqdm(list_time_col, position = 1)
        for time_len, col in pbar_lanchester:
            pbar_lanchester.set_description(f'Make DataFrame - Lanchester LeastSquare / {time_len} / {col}')

            list_df = MD.lanchester_lsq(time_len, col)
            for j, df in enumerate(list_df):
                df.to_csv(f'Data/Lanchester/{tier}/df_lsq_{j+1}_{col}_{time_len}.csv', index = False)
        pbar_lanchester.close()
    pbar_tier.close()