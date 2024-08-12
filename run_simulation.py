import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from api_loader import *
from data_maker import *
from data_loader import *
from models import *
from trainer import *
from utils import *
from constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path = ".hydra", config_name = "config.yaml", version_base = None)
def main(cfg: DictConfig):
    if cfg.run_type.load_api:
        print("Loading API...")
        
        # API KEY
        api_key = cfg.load_api.API_KEY
        
        # Save path
        save_path = cfg.load_api.save_path
                
        # Data type
        data_type = cfg.load_api.data_type
        allowed_values = ['PUUID', 'MATCH_ID', 'MATCH', 'TIMELINE', 'ALL']
                
        if data_type not in allowed_values:
            raise ValueError(f"Invalid value '{data_type}' for 'data_type'. Choose one of: {', '.join(allowed_values)}")
            
        # Tier
        list_tier = cfg.load_api.list_tier
            
        # Start, Count
        start_matchid = cfg.load_api.start_matchid
        count_matchid = cfg.load_api.count_matchid
        start_matchlist = cfg.load_api.start_matchlist
        count_matchlist = cfg.load_api.count_matchlist
        count_dataset = cfg.load_api.count_dataset
        
        page = 1
       
        # PUUID
        if data_type == 'PUUID':
            Load_PUUID(api_key, list_tier, page, save_path)

        # Match ID
        if data_type == 'MATCH_ID':
            Load_MatchID(api_key, list_tier, tiers, start_matchid, count_matchid, save_path)

        # Match
        if data_type == 'MATCH':
            Load_Match(api_key, list_tier, tiers, start_match, count_match, cound_dataset, save_path)
        
        # Timeline
        if data_type == 'TIMELINE':
            Load_Timeline(api_key, list_tier, tiers, save_path)
            
        # All
        if data_type == 'ALL':
            Load_PUUID(api_key, list_tier, tiers, page, save_path)
            Load_MatchID(api_key, list_tier, tiers, start_matchid, count_matchid, save_path)
            Load_Match(api_key, list_tier, tiers, start_match, count_match, cound_dataset, save_path)
            Load_Timeline(api_key, list_tier, tiers, save_path)
            
    if cfg.run_type.make_dataset:
        print("Making dataset...")
        
        # Save path
        save_path = cfg.make_dataset.save_path
        
        # Data type
        data_type = cfg.make_dataset.data_type
        type_allowed_values = ['ALL', 'MEAN', 'WEIGHTEDMEAN', 'POINT', 'TIMESERIES', 'LANCHESTER', 'TOTAL']
        
        # Tier
        list_tier = cfg.make_dataset.list_tier
        list_time_len = cfg.make_dataset.list_time_len
        dic_tier_selected = {t:dic_tier[t] for t in list_tier}
        
        for dt in data_type:
            if dt not in type_allowed_values:
                raise ValueError(f"Invalid value '{data_type}' for 'data_type'. Choose one of: {', '.join(type_allowed_values)}")
        
            if not os.path.exists(f'Data/{dt}/'):
                os.makedirs(f'Data/{dt}/')
                
            if dt == 'ALL':
                data_maker_all(dic_tier_selected)
                zip_directory(dt)
                
            if dt == 'MEAN':
                data_maker_mean(dic_tier_selected, drop_col, use_col, dic_period, list_time_len)
                zip_directory(dt)
                
            if dt == 'WEIGHTEDMEAN':
                data_maker_weightedmean(dic_tier_selected, drop_col, use_col, dic_period, list_time_len)
                zip_directory(dt)
                
            if dt == 'POINT':
                data_maker_point(dic_tier_selected, drop_col, use_col, dic_period, list_time_len)
                zip_directory(dt)
                
            if dt == 'TIMESERIES':
                data_maker_timeseries(dic_tier_selected, drop_col, use_col, dic_period, list_time_col)
                zip_directory(dt)
                
            if dt == 'LANCHESTER':
                data_maker_lanchester(dic_tier_selected, drop_col, use_col, dic_period, list_time_col)
                zip_directory(dt)
                
    if cfg.run_type.load_dataset:
        print("Loading dataset...")
        
        data_type = cfg.load_dataset.data_type
        data_download(data_type)
        
    if cfg.run_type.train:
        print("Training model...")
                
        classifiers = make_classifier(cfg.param_gridsearch.ml)
        list_params = make_nn_param(cfg.param_gridsearch.nn)
        
        if not os.path.exists('scores/'):
            os.makedirs('scores/')
            
        if not os.path.exists('models/'):
            os.makedirs('models/')
            
        if not os.path.exists('hyperparameters/'):
            os.makedirs('hyperparameters/')
        
        for dt in cfg.train.data_type:
                
            if dt == 'MEAN':

                if not os.path.exists('scores/mean/'):
                    os.makedirs('scores/mean/')
                    
                if not os.path.exists('models/mean/'):
                    os.makedirs('models/mean/')
                    
                list_col = cfg.train.MEAN.list_col
                list_tier = cfg.train.MEAN.list_tier
                list_time_len = cfg.train.MEAN.list_time_len
                rand_size = cfg.train.MEAN.rand_size
                
                if cfg.train.MEAN.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.MEAN.col_name:
                        for scaler_name in cfg.train.MEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_mean(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers)
                            
                if cfg.train.MEAN.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.MEAN.col_name:
                        for scaler_name in cfg.train.MEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_mean_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size)
                            
            if dt =='WEIGHTEDMEAN':
            
                if not os.path.exists('scores/weightedmean/'):
                    os.makedirs('scores/weightedmean/')
                    
                if not os.path.exists('models/weightedmean/'):
                    os.makedirs('models/weightedmean/')
                    
                list_col = cfg.train.WEIGHTEDMEAN.list_col
                list_tier = cfg.train.WEIGHTEDMEAN.list_tier
                list_time_len = cfg.train.WEIGHTEDMEAN.list_time_len
                rand_size = cfg.train.WEIGHTEDMEAN.rand_size
                
                if cfg.train.WEIGHTEDMEAN.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.WEIGHTEDMEAN.col_name:
                        for scaler_name in cfg.train.WEIGHTEDMEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_weightedmean(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers)
                            
                if cfg.train.WEIGHTEDMEAN.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.WEIGHTEDMEAN.col_name:
                        for scaler_name in cfg.train.WEIGHTEDMEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_weightedmean_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size)
                            
            if dt == 'POINT':
            
                if not os.path.exists('scores/point/'):
                    os.makedirs('scores/point/')
                    
                if not os.path.exists('models/point/'):
                    os.makedirs('models/point/')
                    
                list_col = cfg.train.POINT.list_col
                list_tier = cfg.train.POINT.list_tier
                list_time_len = cfg.train.POINT.list_time_len
                rand_size = cfg.train.POINT.rand_size
                
                if cfg.train.POINT.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.POINT.col_name:
                        for scaler_name in cfg.train.POINT.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_point(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers)
                            
                if cfg.train.POINT.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.POINT.col_name:
                        for scaler_name in cfg.train.POINT.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_point_All(list_tier, list_time_len, list_col, col_name, scaler_name, classifiers, rand_size)
                            
            if dt == 'TIMESERIES':
            
                if not os.path.exists('scores/timeseries/'):
                    os.makedirs('scores/timeseries/')
                    
                if not os.path.exists('models/timeseries/'):
                    os.makedirs('models/timeseries/')
                
                if not os.path.exists('hyperparameters/timeseries/'):
                    os.makedirs('hyperparameters/timeseries/')
                    
                list_col = cfg.train.TIMESERIES.list_col
                list_tier = cfg.train.TIMESERIES.list_tier
                list_time_len = cfg.train.TIMESERIES.list_time_len
                rand_size = cfg.train.TIMESERIES.rand_size
                
                if cfg.train.TIMESERIES.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.TIMESERIES.col_name:
                        for scaler_name in cfg.train.TIMESERIES.scaler_type:
                            for model_name in cfg.train.TIMESERIES.model_name:
                                print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}, Model : {model_name}")
                                Training_timeseries(list_tier, list_time_len, list_col, col_name, scaler_name, list_params, model_name, device)
                            
                if cfg.train.TIMESERIES.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.TIMESERIES.col_name:
                        for scaler_name in cfg.train.TIMESERIES.scaler_type:
                            for model_name in cfg.train.TIMESERIES.model_name:
                                print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}, Model : {model_name}")
                                Training_timeseries_All(list_tier, list_time_len, list_col, col_name, scaler_name, list_params, model_name, device, rand_size)
                            
            if dt == 'LANCHESTER':
            
                if not os.path.exists('scores/lanchester/'):
                    os.makedirs('scores/lanchester/')
                    
                if not os.path.exists('models/lanchester/'):
                    os.makedirs('models/lanchester/')
                
                list_col = cfg.train.LANCHESTER.list_col
                list_tier = cfg.train.LANCHESTER.list_tier
                list_time_len = cfg.train.LANCHESTER.list_time_len                
                list_lan_type = cfg.train.LANCHESTER.list_lan_type
                rand_size = cfg.train.LANCHESTER.rand_size
                
                if cfg.train.LANCHESTER.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.LANCHESTER.col_name:
                        for scaler_name in cfg.train.LANCHESTER.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_lanchester(list_tier, list_time_len, list_lan_type, list_col, col_name, scaler_name, classifiers)
                            
                if cfg.train.LANCHESTER.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.LANCHESTER.col_name:
                        for scaler_name in cfg.train.LANCHESTER.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_lanchester_All(list_tier, list_time_len, list_lan_type, list_col, col_name, scaler_name, classifiers, rand_size)
                            
    if cfg.run_type.compare_result:
        print("Comparing results...")
        
        
        
    
if __name__ == "__main__":
    main()