import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from api_loader import *
from data_maker import *
from data_loader import *
from models import *
from trainer import *
from utils import *
from result_comparision import *
from constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path = ".hydra", config_name = "config.yaml", version_base = None)
def main(cfg: DictConfig):
    if cfg.run_type.load_api:
        print("Loading API...")
        
        # API KEY
        api_key_ = cfg.load_api.API_KEY
        
        # Save path
        save_path_ = cfg.load_api.save_path
                
        # Data type
        data_type_ = cfg.load_api.data_type
        allowed_values = ['PUUID', 'MATCH_ID', 'MATCH', 'TIMELINE', 'ALL']
                
        if data_type_ not in allowed_values:
            raise ValueError(f"Invalid value '{data_type_}' for 'data_type'. Choose one of: {', '.join(allowed_values)}")
            
        # Tier
        list_tier_ = cfg.load_api.list_tier
            
        # Start, Count
        start_matchid_ = cfg.load_api.start_matchid
        count_matchid_ = cfg.load_api.count_matchid
        start_matchlist_ = cfg.load_api.start_matchlist
        count_matchlist_ = cfg.load_api.count_matchlist
        count_dataset_ = cfg.load_api.count_dataset
        
        page = 1
       
        # PUUID
        if data_type_ == 'PUUID':
            Load_PUUID(api_key_, list_tier_, page, save_path_)

        # Match ID
        if data_type_ == 'MATCH_ID':
            Load_MatchID(api_key_, start_matchid_, count_matchid_, 'ALL', list_tier_, save_path_)

        # Match
        if data_type_ == 'MATCH':
            Load_Match(api_key_, list_tier_, tiers_, start_match_, count_match_, count_dataset_, save_path_)
        
        # Timeline
        if data_type_ == 'TIMELINE':
            Load_Timeline(api_key_, list_tier_, tiers_, save_path_)
            
        # All
        if data_type_ == 'ALL':
            Load_PUUID(api_key_, list_tier_, tiers_, page, save_path_)
            Load_MatchID(api_key_, list_tier_, tiers_, start_matchid_, count_matchid_, save_path_)
            Load_Match(api_key_, list_tier_, tiers_, start_match_, count_match_, cound_dataset_, save_path_)
            Load_Timeline(api_key_, list_tier_, tiers_, save_path_)
            
    if cfg.run_type.make_dataset:
        print("Making dataset...")
        
        # Save path
        save_path_ = cfg.make_dataset.save_path
        
        # Data type
        data_type_ = cfg.make_dataset.data_type
        type_allowed_values = ['ALL', 'MEAN', 'WEIGHTEDMEAN', 'POINT', 'TIMESERIES', 'LANCHESTER', 'TOTAL']
        
        # Tier
        list_tier_ = cfg.make_dataset.list_tier
        list_time_len_ = cfg.make_dataset.list_time_len
        dic_tier_selected = {t:dic_tier[t] for t in list_tier}
        
        for dt in data_type_:
            if dt not in type_allowed_values:
                raise ValueError(f"Invalid value '{data_type_}' for 'data_type'. Choose one of: {', '.join(type_allowed_values)}")
        
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
        
        data_type_ = cfg.load_dataset.data_type
        data_download(data_type_)
        
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
                    
                list_col_ = cfg.train.MEAN.list_col
                list_tier_ = cfg.train.MEAN.list_tier
                list_time_len_ = cfg.train.MEAN.list_time_len
                rand_size_ = cfg.train.MEAN.rand_size
                
                if cfg.train.MEAN.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.MEAN.col_name:
                        for scaler_name in cfg.train.MEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_mean(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers)
                            
                if cfg.train.MEAN.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.MEAN.col_name:
                        for scaler_name in cfg.train.MEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_mean_All(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers, rand_size_)
                            
            if dt =='WEIGHTEDMEAN':
            
                if not os.path.exists('scores/weightedmean/'):
                    os.makedirs('scores/weightedmean/')
                    
                if not os.path.exists('models/weightedmean/'):
                    os.makedirs('models/weightedmean/')
                    
                list_col_ = cfg.train.WEIGHTEDMEAN.list_col
                list_tier_ = cfg.train.WEIGHTEDMEAN.list_tier
                list_time_len_ = cfg.train.WEIGHTEDMEAN.list_time_len
                rand_size_ = cfg.train.WEIGHTEDMEAN.rand_size
                
                if cfg.train.WEIGHTEDMEAN.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.WEIGHTEDMEAN.col_name:
                        for scaler_name in cfg.train.WEIGHTEDMEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_weightedmean(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers)
                            
                if cfg.train.WEIGHTEDMEAN.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.WEIGHTEDMEAN.col_name:
                        for scaler_name in cfg.train.WEIGHTEDMEAN.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_weightedmean_All(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers, rand_size_)
                            
            if dt == 'POINT':
            
                if not os.path.exists('scores/point/'):
                    os.makedirs('scores/point/')
                    
                if not os.path.exists('models/point/'):
                    os.makedirs('models/point/')
                    
                list_col_ = cfg.train.POINT.list_col
                list_tier_ = cfg.train.POINT.list_tier
                list_time_len_ = cfg.train.POINT.list_time_len
                rand_size_ = cfg.train.POINT.rand_size
                
                if cfg.train.POINT.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.POINT.col_name:
                        for scaler_name in cfg.train.POINT.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_point(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers)
                            
                if cfg.train.POINT.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.POINT.col_name:
                        for scaler_name in cfg.train.POINT.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_point_All(list_tier_, list_time_len_, list_col_, col_name, scaler_name, classifiers, rand_size_)
                            
            if dt == 'TIMESERIES':
            
                if not os.path.exists('scores/timeseries/'):
                    os.makedirs('scores/timeseries/')
                    
                if not os.path.exists('models/timeseries/'):
                    os.makedirs('models/timeseries/')
                
                if not os.path.exists('hyperparameters/timeseries/'):
                    os.makedirs('hyperparameters/timeseries/')
                    
                list_col_ = cfg.train.TIMESERIES.list_col
                list_tier_ = cfg.train.TIMESERIES.list_tier
                list_time_len_ = cfg.train.TIMESERIES.list_time_len
                rand_size_ = cfg.train.TIMESERIES.rand_size
                
                if cfg.train.TIMESERIES.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.TIMESERIES.col_name:
                        for scaler_name in cfg.train.TIMESERIES.scaler_type:
                            for model_name in cfg.train.TIMESERIES.model_name:
                                print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}, Model : {model_name}")
                                Training_timeseries(list_tier_, list_time_len_, list_col_, col_name, scaler_name, list_params, model_name, device)
                            
                if cfg.train.TIMESERIES.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.TIMESERIES.col_name:
                        for scaler_name in cfg.train.TIMESERIES.scaler_type:
                            for model_name in cfg.train.TIMESERIES.model_name:
                                print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}, Model : {model_name}")
                                Training_timeseries_All(list_tier_, list_time_len_, list_col_, col_name, scaler_name, list_params, model_name, device, rand_size_)
                            
            if dt == 'LANCHESTER':
            
                if not os.path.exists('scores/lanchester/'):
                    os.makedirs('scores/lanchester/')
                    
                if not os.path.exists('models/lanchester/'):
                    os.makedirs('models/lanchester/')
                
                list_col_ = cfg.train.LANCHESTER.list_col
                list_tier_ = cfg.train.LANCHESTER.list_tier
                list_time_len_ = cfg.train.LANCHESTER.list_time_len                
                list_lan_type_ = cfg.train.LANCHESTER.list_lan_type
                rand_size_ = cfg.train.LANCHESTER.rand_size
                
                if cfg.train.LANCHESTER.ALL_tier in ['1tier', 'Both']:
                    for col_name in cfg.train.LANCHESTER.col_name:
                        for scaler_name in cfg.train.LANCHESTER.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_lanchester(list_tier_, list_time_len_, list_lan_type_, list_col_, col_name, scaler_name, classifiers)
                            
                if cfg.train.LANCHESTER.ALL_tier in ['All', 'Both']:
                    for col_name in cfg.train.LANCHESTER.col_name:
                        for scaler_name in cfg.train.LANCHESTER.scaler_type:
                            print(f"Data type : {dt}, Col : {col_name}, Scaler : {scaler_name}")
                            Training_lanchester_All(list_tier_, list_time_len_, list_lan_type_, list_col_, col_name, scaler_name, classifiers, rand_size_)
                            
    if cfg.run_type.compare_result:
        print("Comparing results...")
        
        if not os.path.exists('results'):
            os.makedirs('results')
                    
        if cfg.compare_result.result_score:
            print("Making score table...")
            result_score(list_dt, list_tier, list_time_len, list_col)
            
        if cfg.compare_result.result_table2:
            print("Making table2...")
            try:
                dfs = pd.read_csv('results/score.csv')
            except FileNotFoundError:
                print("Error: 'results/score.csv' 파일을 찾을 수 없습니다. 'result_score'를 먼저 실행하세요.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            else:
                result_table2(dfs, 'PLATINUM')
            
        if cfg.compare_result.result_table3:
            print("Making table3...")
            try:
                dfs = pd.read_csv('results/score.csv')
            except FileNotFoundError:
                print("Error: 'results/score.csv' 파일을 찾을 수 없습니다. 'result_score'를 먼저 실행하세요.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            else:
                result_table3(dfs, new_names, new_order)
                
        if cfg.compare_result.result_figure123:
            print("Plotting figure1, figure2, figure3...")
            if not os.path.exists('results/fig1'):
                os.makedirs('results/fig1')
            if not os.path.exists('results/fig2'):
                os.makedirs('results/fig2')
            if not os.path.exists('results/fig3'):
                os.makedirs('results/fig3')
            result_figure123('CHALLENGER', list_coef, 'totalDamageDone', [15, 30], list_lan_type, list_fig_label)
            
        if cfg.compare_result.result_fig4:
            print("Plotting figure4...")
            if not os.path.exists('results/fig4'):
                os.makedirs('results/fig4')
            result_fig4('CHALLENGER', 'totalDamageDone', [15, 30], list_scaling)
            
        if cfg.compare_result.result_fig5:
            print("Plotting figure5...")
            if not os.path.exists('results/fig5'):
                os.makedirs('results/fig5')
            try:
                dfs = pd.read_csv('results/score.csv')
            except FileNotFoundError:
                print("Error: 'results/score.csv' 파일을 찾을 수 없습니다. 'result_score'를 먼저 실행하세요.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            else:
                result_fig5(dfs, colors, new_order)
                
        if cfg.compare_result.result_fig6:
            print("Plotting figure6...")
            if not os.path.exists('results/fig6'):
                os.makedirs('results/fig6')
            try:
                dfs = pd.read_csv('results/score.csv')
            except FileNotFoundError:
                print("Error: 'results/score.csv' 파일을 찾을 수 없습니다. 'result_score'를 먼저 실행하세요.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            else:
                result_fig6(dfs, colors, new_order)  

        if cfg.compare_result.result_fig7:
            print("Plotting figure7...")
            if not os.path.exists('results/fig7'):
                os.makedirs('results/fig7')
            try:
                dfs = pd.read_csv('results/score.csv')
            except FileNotFoundError:
                print("Error: 'results/score.csv' 파일을 찾을 수 없습니다. 'result_score'를 먼저 실행하세요.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            else:
                result_fig7(dfs, list_col, colors, new_order)  
                
        if cfg.compare_result.result_predict_time:
            print("Calculating prediction time...")
            result_predict_time(list_tier, list_time_len, list_col, list_lan_type)
            
        if cfg.compare_result.result_count:
            print("Counting data entries...")
            result_count(list_tier, list_time_len, new_order, new_names)        
    
        if cfg.compare_result.result_ivp:
            print("Solving initial value problem...")
            result_ivp(list_tier, list_time_len, list_col)
            
if __name__ == "__main__":
    main()