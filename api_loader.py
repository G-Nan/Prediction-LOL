import os
import json
import time
import requests
import pandas as pd
from glob import glob
from tqdm import tqdm
from itertools import islice

def request_url(api_key, tier, page, save_path):
    
    tier_urls = {}
    divisions = ['I', 'II', 'III', 'IV']
    
    # summoner
    if tier in ['CHALLENGER','GRANDMASTER','MASTER']:
        tier_url = f'https://kr.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/I?page={page}&api_key={api_key}'
        path = f'{save_path}/{tier}' # save path
        tier_urls[path] = tier_url
    
    else:
        for division in divisions:
            tier_url = f'https://kr.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/{division}?page={page}&api_key={api_key}'
            path = f'{save_path}/{tier}/{division}'
            tier_urls[path] = tier_url
    
    return tier_urls

def request_puuid(api_key, tier_urls):
    
    for path, tier_url in tier_urls.items():
        r = requests.get(tier_url) # request summoners for tier
        
        # if status code is not 200
        if r.status_code != 200:
            print(r.status_code)
            break
        
        summoner_df = pd.DataFrame(r.json())
        
        pbar_summoner = tqdm(range(len(summoner_df)), position = 0)
        for i in pbar_summoner:
            pbar_summoner.set_description(f"{path.split('/')[-2]} / {path.split('/')[-1]}")
            
            try:
                summoner = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + summoner_df['summonerName'].iloc[i] + '?api_key=' + api_key 
                r = requests.get(summoner)
                
                while r.status_code == 429:
                    time.sleep(5)
                    summoner = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + summoner_df['summonerName'].iloc[i] + '?api_key=' + api_key 
                    r = requests.get(summoner)
                    
                puu_id = r.json()['puuid']
                summoner_df.loc[i, 'puuId'] = puu_id
                
                # save dataframe to csv
                summoner_df.to_csv(f'{path}/summoner.csv', index=False, encoding='cp949')
            
            except:
                pass
            
def Load_PUUID(api_key, tiers, page, save_path):
    tier_urls = {}
    for t in tiers:
        tier_urls.update(request_url(api_key, tier, page, save_path))
    request_puuid(api_key, tier_urls)
    
def ext_path(save_path, tiers, extension):
    
    ext_list = []
    for tier in tiers:
        for (path, dir, files) in os.walk(f'{save_path}/{tier}/'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == extension:
                    ext_list.append((path + '/' + filename).replace('\\', '/'))

    return ext_list
    
def request_matchid(api_key, path, start, count, set_tqdm):

    summoner_df = pd.read_csv(path, encoding = 'cp949')
    match_ids = []
    save_path = path.replace('summoner.csv', '') # save path for match list
    tier = ''.join(save_path.split('/')[3:]) # current tier
    cnt = 0
    dup = 0
    
    if set_tqdm:
        pbar_summoner = tqdm(range(len(summoner_df)), position = 1)
    else:
        pbar_summoner = range(len(summoner_df))
    
    for i in pbar_summoner:
    
        if set_tqdm:
            pbar_summoner.set_description(f'전체 : {len(match_ids)}, 에러 : {cnt}, 최종 : {len(set(match_ids))}')
        else:
            pass
            
        puuid = summoner_df['puuId'][i]
        try:
            match = f'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start={start}&count={count}&api_key={api_key}'
            r = requests.get(match)
            
            while r.status_code == 429:
                time.sleep(5)
                match = f'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start={start}&count={count}&api_key={api_key}'
                r = requests.get(match)
                
            match_ids.extend(requests.get(match).json())
        
        except:
            if r.status_code == 403:
                print('you need api renewal')
                break
        
        if match_ids[-1] == 'status':
            cnt += 1

    match_list = list(set(match_ids))
    if 'status' in match_list:
        match_list.remove('status')
    
    # save match id list
    with open(f'{save_path}match_list.txt', 'w') as f:
        for matchid in match_list:
            f.write(f'{matchid}\n')
                
def Load_MatchID(api_key, start, count, tiers, save_path):
        
    csv_path_list = ext_path(save_path, tiers, '.csv')

    pbar_csv_path = tqdm(csv_path_list, position = 0)
    for csv_path in pbar_csv_path:
        pbar_csv_path.set_description(f"{csv_path.split('/')[3]} / {csv_path.split('/')[4]}")
        
        request_matchid(api_key, csv_path, start, count, True)
            
    
    txt_path_list = ext_path(tiers, '.txt')

def request_match_json(api_key, path, set_tqdm):

    match_list = []
    with open(path, 'r') as file:
        for match_id in file:
            match_list.append(match_id[:-1])
            
    save_path = path.replace('match_list.txt', '') # save path for match json
    tier = save_path.split('/')[3] # current tier
    division = save_path.split('/')[4]
    
    # request match json for match list    
    match_json_dic = {} # match json
    
    if set_tqdm:
        pbar_match = tqdm(range(len(match_list)), position = 0)
    else:
        pbar_match = range(len(match_list))
        
    for i in pbar_match:
        match_id = match_list[i]
        match_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}'
        r = requests.get(match_url)
        
        if set_tqdm:
            pbar_match.set_description(f'{tier} / {division} - Status : {r.status_code}')
        else:
            pass

        if r.status_code == 200: # response 정상
            time.sleep(0.7)
        
        elif r.status_code == 403: # api 갱신 필요
            print('you need api renewal')
                
        elif r.status_code == 404: # data 호출 실패
            cnt = 1
            while True: 
                if r.status_code == 404:
                    cnt += 1
                    time.sleep(0.7)
                    
                    r = requests.get(match_url)
                    
                    if cnt == 3:
                        break
                    
                elif r.status_code == 200:
                    break
                    
        elif r.status_code != 200:
            
            while True: # 429 error가 해결될 때까지
                if r.status_code != 200:
                    time.sleep(0.7)
                    
                    r = requests.get(match_url)
                    
                elif r.status_code == 200:
                    break
            
        match_json = r.json()
        
        # if gamemode is classic and game duration is more than 15 min
        if match_json.get('info'):
            if match_json['info']['gameMode'] == 'CLASSIC' and \
                (match_json['info']['gameDuration'] // 60) > 15:
                    match_json_dic[match_id] = match_json  
                
    return match_json_dic, save_path

def Load_Match(api_key, start, count, tier, tiers, count_d, save_path):

    if tier == 'ALL':
        t = tiers
    else:
        t = [tier]

    txt_path_list = ext_path(save_path, t, '.txt')
    
    pbar_txt = tqdm(txt_path_list, position = 0)
    for path in pbar_txt:
        
        tier = path.split('/')[3]
        division = path.split('/')[4]
        pbar_txt.set_description(f'{tier} / {division}')
        
        match_json, save_path = request_match_json(api_key, path, False)
        pbar_txt.set_description(f'{tier} / {division} - {len(match_json)}')
        
        if len(match_json) >= count_d:
            print(len(match_json))
            with open(f'{save_path}match_json.json', 'w') as f:
                json.dump(dict(islice(match_json.items(), count_d)), f, ensure_ascii=False, indent=4)
        
        elif len(match_json) < count_d:
            
            while True:
                
                if len(match_json) < count_d:
                    start += count
                    count = 1
                    
                    csv_path = path.replace('match_list.txt', 'summoner.csv')
                    
                    request_matchid(api_key, csv_path, start, count, False)
                    add_json, _ = request_match_json(api_key, path, False)
                    match_json.update(add_json)
                    
                    pbar_txt.set_description(f'{tier} / {division} - {len(match_json)}')
                    
                elif len(match_json) >= count_d:
                    break
                
            # save json
            with open(f'{save_path}match_json.json', 'w') as f:
                json.dump(dict(islice(match_json.items(), count_d)), f, ensure_ascii=False, indent=4)

def request_timeline_json(api_key, path, match_list):
    
    save_path = path.replace('match_json.json','') # json file save path
    tier = save_path.split('/')[3] # current tier
    division = save_path.split('/')[4]

    # request match json for match list
    timeline_json_dic = {}
    
    pbar_match = tqdm(range(len(match_list)), position = 0)
    for i in pbar_match:
        match_id = match_list[i]
        timeline_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline?api_key={api_key}'
        r = requests.get(timeline_url)
        pbar_match.set_description(f'{tier} / {division} - Status : {r.status_code}')
        
        if r.status_code == 200: # response 정상
            time.sleep(0.7)
            
        elif r.status_code != 200:
            pbar_match.set_description(f'{tier} / {division} - Status : {r.status_code}')
            while True: # 429 error가 해결될 때까지
                if r.status_code != 200:
                    time.sleep(0.7)
                    
                    r = requests.get(timeline_url)
                    
                elif r.status_code == 200:
                    break

        # json 파일 생성         
        timeline_json_dic[match_id] = r.json()
        
    # save to json
    with open(f'{save_path}timeline_json.json', 'w') as f:
        json.dump(timeline_json_dic, f, ensure_ascii=False, indent=4)
        
def Load_Timeline(api_key, tier, tiers, save_path):
    if tier == 'ALL':
        t = tiers
    else:
        t = [tier]
        
    json_path_list = ext_path(save_path, t, '.json')
    
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            match_json = json.load(f)    
            
        matchid_list = list(match_json.keys())

        request_timeline_json(api_key, json_path, matchid_list)