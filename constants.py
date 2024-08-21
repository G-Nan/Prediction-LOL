list_tier = ['CHALLENGER', 'GRANDMASTER', 'MASTER', 'DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD' ,'SILVER', 'BRONZE', 'IRON']
list_division = ['I', 'II', 'III', 'IV']
list_time_len = [5, 10, 15, 20, 25, 30]
list_col = ['totalDamageDone', 'totalDamageTaken', 'totalGold', 'xp']
list_time_col = [(time_len, col) for time_len in list_time_len for col in list_col]
list_scaling = ['Standard', 'Ratio']

dic_tier = {tier : list_division if tier not in ['CHALLENGER', 'GRANDMASTER', 'MASTER'] else ['ALL'] for tier in list_tier }

drop_col = ['abilityHaste', 'armorPen', 'armorPenPercent', 'bonusArmorPenPercent', 'bonusMagicPenPercent', 'cooldownReduction', 
            'healthRegen', 'magicPenPercent', 'physicalVamp', 'powerMax', 'powerRegen', 'spellVamp', 'magicDamageDone', 
            'magicDamageDoneToChampions', 'magicDamageTaken', 'physicalDamageDone', 'physicalDamageDoneToChampions', 
            'physicalDamageTaken', 'totalDamageDoneToChampions', 'trueDamageDone', 'trueDamageDoneToChampions', 
            'trueDamageTaken', 'goldPerSecond', 'jungleMinionsKilled', 'level', 'minionsKilled', 'position', 'timeEnemySpentControlled']

use_col = ['abilityPower', 'armor', 'attackDamage', 'attackSpeed', 'ccReduction', 'health', 'healthMax', 'lifesteal', 'magicPen', 
           'magicResist', 'movementSpeed', 'omnivamp', 'totalDamageDone', 'totalDamageTaken', 'power', 'currentGold', 'totalGold',
           'spendGold', 'xp', 'Riftherald', 'Dragon', 'Baron_Nashor', 'Elder_Dragon', 'Ward', 'Tower', 'Tower_Plate', 'Inhibitor', 
           'win', 'participantId', 'match_id', 'timestamp']

dic_period = {
    0 : [0.2, 0.6, 0.2, 0.1, 0.3],
    1 : [0.5, 0.3, 0.3, 0.2, 0.5],
    2 : [0.3, 0.1, 0.5, 0.7, 0.2]    
}

list_dt = ['mean', 'weightedmean', 'point', 'timeseries', 'lanchester']
list_model = ['knn', 'LR', 'RF', 'SVC', 'XGB']
list_lan_type = ['Linear', 'Exponential', 'Mixed']
list_coef = ['Alpha', 'Beta', 'Gamma', 'Delta']

new_names = {
    'IRON' : 'IR',
    'BRONZE' : 'BR',
    'SILVER' : 'SV',
    'GOLD' : 'GD',
    'PLATINUM' : 'PT',
    'EMERALD' : 'EM',
    'DIAMOND' : 'DM',
    'MASTER' : 'MS',
    'GRANDMASTER' : 'GM',
    'CHALLENGER' : 'CH'
}
new_order = ['IR', 'BR', 'SV', 'GD', 'PT', 'EM', 'DM', 'MS', 'GM', 'CH', 'ALL']

list_fig_label = [r'$\alpha_{l}$', r'$\beta_{l}$', r'$\gamma_{l}$', 
                  r'$\alpha_{p}$', r'$\beta_{p}$', r'$\gamma_{p}$', 
                  r'$\alpha_{m}$', r'$\beta_{m}$', r'$\gamma_{m}$', r'$\delta_{m}$']    