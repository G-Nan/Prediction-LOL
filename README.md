# Prediction-LoL (2023-07-12 ~ )
Predicting League of Legends wins and losses using Lanchester's Law. 

# Table of Contents
0. [Running Code](#Running-Code)
1. [Description](#Description)
2. [Datasets](#Datasets)
3. [Preprocessing](#Preprocessing)
4. [EDA](#EDA)
5. [Classification Models](#Classification-Models)
6. [Conclusion](#Conclusion)
7. [About Us](#About-Us)

## Running Code
1. **Requirements**
2. **How to Use**

## Description
- We predicted the wins and losses of the matches, separated by tier. 
- We used five different data types to predict wins and losses. (Mean, Weighted Mean, Point, Timeseries, Lanchester)
- We used six different points in time. (5 min, 10 min, 15 min, 20 min, 25 min, 30 min)
- We used 8 classification models. (LR, SVC, kNN, RF, XGB, RNN, LSTM, CNN_LSTM) 

## Datasets
1. **Data Collection**
   - We collected data from ["Riot Developer Portal"](https://developer.riotgames.com/apis).
      > 1. Go to the ["Riot Developer Portal"](https://developer.riotgames.com/apis) and Get your API Key.
      > 2. Set division, tier to LEAGUE-exp-V4 address and collect summonerName of users in that league. (quene = RANKED_SOLO_5x5)
      > 3. Enter the summoner name you just collected in the by-name of the SUMMONER-V4 address to collect that user's puuid.
      > 4. Enter the puuid you just collected into MATCH-V5's by-puuid address to collect the match_ids of the user's recent matches.
      > 5. Enter the matchId you just collected in the MATCH-V5 address to collect the match data and match timeline data for that match.
   - You can do this through [API/api_request.py](https://github.com/G-Nan/Prediction-LOL/blob/main/API/api_request.ipynb).
2. **Data Description**
   - We collected data from 4000 matches in each tier.
   - We only collected games that ended after 15 minutes, because surrender is possible after 15 minutes, and we assumed that these were normal games.
   - The data we collected are matches played between September 24 and September 29, 2024.
3. **Feature Selection**
   - We used backward elimination to select variables, eliminating unnecessary variables.
   - Backward elimination
     > 1. We removed variables with all zero values, such as *abilityHaste*, *armorPen*, and *physicalVamp*.
     > 2. We also removed variables with more than 90% of values being zero, such as *armorPenPercent*, *magicPenPercent*, and *spellVamp*.
     > 3. We removed variables that were dependent on specific feature variables.
     >    - For **Vamp**, *omnivamp* was affected by *physicalVamp* and *spellVamp*, so we removed both and selected only *omnivamp*.
     >    - For **Damage**, there were four types: *totalDamage*, *physicalDamage*, *magicDamage*, and *trueDamage*, and three types: *Done*, *Taken*, and *ToChampion*, so we selected only *totalDamageDone* and *totalDamageTaken* and removed all others.
     >    - For *minionsKilled*, *jungleMinionsKilled*, and *goldPerSecond*, we removed those variables because they are completely dependent on *xp* and *totalGold*.

## Preprocessing
The data we collected is in the form of a three-dimensional tensor of the form *(5, 28, t)*, because there are *28 variables* for the *5 players* until *minute t*, when the game ends. However, for general classification, they take *vectors* as input, and for regression neural networks dealing with time series, they take matrices as input. **So we needed to convert the data, which is a tensor in three dimensions, into a vector or matrix.** Below are the five conversion methods we used to convert a three-dimensional tensor to a vector or matrix.

1. **Mean**
   - Compute the average of a team's match data over time and players and set it as a proxy for each variable.
   - This converts the three-dimensional tensor data in the form *(5, 28, t)* to a one-dimensional vector in the form *(1, 28, 1)* format, while keeping the number of feature variables the same.
   - This method ensures that all feature variables are equally weighted by player and time of day.
     
2. **Weighted Mean**
   - This conversion assumes that each player's role has different importance to the team's combat power at different times.
   - For example, Jungle has a relatively greater impact on game wins and losses in the early game than in the late game, and AD Carry has a greater impact in the late game than in the early game.
   - To account for this, we weight each line differently by time of day and calculate the representative value of the feature variable for each role as a weighted average over time.
   - This converts a three-dimensional tensor of the form *(5, 28, t)* into a two-dimensional matrix of the form *(5, 28, 1)*. This is then converted back to a one-dimensional vector of the form *(1, 28, 1)* by computing the average over the players.
   - We gave the most weight to jungle roles in the early game, top and support roles in the middle game, and ADC and MID roles in the late game.
   - This the weight table we used.
     > |Lanes|Early|Mid|Late|
     > |:---:|:---:|:---:|:---:|
     > |TOP|0.2|0.5|0.3|
     > |JGL|0.6|0.3|0.1|
     > |MID|0.2|0.3|0.5|
     > |ADC|0.1|0.2|0.7|
     > |SUP|0.3|0.5|0.2|
     
3. **Point**
   - This method computes a per-player average of the values of the variables at a given point in time as a proxy for team combat power.
   - Selecting values at time t=t0 from a three-dimensional tensor of the form (5, 28, t) transforms it into a two-dimensional matrix of the form (5, 28, 1), and each variable is transformed into a one-dimensional matrix of the form (1, 28, 1) by calculating its average across players.
   - This method assumes that all variables have the same weight per line.
   - The advantage of this method is that it predicts wins and losses based only on the values of the variables at t=t0, so it can predict the win or loss of a game even while the game is in progress.
     > t0 = 5, 10, 15, 20, 25, 30

4. **Timeseries**
   - Here, we used two different preprocessing methods because there is a difference between the two inputs of general classification model and regression neural network
   - We used four variables that are generally considered to be the most important in winning or losing a game.
     > Four variables : totalDamageDone, totalDamageTaken, xp, totalGold.
   - The four variables for each of the five players per team exist as time series data up to time t0.
     > t0 = 5, 10, 15, 20, 25, 30
   - One of the methods is a transformation for general classification models.
      > -  For a general classification model, the input is a one-dimensional vector, so we need to make these four variables into one-dimensional vectors.
      > -  First, we find the average of the players for each variable and set it as a representative value for that variable. This transforms it from a three-dimensional tensor of the form (5, 4, t0) to a two-dimensional matrix of the form (1, 4, t0).
      > -  Then flatten the two-dimensional matrix into a one-dimensional vector. This converts the two-dimensional matrix of (1, 4, t0) into a one-dimensional vector of (1, 4*t0).
      > -  For example, if you have **n** columns to use a general classification model, you would transform the data in the following ways
      > - Origin Dataset
      > > |timestamp|col_1|col_2|col_3|..|col_n|
      > > |:---:|:---:|:---:|:---:|:---:|:---:|
      > > |1|..|..|..|..|..|
      > > |2|..|..|..|..|..|
      > > |:|..|..|..|..|..|
      > > |t0|..|..|..|..|..|
      > >
      > - Transformed Dataset
      > > |col_1_1|col_1_2|...|col_1_t0|col_2_1|col_2_2|...|col_2_t0|...|col_n_1|col_n_2|...|col_n_t0|
      > > |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
      > > |..|..|..|..|..|..|..|..|..|..|..|..|..|
   - The other method is for Recurrent neural networks
      > - Regression neural networks can take two-dimensional matrices as input data, so you can use the Origin Dataset above without any preprocessing.

5. **Lanchester**
   - Lanchester's laws are mathematical formulae for calculating the relative strengths of military forces.
   - The Lanchester equations are differential equations describing the time dependence of two armies' strengths A and B as a function of time, with the function depending only on A and B.
   - First, it was determined that certain variables were representative of a team's combat power.
   - Then, the values of these variables were used to estimate the coefficients of the Lanchester model, which predicted wins and losses.
   - As a representation of combat power, we used four variables that are generally considered to be the most important to winning or losing a game.
     > Four variables : totalDamageDone, totalDamageTaken, xp, totalGold.
   - We used three differential equations: linear, exponential, and mixed.
   - Three differential equations
      - **Linear**
         $$R' = \alpha + \beta R + \gamma B$$
      - **Exponential**
         $$R' = \alpha {\beta}^{R} {\gamma}^{B}$$
      - **Mixed**
         $$R' = \alpha + \beta R + \gamma B + \delta R B$$
   - We used the **least squares** method to estimate the coefficients of the three differential equations above. If you want to know more details, we recommend reading the paper.

6. **Scailing Method**
   - Since the values in the feature data have different units, scaling is necessary in the data conversion process. However, we expect that traditional scaling methods such as *Standard Scaling* or *MinMax Scaling* are not suitable because **the end time of each game is different**.
   - This table shows an example where the losing team's gold is higher than the winning team's gold due to the difference in the end of the game.
     > ||Timestamp|Variables|Winning Team|Losing Team|
     > |:---:|:---:|:---:|:---:|:---:|
     > |Game_1|15m|totalGold|**6793**|4832|
     > |Game_2|30m|totalGold|13244|**9871**|
   - Since all variables are affected by **match duration**, and the impact of each variable on combat power depends on the **opposing team**, we needed to account for the share of feature variables.
   - To address these issues, we used the following scaling methods.
      > $$R_{scaled} = \frac{R}{R+B}, \quad B_{scaled} = \frac{B}{R+B} \quad (if \quad R = B = 0, \quad R_{scaled} = B_{scaled} = 0.5)$$

## EDA
   - In this section, we will only show a simple visualization, because we have already applied several preprocessing methods and have a lot of data, it may contain information you don't want.
   - Therefore, we will compare three methods: **Mean**, **Ratio**, and **Lanchester**. We want to compare the values of the four important variables mentioned above for each data point based on the winning and losing teams. 
   - We can't show all 10 tiers, so we've only selected three.
      > The tiers are **IRON**, the lowest, **PLATINUM**, the middle, and **CHALLENGER**, the highest.
   - We chose two viewpoints to visualize.
      > Time points are **15 minutes**, when surrender is possible, and **30 minutes**, representing the long game.
1. Mean
   - It's hard to draw a clear line between the winning and losing teams.
   - At 15 minutes, most of the variables seem to follow a normal distribution, but at 30 minutes, they appear to be non-normal with a longer left tail.
   > ![Mean](https://github.com/G-Nan/Prediction-LOL/blob/main/Images/Important%20variable%20values%20with%20Mean%20for%20winloss%20teams.png)
   
2. Ratio
   - The difference between winning and losing teams is more visible. 
   - Both 15 and 30 minutes appear to follow a normal distribution.
   > ![Mean Ratio](https://github.com/G-Nan/Prediction-LOL/blob/main/Images/Important%20variable%20values%20with%20Mean%20Ratio%20for%20winloss%20teams.png)
   
3. Lanchester Linear Model
   - Alpha is a constant term and doesn't make much difference between winning and losing teams.
   - Beta and Gamma are clearly distinguishable. Beta shows a smaller value for the winning team, while Gamma shows a larger value for the winning team.
   > ![Lanchester Linear](https://github.com/G-Nan/Prediction-LOL/blob/main/Images/Lanchester%20Coefficients%20with%20Linear%20Model%20by%20winloss.png)

4. Lanchester Exponential Model
   - For alpha, it doesn't make much difference.
   - For beta and gamma, it's similar to a linear model.
   > ![Lanchester Exponential](https://github.com/G-Nan/Prediction-LOL/blob/main/Images/Lanchester%20Coefficients%20with%20Exponential%20Model%20by%20winloss.png)
     
5. Lanchester Mixed Model
   - For Alpha and Delta, there is almost no difference.
   - Beta and Gamma have the clearest differences between the above methods.
   >![Lanchester Mixed](https://github.com/G-Nan/Prediction-LOL/blob/main/Images/Lanchester%20Coefficients%20with%20Mixed%20Model%20by%20winloss.png)
     
## Classification Models
   - We used 5 general classification models and 3 neural network models.
      - 5 General Classification Models : Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost
      - 3 Neural Network Models : RNN, LSTM, CNN+LSTM
   - Hyperparameter
      - We utilized GridSearch to find the optimal parameters.
      - Candidate parameters set for each model are
         - Logistic Regression
           > - C : [0.01, 0.1, 1, 10, 100]
         - Support Vector Machine
           > - C : [0.01, 1, 100]
           > - gamma : ['scale', 0.1, 1, 10]
           > - degree : [2, 3, 4]
         - K-Nearest Neighbors
           > - n-neighbors : [3, 5, 7, 9, 11]
         - Random Forest
           > - n_estimators : [100, 200]
           > - max_depth : [None, 2, 4]
           > - min_samples_split : [2, 4]
           > - min_samples_leaf : [1, 2]
         - XGBoost
           > - n_estimators : [100, 150]
           > - learning_rate : [0.2]
           > - max_depth : [2, 4, 6]
           > - subsamples : [0.9]
           > - colsample_bytree : [0.9]
           > - reg_alpha : [0.5, 1]
         - RNN, LSTM, CNN+LSTM
           > - batch size : [256]
           > - num epochs : [1000]
           > - learning rate : [1e-3, 1e-4]
           > - hidden_size : [16, 32]
           > - num_layers : [2, 4]
           > - patience : [10]
           > - dropout : [0, 0.25]
           > - Loss Function : BCELoss()
  - We trained all the candidates for each data and selected the one with the lowest validation loss.

## Conclusion
  - There are 7986 trained models.
    - Mean : 660
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Scaling method : 2 (Standard, Ratio) <br>
      > Models : 5 (Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost) <br>
    - Weighted Mean : 660
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Scaling method : 2 (Standard, Ratio) <br>
      > Models : 5 (Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost) <br>
    - Point : 660
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Scaling method : 2 (Standard, Ratio) <br>
      > Models : 5 (Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost) <br>
    - Timeseries Flat : 660
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Scaling method : 2 (Standard, Ratio) <br>
      > Models : 5 (Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost) <br>
    - Timeseries : 396
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Scaling method : 2 (Standard, Ratio) <br>
      > Models : 3 (RNN, LSTM, CNN+LSTM) <br>
    - Lanchester : 4950
      > Tier : 11 (IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER, ALL) <br>
      > Time Point : 6 (5, 10, 15, 20, 25, 30) <br>
      > Models : 5 (Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, XGBoost) <br>
      > Features : 5 (totalDamageDone, totalDamageTaken, xp, totalGold, ALL) <br>
      > Equations : 3 (Linear, Exponential, Mixed) <br>


## About Us
|<img src="https://github.com/chdaewon.png" width="80">||<img src="https://github.com/G-Nan.png" width="80">|<img src="https://github.com/ddanggu.png" width="80">|
|:---:|:---:|:---:|:---:|
|[chdaewon](https://github.com/chdaewon)|Prof. Jeong|[G-Nan](https://github.com/G-Nan)|[ddanggu](https://github.com/ddanggu)|
|Mathematical Methodology <br> Development|Literature Review|Model Engineering|Data Collection / Visualization|



















