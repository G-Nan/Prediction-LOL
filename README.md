# Prediction-LoL (2023-07-12 ~ )
Predicting League of Legends wins and losses using Lanchester's Law. 

# Table of Contents
0. [Setup and Execution](#Setup-and-Execution)
1. [Datasets](#Datasets)
4. [EDA](#EDA)
5. [Classification Models](#Classification-Models)
6. [Conclusion](#Conclusion)
7. [About Us](#About-Us)

# Setup and Execution

To install the required packages, please ensure you have Python and pip installed. Then, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the necessary Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

This command will install all the required dependencies listed in the `requirements.txt` file.

3. **(Optional) If you are using a virtual environment, activate it before running the `pip install` command**:

    ```bash
    conda create -n Lanchester python=3.10.9
    conda activate Lanchester
    pip install -r requirements.txt
    ```

4. **Running the Simulation**

To run the simulation, you can use the `run_simulation.py` script. Before running the script, ensure that you have configured the `.hydra/config.yaml` file according to your needs.

#### Configuring `config.yaml`

The `config.yaml` file contains various parameters that control how the simulation is run. You can customize the following sections to suit your requirements:

- **run_type**: Choose the actions to perform during the simulation, such as loading data from the API, creating datasets, training models, and comparing results.
- **load_api**: Configure API-related settings, including the API key, data types, tiers, and match limits.
- **make_dataset**: Specify how datasets should be generated, including the tiers, time lengths, and data types to use.
- **load_dataset**: Select the types of datasets to load for training and evaluation.
- **train**: Set the parameters for training models, including the data types, columns, scalers, and training configurations.
- **param_gridsearch**: Configure hyperparameters for machine learning and neural network models, including grid search settings.
- **compare_result**: Set this to `true` if you want to compare the results after training.

If you do not want to load data from the API or perform preprocessing, and only want to use preprocessed data for training and result comparison, you can set the `config.yaml` file as follows:

```yaml
run_type:
  load_api: false
  make_dataset: false
  load_dataset: true
  train: true
  compare_result: true
```

# Data collection
   - We collected data from ["Riot Developer Portal"](https://developer.riotgames.com/apis).
      > 1. Go to the ["Riot Developer Portal"](https://developer.riotgames.com/apis) and Get your API Key.
      > 2. Set division, tier to LEAGUE-exp-V4 address and collect summonerName of users in that league. (quene = RANKED_SOLO_5x5)
      > 3. Enter the summoner name you just collected in the by-name of the SUMMONER-V4 address to collect that user's puuid.
      > 4. Enter the puuid you just collected into MATCH-V5's by-puuid address to collect the match_ids of the user's recent matches.
      > 5. Enter the matchId you just collected in the MATCH-V5 address to collect the match data and match timeline data for that match.

# Preprocessing
For the preprocessing section, I will only describe the preprocessing methods used for the core aspects of this research: **the Lanchester and ratio scaling methods**, as well as the preprocessing methods used to **calculate variables** not covered in the paper. *For all other preprocessing methods, please refer to the paper.*

1. **Variables**
   - Ward
      > - In League of Legends, there are four types of wards: **yellow trinket**, **sight ward**, **blue trinket**, and **control ward**.
      > - In this study, the number of wards present on the map at each time interval for each team was used as a variable.
      > - While the exact time when a ward is placed is recorded, the time when a ward disappears is not (*only the time when a ward is destroyed is recorded, while the natural expiration of a ward is not logged*).
      > - For wards that disappear over time, such as the **yellow trinket** and **sight ward**, only cases where the wards expire naturally were considered, excluding those that were destroyed (*to avoid double-counting wards that are both destroyed and expired*).
      > - For **blue trinkets** and **control wards**, which remain on the map indefinitely unless destroyed, the count was adjusted by +1 when placed and -1 when destroyed.

   - Baron_Nashor, Elder_Dragon
      > - For Baron Nashor and Elder Dragon, the temporary buffs gained after killing these monsters play a much more critical role in the game than the mere number of times they are slain. Therefore, in this study, instead of using the number of kills, the presence of the buff at a given time was used as a variable. The variable is set to 1 if the buff is present and 0 if it is not.
      > - An important detail to note is that if a character is dead at the moment the buff is obtained, they do not receive the buff, and even if they have the buff, it is lost upon death. Therefore, it was necessary to calculate whether each character was alive at the moment of buff acquisition and also to check if they remained alive during the buff duration.
      > - The calculation of death time was referenced from the following link: [Death - League of Legends Wiki](https://leagueoflegends.fandom.com/wiki/Death). The calculation formula is as follows:
      > > - BRW
      > > > ![image](https://github.com/user-attachments/assets/bd685b60-b284-43f0-9a5c-fcb1a6891b42)
      > > - TIF
      > > > ![image](https://github.com/user-attachments/assets/eb2244e7-ccaf-4458-8ba3-afd768876383)
      > > - Total death time = BRW + BRW × TIFx | where x = current minute-half.
      > 

2. **Lanchester**
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

3. **Scailing Method**
   - Since the values in the feature data have different units, scaling is necessary in the data conversion process. However, we expect that traditional scaling methods such as *Standard Scaling* or *MinMax Scaling* are not suitable because **the end time of each game is different**.
   - This table shows an example where the losing team's gold is higher than the winning team's gold due to the difference in the end of the game.
     > ||Timestamp|Variables|Winning Team|Losing Team|
     > |:---:|:---:|:---:|:---:|:---:|
     > |Game_1|15m|totalGold|**6793**|4832|
     > |Game_2|30m|totalGold|13244|**9871**|
   - Since all variables are affected by **match duration**, and the impact of each variable on combat power depends on the **opposing team**, we needed to account for the share of feature variables.
   - To address these issues, we used the following scaling methods.
      > $$R_{scaled} = \frac{R}{R+B}, \quad B_{scaled} = \frac{B}{R+B} \quad (if \quad R = B = 0, \quad R_{scaled} = B_{scaled} = 0.5)$$

# EDA
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
     
# Classification Models
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

# Summary of Model Training Configurations
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

# Conclusion


# About Us

<table>
  <tr>
    <td align="center"><img src="https://github.com/chdaewon.png" width="80"></td>
    <td></td>
    <td align="center"><img src="https://github.com/G-Nan.png" width="80"></td>
    <td align="center"><img src="https://github.com/ddanggu.png" width="80"></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/chdaewon">chdaewon</a></td>
    <td align="center">Prof. Jeong</td>
    <td align="center"><a href="https://github.com/G-Nan">G-Nan</a></td>
    <td align="center"><a href="https://github.com/ddanggu">ddanggu</a></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Mathematical Methodology Development <br> Literature Review</td>
    <td align="center">Model Engineering</td>
    <td align="center">Data Collection / Visualization</td>
  </tr>
</table>

















