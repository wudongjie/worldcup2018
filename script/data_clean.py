import pandas as pd
import json
import numpy as np
from sklearn import preprocessing


countryList = []
with open("../data/countries.json", "r") as jsonfile:
    data = json.load(jsonfile)
    for key, val in data.items():
        countryList.append({'countryName': val['name'],
                            'continent': val['continent']})

continent_df = pd.DataFrame(countryList)
continent_df = continent_df.replace({"United Kingdom": "England",
                                     "China": "China PR"})
continent_df2 = pd.DataFrame({"countryName": ["Scotland", "Northern Ireland", "Wales"],
                              "continent": ["EU", "EU", "EU"]})
continent_df = continent_df.append(continent_df2)


rankings = pd.read_csv('../data/fifa_ranking.csv')
rankings = rankings.loc[:, ['rank', 'country_full',
                            'cur_year_avg_weighted', 'rank_date',
                            'two_year_ago_weighted',
                            'three_year_ago_weighted']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['weighted_points'] = rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

matches = pd.read_csv('../data/results.csv')
matches = matches.replace({'Germany DR': 'Germany',
                           'China': 'China PR',
                           'Czechoslovakia': 'Czech Republic'})
matches['date'] = pd.to_datetime(matches['date'])
#matches = matches.head(1000)  # Use the 1000 samples as a test for the model
print("matches sampling done")



# Fill in the rank for every day
rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()

# join the ranks
matches = matches.merge(rankings,
                        left_on=['date', 'home_team'],
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings,
                        left_on=['date', 'away_team'],
                        right_on=['rank_date', 'country_full'],
                        suffixes=('_home', '_away'))
print("Join Done!")
matches.drop(["country_full_away", "country_full_home",
                        "rank_date_home", "rank_date_away"], axis=1, inplace=True)

# separate the home team and the opponent team
matches_op = matches.copy()
matches_op = matches_op.rename(index=str, columns={'away_team': 'target_team',
                                        'home_team': 'opponent_team',
                                        'away_score': 'target_score',
                                        'home_score': 'opponent_score',
                                        'rank_away': 'target_rank',
                                        'rank_home': 'opponent_rank',
                                        'weighted_points_away': 'target_weighted_points',
                                        'weighted_points_home': 'opponent_weighted_points',
                                        'cur_year_avg_weighted_away': 'target_cur_avg_points',
                                        'cur_year_avg_weighted_home': 'opponent_cur_avg_points',
                                        'two_year_ago_weighted_away': 'target_2y_avg_points',
                                        'two_year_ago_weighted_home': 'opponent_2y_avg_points',
                                        'three_year_ago_weighted_away': 'target_3y_avg_points',
                                        'three_year_ago_weighted_home': 'opponent_3y_avg_points'})
matches_host = matches.rename(index=str, columns={'home_team': 'target_team',
                                       'away_team': 'opponent_team',
                                       'home_score': 'target_score',
                                       'away_score': 'opponent_score',
                                       'rank_home': 'target_rank',
                                       'rank_away': 'opponent_rank',
                                       'weighted_points_home': 'target_weighted_points',
                                       'weighted_points_away': 'opponent_weighted_points',
                                        'cur_year_avg_weighted_home': 'target_cur_avg_points',
                                        'cur_year_avg_weighted_away': 'opponent_cur_avg_points',
                                        'two_year_ago_weighted_home': 'target_2y_avg_points',
                                        'two_year_ago_weighted_away': 'opponent_2y_avg_points',
                                        'three_year_ago_weighted_home': 'target_3y_avg_points',
                                        'three_year_ago_weighted_away': 'opponent_3y_avg_points'})

matches_host['is_home'] = 1
matches_op['is_home'] = 0
df = pd.concat([matches_op, matches_host],ignore_index=True)
# feature generation
df['yeardiff'] = 2018 - df['date'].dt.year
df['month'] = df['date'].dt.month
df['rank_difference'] = df['target_rank'] - df['opponent_rank']
df['average_rank'] = (df['target_rank'] + df['opponent_rank']) / 2
df['point_difference'] = df['target_weighted_points'] - df['opponent_weighted_points']
df['is_world_cup'] = df['tournament'].str.contains("World Cup").astype(int)
df['is_stake'] = df['tournament'] != 'Friendly'
print("feature Generated!")

#Team Specific Data
df['tscore_lastgame'] = df.groupby('target_team')['target_score'].shift(1)
df['opteam_lastgame'] = df.groupby('target_team')['opponent_team'].shift(1)
df['opscore_lastgame'] = df.groupby('target_team')['opponent_score'].shift(1)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=3, min_periods=1).mean())
df['tscore_3ma'] = new_col.reset_index(level=0, drop=True)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=5, min_periods=1).mean())
df['tscore_5ma'] = new_col.reset_index(level=0, drop=True)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=10, min_periods=1).mean())
df['tscore_10ma'] = new_col.reset_index(level=0,drop=True)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=15, min_periods=1).mean())
df['tscore_15ma'] = new_col.reset_index(level=0, drop=True)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=30, min_periods=1).mean())
df['tscore_30ma'] = new_col.reset_index(level=0, drop=True)
print(df[['tscore_3ma','tscore_10ma', 'tscore_30ma', 'target_score']][df['target_team'] == 'Brazil'])

## Merge continent to df
df = df.merge(continent_df,
              left_on=['target_team'],
              right_on=['countryName'])
df = df.merge(continent_df,
                        left_on=['opponent_team'],
                        right_on=['countryName'],
                        suffixes=('_target', '_opponent'))
df.to_csv('../data/cleaned.csv')
print("data cleaned!")

# Clean the world cup 2018 dataset
world_cup = pd.read_csv('../data/World Cup 2018 Dataset.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst',
                              'Second match\n against',
                              'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran",
                               "Costarica": "Costa Rica",
                               "Porugal": "Portugal",
                               "Columbia": "Colombia",
                               "Korea": "Korea Republic"})

first_match = world_cup[['Team', 'First match \nagainst']]
second_match = world_cup[['Team', 'Second match\n against']]
third_match = world_cup[['Team', 'Third match\n against']]

colnames = ['target_team', 'opponent_team']
first_match.columns = colnames
second_match.columns = colnames
third_match.columns = colnames
dfc = pd.concat([first_match, second_match, third_match], ignore_index=True)

# Extract the latest match for the home team
grouped = df.groupby('target_team').agg({'date':'max'})
latest = grouped.merge(df, how='left',
                       left_on=['target_team', 'date'],
                       right_on=['target_team', 'date'])
latest = latest[['target_team', 'tscore_3ma', 'tscore_5ma', 'tscore_10ma', 'tscore_15ma',
                 'tscore_30ma', 'target_score','opponent_score', 'opponent_team']]
latest = latest.rename(columns={'target_score': 'tscore_lastgame',
                                'opponent_score': 'opscore_lastgame',
                                'opponent_team': 'opteam_lastgame'})

# Merge the latest match to world cup dataset
dfc = dfc.merge(latest,
                            left_on=['target_team'],
                            right_on=['target_team'])

dfc['is_home'] = dfc['target_team'] == 'Russia'
dfc['is_home'] = dfc['is_home'].astype(int)
dfc['is_world_cup'] = 1
dfc['is_stake'] = 1
dfc['neutral'] = (dfc['target_team'] != 'Russia') & (dfc['opponent_team'] != 'Russia')
dfc['neutral'] = dfc['neutral'].astype(int)
dfc['date'] = pd.to_datetime('2018-06-07')
## Merge continent to df
dfc = dfc.merge(continent_df,
              left_on=['target_team'],
              right_on=['countryName'])
dfc = dfc.merge(continent_df,
              left_on=['opponent_team'],
              right_on=['countryName'],
              suffixes=('_target', '_opponent'))
dfc = dfc.merge(rankings,
              left_on=['date', 'target_team'],
              right_on=['rank_date', 'country_full'])
dfc = dfc.merge(rankings,
              left_on=['date', 'opponent_team'],
              right_on=['rank_date', 'country_full'],
              suffixes=('_home', '_away'))
dfc = dfc.rename(index=str, columns={'rank_home': 'target_rank',
                                   'rank_away': 'opponent_rank',
                                   'weighted_points_home': 'target_weighted_points',
                                   'weighted_points_away': 'opponent_weighted_points',
                                   'cur_year_avg_weighted_home': 'target_cur_avg_points',
                                   'cur_year_avg_weighted_away': 'opponent_cur_avg_points',
                                   'two_year_ago_weighted_home': 'target_2y_avg_points',
                                   'two_year_ago_weighted_away': 'opponent_2y_avg_points',
                                   'three_year_ago_weighted_home': 'target_3y_avg_points',
                                   'three_year_ago_weighted_away': 'opponent_3y_avg_points'})

dfc['rank_difference'] = dfc['target_rank'] - dfc['opponent_rank']
dfc['average_rank'] = (dfc['target_rank'] + dfc['opponent_rank']) / 2
dfc['point_difference'] = dfc['target_weighted_points'] - dfc['opponent_weighted_points']
print(dfc.shape)


le = preprocessing.LabelEncoder()
categoricals2 = ['continent_target', 'continent_opponent']
dfc[categoricals2] = dfc[categoricals2].apply(le.fit_transform)
np.save('../labels/classes2.npy', le.classes_)
# Predict the World Cup 2018
categoricals = ['target_team','opponent_team',
                'opteam_lastgame']

team_points = ['target_cur_avg_points',
               'target_2y_avg_points',
               'target_3y_avg_points',
               'opponent_cur_avg_points',
               'opponent_2y_avg_points',
               'opponent_3y_avg_points']

varname_X = ['is_home','is_world_cup',
             'neutral', 'average_rank',
             'rank_difference', 'point_difference',
             'is_stake', 'tscore_lastgame',
             'tscore_3ma', 'tscore_5ma', 'tscore_10ma',
             'tscore_15ma', 'tscore_30ma'] + team_points

dfc = dfc[varname_X + categoricals + categoricals2]

print(dfc.shape)
dfc.to_csv('../data/pred_world_cup.csv', index=False)

