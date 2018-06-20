import pandas as pd
import json
import numpy as np
countryList = []
with open("countries.json", "r") as jsonfile:
    data = json.load(jsonfile)
    print(data)
    for key, val in data.items():
        countryList.append({'countryName': val['name'],
                            'continent': val['continent']})

continent_df = pd.DataFrame(countryList)
print(continent_df.head(5))

# Team Specific Data
df = pd.read_csv('./cleaned.csv')
df['tscore_lastgame'] = df.groupby('target_team')['target_score'].shift(1)
df['opteam_lastgame'] = df.groupby('target_team')['opponent_team'].shift(1)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=3, min_periods=1).mean())
df['tscore_3ma'] = new_col.reset_index(level=0, drop=True)
new_col = df.groupby('target_team', as_index=False).apply(lambda x: x['target_score'].rolling(window=10, min_periods=1).mean())
df['tscore_10ma'] = new_col.reset_index(level=0,drop=True)
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
df.to_csv('./cleaned.csv')