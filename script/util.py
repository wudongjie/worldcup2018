import pandas as pd

def pointGenerator(tscore, opscore):
    if (tscore > opscore):
        point = 3
    elif (tscore == opscore):
        point = 1
    else:
        point = 0
    return point


def pointVisualizer(df, world_cup_csv):
    dfw = pd.read_csv(world_cup_csv)
    dfw = dfw.replace({"IRAN": "Iran",
                                   "Costarica": "Costa Rica",
                                   "Porugal": "Portugal",
                                   "Columbia": "Colombia",
                                   "Korea": "Korea Republic"})
    grouped = df.groupby('target_team').agg({'point': 'sum',
                                             'target_score_x': 'sum',
                                             'target_score_y': 'sum'})
    grouped['net_score'] = grouped['target_score_x'] - grouped['target_score_y']
    grouped = grouped.merge(dfw,
                            left_on=['target_team'],
                            right_on=['Team'])
    grouped = grouped[['Team', 'Group', 'point', 'net_score','target_score_x']]
    grouped = grouped.rename(columns={'target_score_x': 'total_score'})
    grouped = grouped.sort_values(['Group', 'point', 'net_score', 'total_score'], ascending=[1, 0, 0, 0])
    grouped[['net_score', 'total_score']] = grouped[['net_score', 'total_score']].astype(int)
    return grouped

def main():  # test
    df = pd.read_csv("../result/all_pred.csv")
    df['point'] = df.apply(lambda x: pointGenerator(x['target_score_x'], x['target_score_y']), axis=1)
    print(df.shape)
    grouped = pointVisualizer(df, '../data/World Cup 2018 Dataset.csv')
    grouped.to_csv("../result/group_table.csv", index=False)
    print(grouped)

main()