
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin
from sklearn import preprocessing
df = pd.read_csv('../data/cleaned.csv')

df = df.dropna()
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
categoricals = ['target_team','opponent_team',
            'opteam_lastgame']
df[categoricals] = df[categoricals].apply(le1.fit_transform)
np.save('../labels/classes1.npy', le1.classes_)
# Encode the Continent variables
categoricals2 = ['continent_target', 'continent_opponent']
le2.classes_ = np.load('../labels/classes2.npy')
df[categoricals2] = df[categoricals2].apply(le2.fit_transform)
df['neutral'] = df['neutral'].astype(int)
df['is_stake'] = df['is_stake'].astype(int)




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
             'tscore_3ma', 'tscore_5ma','tscore_10ma',
             'tscore_15ma','tscore_30ma'] + team_points
X, y = df.loc[:, varname_X + categoricals + categoricals2], df['target_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Split Done! The shapes of X_train and Y_train are:")
print(X_train.shape, y_train.shape)

#df = df[df['target_score'] < 10]
#enc = preprocessing.OneHotEncoder()
#enc.fit(df[categoricals])
#df_country = enc.transform(df[categoricals]).toarray()
#X = np.hstack((X, df_country))


class RidgeTransformer(RidgeClassifier, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

class RandomForestTransformer(RandomForestClassifier, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

class GbcTransformer(GradientBoostingClassifier, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

def build_model():
    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', RidgeTransformer()),
            ('rf', RandomForestTransformer()),
            ('gbc', GbcTransformer())
        ]
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model


params = {
        'pred_union__ridge__alpha': 0.5,
        'pred_union__rf__n_estimators': 500,
        'pred_union__rf__random_state': 42,
        'pred_union__gbc__n_estimators': 500,
        'pred_union__gbc__learning_rate': 0.1
        }


def train_models(X_train, y_train, params):
    model = build_model()
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model

def prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("First 10 Prediction Results: {0}".format(y_pred[:10]))
    score = model.score(X_test, y_test)
    guess = np.rint(y_pred)
    guess[guess <= 0] = 0.0
    print('Score: {0}'.format(score))
    prediction_accuracy = guess - y_test
    num_correct = prediction_accuracy[prediction_accuracy==0.0].size
    total_length = prediction_accuracy.size
    accuracy = num_correct/float(total_length)
    print("The correct prediction count: {0} given {1}, the accuracy is: {2}".format(num_correct, total_length,
                                                                                     accuracy))


def train_random_forest(X_train, y_train, n_estimators=500, criterion='entropy', random_state=42):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=criterion, random_state=random_state)
    classifier.fit(X_train, y_train)
    features = X.columns.values
    importances = classifier.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()
    return classifier

#model = train_models(X_train, y_train, params)
model = train_random_forest(X_train, y_train) # Use the Random Forest only and generate the feature importance figure
prediction(model, X_test, y_test)

# Predict world cup 2018
dfw = pd.read_csv("../data/pred_world_cup.csv")
dfw[categoricals] = dfw[categoricals].apply(le1.transform)
pred = model.predict(dfw)
guess_wc = np.rint(pred)
dfw['target_score'] = guess_wc
dfw[categoricals] = dfw[categoricals].apply(le1.inverse_transform)
df_out = dfw[['target_score', 'target_team', 'opponent_team']]
df_out2 = df_out.rename(columns={ 'target_team': 'opponent_team',
                                  'opponent_team': 'target_team',
                                  'opponent_score': 'target_score'})
df_out = df_out.merge(df_out2,
                      left_on=['opponent_team', 'target_team'],
                      right_on=['opponent_team', 'target_team'])
length = df_out.shape[0]
removeList = []
for i in range(length):
    for j in range(i+1, length):
        if df_out['opponent_team'][i] == df_out['target_team'][j] and df_out['target_team'][i] == df_out['opponent_team'][j]:
            removeList.append(j)
df_out = df_out.drop(removeList, 0)
print(df_out)
df_out.to_csv("../result/my_pred.csv")
