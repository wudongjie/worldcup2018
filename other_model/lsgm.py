
import lightgbm as lgb

feature_name = varname_X + categoricals
data_lgbm = df.loc[:, feature_name + ['target_score']]
train = data_lgbm.sample(frac=0.8, random_state=200)
test = data_lgbm.drop(train.index)
print(train.shape)
print(len(feature_name))
train_X, train_y = train[feature_name], train['target_score']
test_X, test_y = test[feature_name], test['target_score']
print(train_X.dtypes)
print(train_X.shape)


train_data = lgb.Dataset(train_X, label=train_y,
                         feature_name=feature_name,
                         categorical_feature=categoricals+['is_home',
                                                           'is_world_cup',
                                                           'neutral',
                                                           'is_stake'])
test_data = lgb.Dataset(test_X, label=test_y, reference=train_data)
param = {'num_leaves':32, 'learning_rate':0.01, 'num_iterations': 100,
         'objective':'multiclass', 'num_class':32}
param['metric'] = 'multi_logloss'
#param = {'num_leaves':18, 'learning_rate':0.01, 'num_iterations': 5000,
#         'objective':'regression_l2', 'boosting': 'dart'}
#param['metric'] = 'rmse'
num_round = 10
bst = lgb.train(param, train_data, num_round,valid_sets=[test_data])
bst.save_model('model1.txt')
y_pred = bst.predict(test_X)
x_pred = bst.predict(train_X)
best_preds_svm = [np.argmax(line) for line in y_pred]
x_preds_svm = [np.argmax(line) for line in x_pred]
#best_preds_svm = np.rint(y_pred)
print("First 10 predictions: {0}".format(best_preds_svm[:10]))
# Make guess of goals
prediction_corr = best_preds_svm - test_y
plt.hist(prediction_corr)      #use this to draw histogram of your data
plt.show()
print("The correct prediction count: {0} given {1}".format(prediction_corr[prediction_corr==0.0].size,
                                                           prediction_corr.size))
