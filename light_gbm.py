# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:25:04 2019

@author: Mats Ole
"""

import lightgbm as lgb
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, np.log(y_train))
lgb_test = lgb.Dataset(X_test, np.log(y_test))


# specify your configurations as a dict
params = {
    'task': 'train',
    'objective': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 42
}

print('Start training...')
# train
evals_result = {} # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1500,
                #evals_result=evals_result,
                #early_stopping_rounds=5
                )

y_pred = np.exp(gbm.predict(X_test))

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)
subplot_title = "{} \n Median AE: {:.0f}, Mean AE: {:.0f}, Median APE: {:.3f}, Mean APE: {:.3f}".format(
            "LightGBM", 
            median_absolute_error(y_test, y_pred), 
            mean_absolute_error(y_test, y_pred),
            median_absolute_percentage_error(y_test,y_pred),
            mean_absolute_percentage_error(y_test, y_pred))
    #ax[i].get_xaxis().get_major_formatter().set_scientific(False)
ax.set(
                title=subplot_title,
                xlabel='Actual selling price in $',
                ylabel='Predicted selling price in $'
                )
fig.suptitle("Final model comparison of prediction vs actuals", size=16)


gbm.save_model('model.txt')