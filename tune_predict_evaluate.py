# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:28:37 2019

@author: Mats Ole
"""
import pandas as pd
import numpy as np
import sklearn as sk
import time
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn import ensemble

random_seed = 42

logging.basicConfig(
    filename='tune_predict_evaluate_info.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """Mean absolute percentage error regression loss.

    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
        """
    output_errors = np.average(np.abs(y_pred / y_true - 1),
                               weights=sample_weight, axis=0)

    return(output_errors)


def median_absolute_percentage_error(y_true, y_pred):
    """Median absolute percentage error regression loss.

    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
        """
    output_errors = np.median(np.abs(y_pred / y_true - 1), axis=0)

    return(output_errors)


def plot_model_performance(y_pred, y_test, model_name, zoom=False):
    """Save a scatter plot of the predicted vs actuals."""

    "zoom: If yes, zoom in on the part of the distribution where most data lie."

    if (zoom == True):
        axes_limit = 0.2 * 1e7
        path_suffix = "_zoom"
    else:
        axes_limit = y_pred.max()*1.1
        path_suffix = ""

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred, alpha=0.1)
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    subplot_title = "{} \n Median AE: {:.0f}, MAE: {:.0f}, \n Median APE: {:.3f}, MAPE: {:.3f}".format(
                model_name,
                median_absolute_error(y_test, y_pred),
                mean_absolute_error(y_test, y_pred),
                median_absolute_percentage_error(y_test, y_pred),
                mean_absolute_percentage_error(y_test, y_pred))
        #ax[i].get_xaxis().get_major_formatter().set_scientific(False)
    ax.set(title=subplot_title,
           xlabel='Actual selling price in $',
           ylabel='Predicted selling price in $',
           xlim=(0, axes_limit),
           ylim=(0, axes_limit)
    )
    fig.savefig("./figures/model_performance_" + model_name + path_suffix + ".png", dpi=1000)
    plt.close(fig)


# train-test split ------------------------------------------------------------

df_clean = pd.read_csv('./output/data_clean.csv')

X = df_clean.drop(columns=["selling_price"])
y = df_clean["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=random_seed
                                                    )

# Define models to be tuned----------------------------------------------------
scoring = 'neg_mean_absolute_error'
models = []
models.append((
        'Lasso',
        ElasticNet(normalize=True, tol=0.1),
        [{'ttregressor__regressor__l1_ratio': [1],
          'ttregressor__regressor__alpha': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]}]
        )
)

models.append((
        'Ridge',
        ElasticNet(normalize=True, tol=0.1),
        [{'ttregressor__regressor__l1_ratio': [0],
          'ttregressor__regressor__alpha': [5e-3, 1e-3, 5e-4, 1e-4]}]
        )
)

models.append((
    'RF',
    RandomForestRegressor(random_state=random_seed),
    [{
        'ttregressor__regressor__max_features': [200, 300],
        "ttregressor__regressor__max_depth": [15],
        "ttregressor__regressor__n_estimators": [500]}]
        )
        )
models.append((
    'GB_lad',
    ensemble.GradientBoostingRegressor(random_state=random_seed,
                                       loss='lad',
                                       learning_rate=0.01,
                                       n_estimators=1500
                                       ),
    [{'ttregressor__regressor__max_features': [10, 50, 100]}]
))
models.append((
    'GB_ls',
    ensemble.GradientBoostingRegressor(random_state=random_seed,
                                       loss='ls',
                                       learning_rate=0.01,
                                       n_estimators=1500
                                       ),
    [{'ttregressor__regressor__max_features': [10, 50, 100]}]
))

# Tune and evaluate models-----------------------------------------------------
tuned_models = []
for name, model, grid in models:
    my_pipeline = sk.pipeline.Pipeline([
             ('scale', StandardScaler()),
             ('ttregressor',
              TransformedTargetRegressor(
                      regressor=model,
                      func=np.log,
                      inverse_func=np.exp
                      )
              )
              ])
    t0 = time.time()
    logging.info("# Tuning hyper-parameters for %s---------------------" % name)
    print("# Tuning hyper-parameters for %s" % name)
    current_model = GridSearchCV(my_pipeline,
                                 grid,
                                 cv=3,
                                 scoring=scoring
                                 )

    current_model.fit(X_train, y_train)
    tuned_models.append((name, current_model.best_estimator_))

    print("Best parameters set found on train set:")
    print(current_model.best_params_)
    print("Grid scores on train set:")
    means = current_model.cv_results_['mean_test_score']
    stds = current_model.cv_results_['std_test_score']
    for mean, std, params in zip(
                                 means,
                                 stds,
                                 current_model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    logging.info("Score on the test set:")
    y_pred = current_model.predict(X_test)
    logging.info("R2 score: " + str(round(r2_score(y_test, y_pred), 3)))
    logging.info('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 2)))
    t1 = time.time()
    msg_time = 'Tuning the ' + name + ' model took ' + str(round(t1 - t0, 2)) + ' seconds.'
    print(msg_time)
    plot_model_performance(y_pred, y_test, name)
    plot_model_performance(y_pred, y_test, name, zoom=True)

# Models that are not tuned----------------------------------------------------

# Linear Regression
lin_reg = Pipeline([('scaling', StandardScaler()),
                    ('ttregressor',
                     TransformedTargetRegressor(
                             regressor=LinearRegression(),
                             func=np.log,
                             inverse_func=np.exp
                             )
                     )
                    ])
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
plot_model_performance(y_pred, y_test, "lin_reg")
plot_model_performance(y_pred, y_test, "lin_reg", zoom=True)

# LGBM
# create dataset for lightgbm using log-transform for selling price
lgb_train = lgb.Dataset(X_train, np.log(y_train))
lgb_test = lgb.Dataset(X_test, np.log(y_test))

lgb_params = {
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

t0 = time.time()
# train
print("# Tuning hyper-parameters for LGBM")
logging.info("# Tuning hyper-parameters for LGBM-----------------------------")
evals_result = {}  # to record eval results for plotting
gbm = lgb.train(lgb_params,
                lgb_train,
                num_boost_round=1500,
                # evals_result=evals_result,
                # early_stopping_rounds=5
                )
t1 = time.time()
msg_time = ('Tuning the Light GBM' +
            'model took {} seconds.'.format(str(round(t1 - t0, 2)))
            )

logging.info(msg_time)
y_pred = np.exp(gbm.predict(X_test))
logging.info("Score on the test set:")
logging.info("R2 score: " + str(round(r2_score(y_test, y_pred), 3)))
logging.info('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 2)))

plot_model_performance(y_pred, y_test, "lgbm")
plot_model_performance(y_pred, y_test, "lgbm", zoom=True)
