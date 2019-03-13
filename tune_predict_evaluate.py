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
from sklearn.preprocessing import StandardScaler

random_seed = 42

logging.basicConfig(
    filename='tune_predict_evaluate_info.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# define custom error functions

def mean_absolute_percentage_error(y_true, y_pred,
                        sample_weight=None):
    """Mean absolute percentage error regression loss
    
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights."""
   

    output_errors = np.average(np.abs(y_pred / y_true -1),
                               weights=sample_weight, axis=0)

    return(output_errors)
    
    
def median_absolute_percentage_error(y_true, y_pred):
    """Median absolute percentage error regression loss
    
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights."""
   

    output_errors = np.median(np.abs(y_pred / y_true -1), axis=0)

    return(output_errors)

# train-test split ------------------------------------------------------------

df_clean = pd.read_csv('./output/data_clean.csv')

X = df_clean.drop(columns=["selling_price"])
y = df_clean["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=random_seed
                                                    )

# Define models----------------------------------------------------------------
scoring = 'neg_mean_absolute_error'

models = []
models.append((
        'Lasso', 
        ElasticNet(normalize=True, tol=0.1),
        [{'ttregressor__regressor__l1_ratio': [1], 
          'ttregressor__regressor__alpha': [1e-1, 1e-2, 1e-3, 1e-4]}]
        )
)

models.append((
        'Ridge', 
        ElasticNet(normalize=True, tol=0.1),
        [{'ttregressor__regressor__l1_ratio': [0], 
          'ttregressor__regressor__alpha': [1e-1, 1e-2, 1e-3, 1e-4]}]
        )
)

models.append((
        'RF', 
         RandomForestRegressor(
                               random_state = random_seed),
                               [{'ttregressor__regressor__max_features':[2,3],
                                 "ttregressor__regressor__max_depth":[3, 9],
                                 "ttregressor__regressor__n_estimators":[50, 500]}])
         )

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
    #print("# Tuning hyper-parameters for %s" % name)
    logging.info("# Tuning hyper-parameters for %s" % name)
    print()
    current_model = GridSearchCV(my_pipeline, grid, cv=3,
                       scoring=scoring)
    

    current_model.fit(X_train, y_train)
    tuned_models.append((name, current_model.best_estimator_))

    logging.info("Best parameters set found on train set:")
    logging.info(current_model.best_params_)
    print()
    logging.info("Grid scores on train set:")
    print()
    means = current_model.cv_results_['mean_test_score']
    stds = current_model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, current_model.cv_results_['params']):
        logging.info("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    logging.info("Score on the test set:")
    y_pred = current_model.predict(X_test)
    logging.info("R2 score: " + str(round(r2_score(y_test, y_pred),3)))
    logging.info('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 2)))
    print()
    t1 = time.time()
    msg_time = 'Tuning the ' + name + ' model took ' + str(round(t1 - t0, 2)) + ' seconds.'
    logging.info(msg_time)
    
    
fig, ax = plt.subplots(1,len(tuned_models))
for i in range(len(tuned_models)):
    y_pred = tuned_models[i][1].predict(X_test)
    ax[i].scatter(y_test, y_pred)
    subplot_title = "{} \n Median AE: {:.0f}, MAE: {:.0f}, \n Median APE: {:.3f}, MAPE: {:.3f}".format(
            tuned_models[i][0], 
            median_absolute_error(y_test, y_pred), 
            mean_absolute_error(y_test, y_pred),
            median_absolute_percentage_error(y_test,y_pred),
            mean_absolute_percentage_error(y_test, y_pred)
            )
    #ax[i].get_xaxis().get_major_formatter().set_scientific(False)
    ax[i].set(
                title=subplot_title,
                xlabel='Actual selling price in $',
                ylabel='Predicted selling price in $'
                )
fig.suptitle("Final model comparison of prediction vs actuals", size=16)
fig.savefig("./figures/model_performance_comparison.png")
plt.close(fig)

# TO DO

# Impute missing data