# Predicting NYC House Prices

## Intro
In this project, I try to predict the price of houses sold in NYC in 2016 and 2017.

## Data
The data is provided at and was downloaded from kaggle (https://www.kaggle.com/new-york-city/nyc-property-sales/version/1). The data on kaggle is a “concatenated and slightly cleaned-up version of the New York City Department of Finance’s Rolling Sales dataset”. Info on the variable definitions is available (https://www.kaggle.com/saswataroy09/new-york-house-price-prediction)

## Methodology
Using a 80-20 train-test split, I train different models on the training set, tune them via cross-validation and evaluate their predictive performance on the test set. The models used are:

* Linear regression
* Linear regression with L1 penalty (Lasso)
* Linear regression with L2 penalty (Ridge)
* Random Forrests
* Gradient Boosting (Light GBM)

## Main results
So far, the best results are attained by Gradient Boosting, with an Median absolute error of $119k and a median percentage error of 18% given a median selling price of $630k.

![Results](/figures/model_performance_lgbm_zoom.png)

## To do

* Add in borroughs as feature
* make nice exploratory plot
* Tune LGM parameters
* add in SVR

The full code is located in the [pre_process file](pre_process.py) and the  [tune_predict_evaluate file](tune_predict_evaluate.py).
