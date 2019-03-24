# Predicting NYC House Prices

## Intro
In this project, I try to predict the price of houses sold in NYC in 2016 and 2017.

## Data
The data is provided at and was downloaded from kaggle (https://www.kaggle.com/new-york-city/nyc-property-sales/version/1). The data on kaggle is a “concatenated and slightly cleaned-up version of the New York City Department of Finance’s Rolling Sales dataset”. Info on the variable definitions is available at (https://www.kaggle.com/saswataroy09/new-york-house-price-prediction and https://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf)

## Methodology
Using a 80-20 train-test split, I train linear and non-linear models on the training set, tune them via 3-fold cross-validation and finally evaluate their predictive performance on the test set. The models used are:

* Linear regression
* Linear regression with L1 penalty (Lasso)
* Linear regression with L2 penalty (Ridge)
* Random Forrests
* Gradient Boosted Trees (sklearn)
* Gradient Boosted Trees (Light GBM)

## Main results
The best results are attained by the Light GBM version of Gradient Boosted Trees, with a test-set performance of

* Median absolute error: $119k
* Median percentage error: 18%

For the combined train and test sets, the selling price has a median of $630k and a standard deviation of $17,159k.

[Detailed results](results.md)

![Results](/figures/model_performance_lgbm_zoom.png)

## To do

* Add exploratory feature plots
* Fine tune Gradient Boosting algos
* Add SVR

The full code is located in the [pre_process file](pre_process.py) and the  [tune_predict_evaluate file](tune_predict_evaluate.py) files.
