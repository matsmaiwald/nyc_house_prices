# Model Comparison

| Model        | Meadian AE in thousand $ | Median APE  |
| ------------- |:-------------:| -----:|
| Linear Regression | 157.1 | 26.0% |
| Ridge Regression  | 212.1 | 33.8% |
| Lasso Regression  | 221.0  | 36.3% |
| Random Forrest    | 160.9  | 25.2% |
| GBM (scikit-learn)|  145.6   | 21.5% |
| LightGBM          |  115.7 | 17.5% |

# A note on prediction errors
Given the presence of very large outliers -- there are properties sold at $170 million while the median selling price is only $630 thousand -- the models are evaluated with regards to their **Median AE** (median absolute prediction error) and **Median APE** (median absolute percentage error) on the test set as opposed to the typical MAE and MAPE errors.

As house selling prices turn out to be log-normally distributed, I fit the model on the log-transforms of selling prices and use the standard **MAE** (mean absolute error) as loss function**. The reported prediction errors are for the orginial (non-log-transformed) selling prices, however.

![Results](/figures/selling_prices_dist_log.png)

# LightGBM predictions in detail

### Test set price predictions vs actuals for *majority* of data points
![Results](/figures/model_performance_lgbm_zoom.png)

### Test set price predictions vs actuals for *all* data points

![Results](/figures/model_performance_lgbm.png)
