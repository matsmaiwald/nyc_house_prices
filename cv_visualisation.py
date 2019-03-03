# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:28:15 2019

@author: Mats Ole
"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Use this function to create a plot    
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Create an array of alphas and lists to store scores
models = []
models.append(('Ridge', Ridge(normalize=True)))
models.append(('Lasso', Lasso(normalize=True, tol=0.1)))

alpha_space = np.logspace(-4, 0, 50)


for name, model in models:
    model_scores = []
    model_scores_std = []
    # Create a ridge regressor: ridge
    
    # Compute scores over range of alphas
    for alpha in alpha_space:
    
        # Specify the alpha value to use: ridge.alpha
        model.alpha = alpha
        
        # Perform 10-fold CV: ridge_cv_scores
        model_cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring ='neg_mean_absolute_error')
        
        # Append the mean of ridge_cv_scores to ridge_scores
        model_scores.append(np.mean(model_cv_scores))
        
        # Append the std of ridge_cv_scores to ridge_scores_std
        model_scores_std.append(np.std(model_cv_scores))
    display_plot(model_scores, model_scores_std)