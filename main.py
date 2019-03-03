import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import model_selection
import seaborn as sns

name_df_input = "nyc-rolling-sales.csv"
df = pd.read_csv(name_df_input)

# TO DO: first look at data
df.head()
df.columns = map(str.lower, df.columns)
df.columns = map(lambda s: s.replace(" ", "_"), df.columns)

cols_to_drop = ["unnamed:_0", "ease-ment"]
df.drop(cols_to_drop, axis="columns", inplace=True)

borough_map = {1: "Manhattan",
               2: "Bronx",
               3: "Brooklyn",
               4: "Queens",
               5: "Staten Island"
               }
df["borough_name"] = df["borough"].map(borough_map)


#------------------------------------------------------------------------------


df.isnull().sum()
# No zero values

df.info()

# because sales price and sale date have missing values, pandas reads them by 
# default as strings
for col_name in ["sale_price", "land_square_feet", "gross_square_feet"]:
    df[col_name] = pd.to_numeric(df[col_name], errors = "coerce")

df["selling_price"] = pd.to_numeric(df["sale_price"], errors='coerce') # rename
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month

df["age_at_sale"] = df["sale_year"] - df["year_built"]

#Both tax class attributes, zip code and year plus month should be categorical
for col_name in ["tax_class_at_time_of_sale", 
                 "tax_class_at_present", 
                 "zip_code",
                 "sale_year",
                 "sale_month"
                 ]:
    df[col_name] = df[col_name].astype("category")

#Check for possible duplicates and drop them
sum(df.duplicated(df.columns))
df = df.drop_duplicates(df.columns)


# Remove observations with missing sales price or year built
sum(df["selling_price"].isnull())
df = df[df["selling_price"].notnull()]

sum(df["selling_price"] == 0)
df = df[df["selling_price"] > 10]

sum(df["year_built"] < 1500)
df = df[df["year_built"] > 1500]

sum(~(df['land_square_feet'] > 0))
df = df[df["land_square_feet"] > 0]

list(df.columns.values)

# Name vars to keep
vars_num = ["residential_units",
            "commercial_units",
            "land_square_feet",   # TO DO: include these and infer missing values
            #"gross_square_feet",
            "selling_price",
            "age_at_sale"
            ]

vars_one_hot_full = ["neighborhood",
                     "building_class_category",
                     "zip_code",
                     "tax_class_at_time_of_sale",
                     "borough_name",
                     "sale_year",
                     "sale_month"
                     ]

vars_one_hot_simple = ["tax_class_at_time_of_sale",
                       "borough_name",
                       "sale_year"
                       ]

vars_one_hot = vars_one_hot_simple

vars_to_keep = vars_num + vars_one_hot

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df[vars_one_hot])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

#Drop original categorical features and keep one hot encoded feature
df_temp = df[vars_to_keep]

df_temp.drop(vars_one_hot,axis=1,inplace=True)
df_clean = pd.concat([df_temp, one_hot_encoded], axis=1)
del df_temp
df_clean.sort_values(by=['selling_price'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# Summary stats
df_clean_summary = df_clean.describe()

# Plotting distribution of Selling prices
selling_price_pct_95 = np.percentile(df_clean['selling_price'], 95)
plt.style.use('seaborn')

fig, ax = plt.subplots(1,2)
ax[0].hist(
        df_clean[df_clean['selling_price'] < selling_price_pct_95]['selling_price'], 
        bins = 100)
ax[0].set(
        title='Distribution of bottom 95% of selling prices', 
        xlabel='Selling price in $', 
        ylabel='Frequency'
        )
ax[1].hist(df_clean['selling_price'].apply(np.log), bins = 100)
ax[1].set(
        title='Distribution of log-transformed selling price',
        xlabel='Log(selling price) in $',
        ylabel='Frequency'
        )
fig.suptitle("Sales prices are approximately log-normally distributed", size=16)




fig, ax = plt.subplots(5)
fig.suptitle("Selling prices vary across boroughs and are highest in Manhattan", 
             size=16)
for i in range(0,5):
    print(borough_map[i+1])
    ax[i].hist(
            df_clean[df_clean[
                    'borough_name_' + borough_map[i+1]
                    ] == 1]['selling_price'].apply(np.log), 
            bins = 100)
    ax[i].set(
            title=borough_map[i+1],
            xlabel="Log(selling price) in $",
            ylabel='Frequency',
            xlim=(10,20)
            )

X = df_clean.drop(columns=["selling_price"])
y = df_clean["selling_price"]

# train-test split ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

random_seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=random_seed
                                                    )



# Tune models------------------------------------------------------------------
models = []
models.append(('ELN', ElasticNet(normalize=True, tol=0.1)))


scoring = 'neg_mean_absolute_error'
# Set the parameters by cross-validation
tuned_parameters = [{'l1_ratio': [1], 'alpha': [1e-1, 1e-2, 1e-3, 1e-4]},
                    {'l1_ratio': [0], 'alpha': [1e-1, 1e-2, 1e-3, 1e-4]}
                    ]

for name, model in models:
    print("# Tuning hyper-parameters for %s" % name)
    print()

    clf = GridSearchCV(model, tuned_parameters, cv=3,
                       scoring=scoring)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(r2_score(y_true, y_pred))
    print(mean_absolute_error(y_true, y_pred))
    print()


# Comparing CV Results---------------------------------------------------------
models = []
models.append(('ELN', ElasticNet()))
models.append(('OLS', LinearRegression()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=random_seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
