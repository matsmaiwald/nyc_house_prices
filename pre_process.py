"""File to pre_process the data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    filename='pre_process_info.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

name_df_input = "nyc-rolling-sales.csv"
df = pd.read_csv(name_df_input)

# Renaming columns to have nicer format.
df.columns = map(str.lower, df.columns)
df.columns = map(lambda s: s.replace(" ", "_"), df.columns)

# Drop columns that are empty and contain no information.
cols_to_drop = ["unnamed:_0", "ease-ment"]
df.drop(cols_to_drop, axis="columns", inplace=True)

# Add names of boroughs.
borough_map = {1: "Manhattan",
               2: "Bronx",
               3: "Brooklyn",
               4: "Queens",
               5: "Staten Island"
               }
df["borough_name"] = df["borough"].map(borough_map)

# because sales price and sale date have missing values, pandas reads them by
# default as strings, so need to cast them as numerics.
for col_name in ["sale_price", "land_square_feet", "gross_square_feet"]:
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

df.rename(index=str, columns={"sale_price": "selling_price"}, inplace=True)

df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month

df["age_at_sale"] = df["sale_year"] - df["year_built"]

# Tax class attributes, zip code and year plus month should be categorical.
for col_name in ["tax_class_at_time_of_sale",
                 "tax_class_at_present",
                 "zip_code",
                 "sale_year",
                 "sale_month"
                 ]:
    df[col_name] = df[col_name].astype("category")

# Check for possible duplicates and drop them
msg = 'There are {:.0f} duplicated rows which will be dropped.'.format(
        sum(df.duplicated(df.columns))
        )
logging.info(msg)
df = df.drop_duplicates(df.columns)


# Remove observations with missing sales price or year built
msg = '{:.0f} rows without selling price info were dropped.'.format(
        sum(df["selling_price"].isnull())
        )
df = df[df["selling_price"].notnull()]
logging.info(msg)

msg = '{:.0f} rows with a selling price of $10 or below were dropped.'.format(
        sum(df["selling_price"] <= 10)
        )
df = df[df["selling_price"] > 10]
logging.info(msg)

msg = '{:.0f} rows with a construction date before 1500 were dropped.'.format(
        sum(df["year_built"] < 1500)
        )
df = df[df["year_built"] > 1500]
logging.info(msg)

msg = '{:.0f} rows with a size of zero square feet were dropped.'.format(
        sum(~(df['land_square_feet'] > 0))
        )
df = df[df["land_square_feet"] > 0]
logging.info(msg)


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

# TO DO: include full set or a larger subset
vars_one_hot = vars_one_hot_simple

vars_to_keep = vars_num + vars_one_hot

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df[vars_one_hot])
#one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

#Drop original categorical features and keep one hot encoded dummies
df_temp = df[vars_to_keep]

df_temp.drop(vars_one_hot,axis=1,inplace=True)
df_clean = pd.concat([df_temp, one_hot_encoded], axis=1)
del df_temp

# index by selling price
df_clean.sort_values(by=['selling_price'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# Recast dummies from int to float format since some algos require all floats
df_clean = df_clean.astype('float64')

# Summary stats
df_clean.apply(lambda x: logging.info(x.describe()), axis=0)

# Export pre processed dataset
df_clean.to_csv('./output/data_clean.csv', index=False)

# Plotting distributions of Selling prices
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
fig.suptitle("Selling prices are approximately log-normally distributed", size=16)
fig.savefig("selling_prices_dist_log.png")
plt.close(fig)



fig, ax = plt.subplots(5)
fig.suptitle("Selling prices vary across boroughs and are highest in Manhattan",
             size=16)
for i in range(0,5):
    ax[i].hist(
            df_clean[df_clean[
                    'borough_name_' + borough_map[i+1] # borough map starts at 1
                    ] == 1]['selling_price'].apply(np.log),
            bins = 100)
    ax[i].set(
            title=borough_map[i+1],
            xlabel="Log(selling price) in $",
            ylabel='Frequency',
            xlim=(10,20)
            )
    fig.savefig("selling_prices_dist_boroughs.png")
    plt.close(fig)
