"""Script to pre-process the raw data.

This file loads the raw data, performs different steps of data cleaning and
saves a clean version of the processed data to be used as input by the
tune_predic_evaluate.py script.

Parameters
----------
name_df_input : str
    name of the raw data file.
name_data_clean : str
    name under which the clean data is saved in the 'output' folder.
pre_process_info.log : str
    name of logging outputfile.

"""

import logging
import pandas as pd
# import seaborn as sns
# from sklearn.preprocessing import Imputer

# Set parameters and logging configuration ------------------------------------
name_df_input = "nyc-rolling-sales.csv"
name_data_clean = 'data_clean.csv'
name_log_file = 'pre_process_info.log'

logging.basicConfig(
    filename=name_log_file,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# Load and clean raw data -----------------------------------------------------
df = pd.read_csv(name_df_input)

# Remove blanks in column names.
df.columns = map(str.lower, df.columns)
df.columns = map(lambda s: s.replace(" ", "_"), df.columns)

# Drop columns that are empty and contain no information.
cols_to_drop = ["unnamed:_0", "ease-ment"]
df.drop(cols_to_drop, axis="columns", inplace=True)

# Add variable with names rather than id of boroughs.
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

# for conveniance, use 'selling price' rather than 'sale price' from now on
df.rename(index=str, columns={"sale_price": "selling_price"}, inplace=True)

# split date out into year and month variables
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month

# compute age of building at moment of sale
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


# Remove observations with non-sensical data values
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

# choose columns to keep in the clean data ------------------------------------

keep_vars_num = ["residential_units",
                 "commercial_units",
                 "land_square_feet",
                 # "gross_square_feet", TO DO: infer missing values
                 "selling_price",
                 "age_at_sale"
                 ]

keep_vars_one_hot_full = ["neighborhood",
                          "building_class_category",
                          # "zip_code",
                          "tax_class_at_time_of_sale",
                          "borough_name",
                          "sale_year",
                          "sale_month"
                          ]

keep_vars_one_hot_simple = ["tax_class_at_time_of_sale",
                            "borough_name",
                            "sale_year"
                            ]

# TO DO: include full set or a larger subset
keep_vars_one_hot = keep_vars_one_hot_full

keep_vars_all = keep_vars_num + keep_vars_one_hot

# Convert categorical variables into dummy variables.
one_hot_encoded = pd.get_dummies(df[keep_vars_one_hot])
# one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

# Drop original categorical features and keep one hot encoded dummies
df_temp = df[keep_vars_all]

df_temp.drop(keep_vars_one_hot, axis=1, inplace=True)
df_clean = pd.concat([df_temp, one_hot_encoded], axis=1)
del df_temp

# index by selling price
df_clean.sort_values(by=['selling_price'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# Recast dummies from int to float format since some algos require all floats
df_clean = df_clean.astype('float64')

# Summary stats
df_clean.apply(lambda x: logging.info(x.describe()), axis=0)

# Export pre-processed dataset
df_clean.to_csv('./output/' + name_data_clean, index=False)
