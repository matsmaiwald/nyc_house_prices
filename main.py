import numpy as np
import pandas as pd

name_df_input = "nyc-rolling-sales.csv"
df = pd.read_csv(name_df_input)

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

df["sale_price"] = pd.to_numeric(df["sale_price"], errors='coerce')
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


# Remove observations with missing SALE PRICE
sum(df["sale_price"].isnull())
df = df[df["sale_price"].notnull()]



# TO DO: consider which variables to keep and do one hot feature encoding

column_model=['BOROUGH','BUILDING CLASS CATEGORY','COMMERCIAL UNITS','GROSS SQUARE FEET',
               'SALE PRICE','Building Age During Sale','LAND SQUARE FEET','RESIDENTIAL UNITS','seasons']
data_model=data.loc[:,column_model]

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(data_model[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

#Drop original categorical features and keep one hot encoded feature
data_model.drop(one_hot_features,axis=1,inplace=True)
data_model=pd.concat([data_model,one_hot_encoded],axis=1)
data_model.head()
