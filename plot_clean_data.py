"""Script to plot the pre-processed data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

borough_map = {1: "Manhattan",
               2: "Bronx",
               3: "Brooklyn",
               4: "Queens",
               5: "Staten Island"
               }

df_clean = pd.read_csv('./output/data_clean.csv')

# Plotting distributions of Selling prices
selling_price_pct_95 = np.percentile(df_clean['selling_price'], 95)
plt.style.use('seaborn')

fig, ax = plt.subplots(1, 2)
ax[0].hist(
        df_clean[
            df_clean['selling_price'] < selling_price_pct_95
            ]['selling_price'], bins=100)
ax[0].set(
        title='Distribution of bottom 95% of selling prices',
        xlabel='Selling price in $',
        ylabel='Frequency'
        )
ax[1].hist(df_clean['selling_price'].apply(np.log), bins=100)
ax[1].set(
        title='Distribution of log-transformed selling price',
        xlabel='Log(selling price) in $',
        ylabel='Frequency'
        )
fig.suptitle(
    "Selling prices are approximately log-normally distributed", size=16
    )
# fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig("./figures/selling_prices_dist_log.png", dpi=1000)
plt.close(fig)

# PLOTTING CORRPLOTS FOR THE MOST IMPORTANT VARIABLES ------------------------

cols = list(df_clean)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('selling_price')))
df_clean = df_clean.loc[:, cols]
corr = df_clean.corr()

correlation = corr['selling_price'].sort_values(ascending=False)

df_corr = df_clean[correlation.head(10).index]



corr = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


df_corr = df_clean[correlation.tail(10).index]



corr = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



# =============================================================================
# fig, ax = plt.subplots(5)
# fig.suptitle(
#     "Selling prices vary across boroughs and are highest in Manhattan",
#     size=16
#     )
# for i in range(0, 5):
#     ax[i].hist(
#             df_clean[df_clean[  # borough_map starts at 1
#                     'borough_name_' + borough_map[i+1]
#                     ] == 1]['selling_price'].apply(np.log),
#             bins=100)
#     ax[i].set(
#             title=borough_map[i+1],
#             xlabel="Log(selling price) in $",
#             ylabel='Frequency',
#             xlim=(10, 20)
#             )
#     # fig.tight_layout()
#     fig.subplots_adjust(top=0.85)
#     fig.savefig("./figures/selling_prices_dist_boroughs.png", dpi=1000)
#     plt.close(fig)
# =============================================================================
