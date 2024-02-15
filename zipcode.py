#thesis time?

import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
from scipy.stats import linregress
from scipy.stats import pearsonr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import pingouin as pg

# Load the CSV files into DataFrames
df1 = pd.read_csv('./soc_cap_zip copy.csv')
df2 = pd.read_csv('./census copy.csv', skiprows=[1])

# Assuming column 1 in df1 contains zip codes with "ZCTA5 " prefix
# Remove the prefix and convert zip codes to integers
df2['NAME'] = df2['NAME'].str.lstrip('ZCTA5 ')

print(df2.head)

df1['zip'] = df1['zip'].apply(lambda x: f"{int(x):05d}" if pd.notna(x) and isinstance(x, (int, float)) else x)

print(df1.head)

merged_df = pd.merge(df2, df1, left_on='NAME', right_on='zip', how='left')

merged_df.to_csv('merged_final.csv', index=False)

df3 = pd.read_csv('./merged_final.csv')

print(df3.head)

value_at_row_135 = df3.loc[134, 'NAME']
print(value_at_row_135)

val135 = df3.loc[134, 'zip']
print(val135)

third_column_name = df3.columns[2]
print(third_column_name)

#Proportions of CNO

df3['Hispanic Proportion of Total Pop'] = df3['B03001_003E'] / df3['B03001_001E']

print(df3['Hispanic Proportion of Total Pop'])

df3['Mexican Proportion of Hispanic Pop'] = df3['B03001_004E'] / df3['B03001_003E']

print(df3['Mexican Proportion of Hispanic Pop'])

df3['PR Proportion of Hispanic Pop'] = df3['B03001_005E'] / df3['B03001_003E']

print(df3['PR Proportion of Hispanic Pop'])

df3['Cuban Proportion of Hispanic Pop'] = df3['B03001_006E'] / df3['B03001_003E']

print(df3['Cuban Proportion of Hispanic Pop'])

df3['Dominican Proportion of Hispanic Pop'] = df3['B03001_007E'] / df3['B03001_003E']

print(df3['Dominican Proportion of Hispanic Pop'])

df3['Central American Proportion of Hispanic Pop'] = df3['B03001_008E'] / df3['B03001_003E']

print(df3['Central American Proportion of Hispanic Pop'])

df3['Costa Rican Proportion of Hispanic Pop'] = df3['B03001_009E'] / df3['B03001_003E']

print(df3['Costa Rican Proportion of Hispanic Pop'])

df3['Guatemalan Proportion of Hispanic Pop'] = df3['B03001_010E'] / df3['B03001_003E']

print(df3['Guatemalan Proportion of Hispanic Pop'])

df3['Honduran Proportion of Hispanic Pop'] = df3['B03001_011E'] / df3['B03001_003E']

print(df3['Honduran Proportion of Hispanic Pop'])

df3['Nicaraguan Proportion of Hispanic Pop'] = df3['B03001_012E'] / df3['B03001_003E']

print(df3['Nicaraguan Proportion of Hispanic Pop'])

df3['Panamanian Proportion of Hispanic Pop'] = df3['B03001_013E'] / df3['B03001_003E']

print(df3['Panamanian Proportion of Hispanic Pop'])

df3['Salvadoran Proportion of Hispanic Pop'] = df3['B03001_014E'] / df3['B03001_003E']

print(df3['Salvadoran Proportion of Hispanic Pop'])

df3['Other CA Proportion of Hispanic Pop'] = df3['B03001_015E'] / df3['B03001_003E']

print(df3['Other CA Proportion of Hispanic Pop'])

df3['South American Proportion of Hispanic Pop'] = df3['B03001_016E'] / df3['B03001_003E']

print(df3['South American Proportion of Hispanic Pop'])

df3['Argentinean Proportion of Hispanic Pop'] = df3['B03001_017E'] / df3['B03001_003E']

print(df3['Argentinean Proportion of Hispanic Pop'])

df3['Bolivian Proportion of Hispanic Pop'] = df3['B03001_018E'] / df3['B03001_003E']

print(df3['Bolivian Proportion of Hispanic Pop'])

df3['Chilean Proportion of Hispanic Pop'] = df3['B03001_019E'] / df3['B03001_003E']

print(df3['Chilean Proportion of Hispanic Pop'])

df3['Colombian Proportion of Hispanic Pop'] = df3['B03001_020E'] / df3['B03001_003E']

print(df3['Colombian Proportion of Hispanic Pop'])

df3['Ecuadorian Proportion of Hispanic Pop'] = df3['B03001_021E'] / df3['B03001_003E']

print(df3['Ecuadorian Proportion of Hispanic Pop'])

df3['Paraguayan Proportion of Hispanic Pop'] = df3['B03001_022E'] / df3['B03001_003E']

print(df3['Paraguayan Proportion of Hispanic Pop'])

df3['Peruvian Proportion of Hispanic Pop'] = df3['B03001_023E'] / df3['B03001_003E']

print(df3['Peruvian Proportion of Hispanic Pop'])

df3['Uruguayan Proportion of Hispanic Pop'] = df3['B03001_024E'] / df3['B03001_003E']

print(df3['Uruguayan Proportion of Hispanic Pop'])

df3['Venezuelan Proportion of Hispanic Pop'] = df3['B03001_025E'] / df3['B03001_003E']

print(df3['Venezuelan Proportion of Hispanic Pop'])

df3['Other SA Proportion of Hispanic Pop'] = df3['B03001_026E'] / df3['B03001_003E']

print(df3['Other SA Proportion of Hispanic Pop'])

df3['Other HL Proportion of Hispanic Pop'] = df3['B03001_027E'] / df3['B03001_003E']

print(df3['Other HL Proportion of Hispanic Pop'])

df3['Spaniard Proportion of Hispanic Pop'] = df3['B03001_028E'] / df3['B03001_003E']

print(df3['Spaniard Proportion of Hispanic Pop'])

df3['Spanish Proportion of Hispanic Pop'] = df3['B03001_029E'] / df3['B03001_003E']

print(df3['Spanish Proportion of Hispanic Pop'])

df3['Spanish American Proportion of Hispanic Pop'] = df3['B03001_030E'] / df3['B03001_003E']

print(df3['Spanish American Proportion of Hispanic Pop'])

df3['Other Proportion of Hispanic Pop'] = df3['B03001_031E'] / df3['B03001_003E']

print(df3['Other Proportion of Hispanic Pop'])

#Correlation Between CNO and Clustering

mexican_corr = df3['Mexican Proportion of Hispanic Pop'].corr(df3['clustering_zip'])
cuban_corr = df3['Cuban Proportion of Hispanic Pop'].corr(df3['clustering_zip'])

print(f"Correlation between Mexican Proportion and clustering_zip: {mexican_corr}")
print(f"Correlation between Cuban Proportion and clustering_zip: {cuban_corr}")

# Assuming mexican_corr and cuban_corr are your correlation coefficients
z_stat, p_value = fisher_exact([[1, len(df3) - 3], [1, len(df3) - 3]], alternative='two-sided')

print(f"Fisher's z-test p-value: {p_value}")

# Assuming df is your DataFrame
X = df3[['Mexican Proportion of Hispanic Pop', 'Cuban Proportion of Hispanic Pop']]
y = df3['clustering_zip']

# Drop any missing values if necessary
df_cleaned = pd.concat([X, y], axis=1).dropna()

# Perform linear regression
result = linregress(df_cleaned['Mexican Proportion of Hispanic Pop'], df_cleaned['clustering_zip'])

# Print the regression results
print(f"Slope (m): {result.slope}")
print(f"Intercept (b): {result.intercept}")
print(f"R-squared: {result.rvalue**2}")
print(f"P-value: {result.pvalue}")
print(f"Standard error: {result.stderr}")

print(mpl.__version__)


mexican_corr = df3['Mexican Proportion of Hispanic Pop'].corr(df3['ec_high_se_zip'])
cuban_corr = df3['Cuban Proportion of Hispanic Pop'].corr(df3['ec_high_se_zip'])

print(f"Correlation between Mexican Proportion and ec_high_se_zip: {mexican_corr}")
print(f"Correlation between Cuban Proportion and ec_high_se_zip: {cuban_corr}")

df3['Spaniard Proportion of Hispanic Pop']

span_corr = df3['Spaniard Proportion of Hispanic Pop'].corr(df3['ec_high_se_zip'])

print(f"Correlation between Spaniard Proportion and ec_high_se_zip: {span_corr}")

edu = pd.read_csv('./education.csv')
pov = pd.read_csv('./poverty.csv')
urb = pd.read_csv('./rural.csv')
tech = pd.read_csv('./tech.csv')

print(pov.head)
print(edu.head)

edu['NAME'] = edu['NAME'].str.lstrip('ZCTA5 ')

edu.rename(columns={'GEO_ID': 'GEOID'}, inplace=True)

edu.rename(columns={'NAME': 'ZCODE'}, inplace=True)

print(edu.head)

pov['NAME'] = pov['NAME'].str.lstrip('ZCTA5 ')

pov.rename(columns={'NAME': 'ZIPCODE'}, inplace=True)

pov.rename(columns={'GEO_ID': 'GID'}, inplace=True)

print(pov.head)

merged_df2 = pd.merge(df3, edu, left_on='GEO_ID', right_on='GEOID', how='left')

merged_df2.to_csv('merged_forreal2.csv', index=False)

merged_df3 = pd.merge(merged_df2, pov, left_on='GEO_ID', right_on='GID', how='left')

merged_df3.to_csv('merged_forreal3.csv', index=False)

num_rows = len(urb)
print(f"Number of rows: {num_rows}")

nub_rows = len(merged_df3)
print(f"Number of rows: {nub_rows}")

merged_df5 = pd.merge(merged_df3, tech, left_on='NAME', right_on='zcta19', how='left')

merged_df5.to_csv('merged_forreal5.csv', index=False)

merged_df6 = pd.merge(merged_df5, urb, left_on='NAME', right_on='ZIP_CODE', how='inner')

merged_df6.to_csv('merged_forreal6.csv', index=False)

# Assuming merged_df6 is your DataFrame and columns_of_interest includes "ec_grp_mem_high_zip" and other relevant columns
columns_of_interest = merged_df6.columns[merged_df6.columns.get_loc("Hispanic Proportion of Total Pop"):
                                          merged_df6.columns.get_loc("Other Proportion of Hispanic Pop") + 1]

# Add "ec_grp_mem_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "ec_grp_mem_high_zip")

# Calculate partial correlation coefficients for each column
partial_correlations = {}
for column in columns_of_interest[1:]:
    partial_corr = pg.partial_corr(merged_df6, x='ec_grp_mem_high_zip', y=column, covar='avg_download_speed')['r'].values[0]
    partial_correlations[column] = partial_corr

# Display partial correlation coefficients for each column
for column, partial_corr in partial_correlations.items():
    print(f"Partial correlation between {column} and ec_grp_mem_high_zip controlling for avg_download_speed: {partial_corr:.2f}")

# Create a DataFrame with the partial correlation coefficients
plot_data = pd.DataFrame(list(partial_correlations.items()), columns=['variable', 'partial_corr'])

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='variable', y='partial_corr', data=plot_data, marker='o', color='blue')

# Customize plot
plt.title('Partial Correlation with ec_grp_mem_high_zip controlling for avg_download_speed')
plt.xlabel('Variable')
plt.ylabel('Partial Correlation Coefficient')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

plt.tight_layout()
plt.show()





# %%
#pd.Series(df3['ec_high_se_zip']).plot(kind='density') # or pd.Series()

#density = stats.kde.gaussian_kde(df3['ec_high_se_zip'])
#x = np.arange(0., 8, .1)
#mpl.plot(x, density(x))
#mpl.show()