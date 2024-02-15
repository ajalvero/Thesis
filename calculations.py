import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm

df = pd.read_csv('./merged_forreal6.csv')

# Assuming your DataFrame is named df

#Bachelors or Higher
# Add the values in columns C15002I_006E and C15002I_0011E together for each row
df['combined_bachelors'] = df['C15002I_006E'] + df['C15002I_011E']

# Calculate the proportion of the combined values relative to C15002I_001E
df['bachelors'] = df['combined_bachelors'] / df['C15002I_001E']

# Now, df['bachelors'] contains the calculated proportions

# Display the first few rows of the DataFrame with the new columns
print(df[['C15002I_006E', 'C15002I_011E', 'combined_bachelors', 'C15002I_001E', 'bachelors']].head())

#Less than High School
# Add the values in columns C15002I_003E and C15002I_008E together
df['combined_lesshigh'] = df['C15002I_003E'] + df['C15002I_008E']

# Calculate the proportion of combined values to C15002I_001E
df['lesshigh'] = df['combined_lesshigh'] / df['C15002I_001E']

# Display the first few rows of the DataFrame with the new columns
print(df[['C15002I_003E', 'C15002I_008E', 'combined_lesshigh', 'C15002I_001E', 'lesshigh']].head())

# Add the values in columns C15002I_004E and C15002I_009E together
df['combined_HS'] = df['C15002I_004E'] + df['C15002I_009E']

# Calculate the proportion of combined values to C15002I_001E
df['HS'] = df['combined_HS'] / df['C15002I_001E']

# Display the first few rows of the DataFrame with the new columns
print(df[['C15002I_004E', 'C15002I_009E', 'combined_HS', 'C15002I_001E', 'HS']].head())

# Add the values in columns C15002I_004E and C15002I_010E together
df['combined_AA'] = df['C15002I_004E'] + df['C15002I_010E']

# Calculate the proportion of combined values to C15002I_001E
df['AA'] = df['combined_AA'] / df['C15002I_001E']

# Display the first few rows of the DataFrame with the new columns
print(df[['C15002I_004E', 'C15002I_010E', 'combined_AA', 'C15002I_001E', 'AA']].head())

df['poverty'] = df['B17020I_002E'] / df['B17020I_001E']


# Assuming df is your DataFrame
columns_of_interest = df.columns[df.columns.get_loc("Hispanic Proportion of Total Pop"):
                                  df.columns.get_loc("Other Proportion of Hispanic Pop") + 1]

# Add "civic_organizations_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "civic_organizations_zip")

# Select only the columns of interest
df_selected = df[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()

# Display the correlation coefficients for each column with "civic_organizations_zip"
for column in columns_of_interest:
    correlation_with_civic_organizations_zip = correlation_matrix.loc[column, "civic_organizations_zip"]
    print(f"Correlation between {column} and civic_organizations_zip: {correlation_with_civic_organizations_zip}")

# Assuming correlation_matrix is the correlation matrix you calculated
# columns_of_interest includes "civic_organizations_zip" and other relevant columns
sns.set(style="whitegrid")

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=correlation_matrix, x="civic_organizations_zip", y=columns_of_interest, marker='o', color='blue')

# Customize plot
plt.title('Correlation with civic_organizations_zip')
plt.xlabel('civic_organizations_zip')
plt.ylabel('Correlation Coefficient')

# Display the correlation values on the data points
for column in columns_of_interest:
    correlation_with_civic_organizations_zip = correlation_matrix.loc[column, "civic_organizations_zip"]
    plt.annotate(f'{correlation_with_civic_organizations_zip:.2f}', (correlation_with_civic_organizations_zip, column), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()


"""
# Assuming df is your DataFrame and columns_of_interest includes "civic_organizations_zip" and other relevant columns
columns_of_interest = df.columns[df.columns.get_loc("Hispanic Proportion of Total Pop"):
                                  df.columns.get_loc("Other Proportion of Hispanic Pop") + 1]

# Add "civic_organizations_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "civic_organizations_zip")

# Calculate partial correlation coefficients for each column
partial_correlations = {}
for column in columns_of_interest[1:]:
    partial_corr = pg.partial_corr(df, x='civic_organizations_zip', y=column, covar='poverty')['r'].values[0]
    partial_correlations[column] = partial_corr

# Display partial correlation coefficients for each column
for column, partial_corr in partial_correlations.items():
    print(f"Partial correlation between {column} and civic_organizations_zip controlling for proportion of people in poverty: {partial_corr:.2f}")

# Create a DataFrame with the partial correlation coefficients
plot_data = pd.DataFrame(list(partial_correlations.items()), columns=['variable', 'partial_corr'])

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='variable', y='partial_corr', data=plot_data, marker='o', color='blue')

# Customize plot
plt.title('Partial Correlation with civic_organizations_zip controlling for proportion of people in poverty')
plt.xlabel('Variable')
plt.ylabel('Partial Correlation Coefficient')

# Display the partial correlation values on the data points
for i, row in plot_data.iterrows():
    plt.annotate(f'{row["partial_corr"]:.2f}', (i, row['partial_corr']),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.xticks(range(len(plot_data)), plot_data['variable'], rotation=45, ha='right')  # Use range(len(plot_data)) for custom x-axis ticks

plt.tight_layout()
plt.show()
"""

"""
# Assuming df is your DataFrame
columns_of_interest = df.columns[df.columns.get_loc("Hispanic Proportion of Total Pop"):
                                  df.columns.get_loc("Other Proportion of Hispanic Pop") + 1]

# Add "civic_organizations_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "civic_organizations_zip")

# Initialize containers for results
partial_correlations = {}
p_values = {}
r_squared_values = {}

for column in columns_of_interest[1:]:
    # Skip rows with NaN values
    df_temp = df[[column, 'civic_organizations_zip', 'poverty']].dropna()

    # Calculate partial correlation coefficients
    partial_corr_result = pg.partial_corr(df_temp, x='civic_organizations_zip', y=column, covar='poverty')
    partial_corr = partial_corr_result['r'].values[0]
    partial_correlations[column] = partial_corr

    # Extract p-value
    p_value = partial_corr_result['p-val'].values[0]
    p_values[column] = p_value

    # Calculate R-squared using linear regression
    X = sm.add_constant(df_temp[['civic_organizations_zip', 'poverty']])  # Add a constant term
    y = df_temp[column]

    model = sm.OLS(y, X).fit()
    r_squared_values[column] = model.rsquared

    # If the variables are categorical, perform a chi-squared test
    contingency_table = pd.crosstab(df_temp['civic_organizations_zip'], df_temp[column])
    _, chi2_p_value, _, _ = chi2_contingency(contingency_table)
    p_values[column + '_chi2'] = chi2_p_value

# Create a DataFrame with the partial correlation coefficients, p-values, and R-squared values
plot_data = pd.DataFrame({
    'variable': list(partial_correlations.keys()),
    'partial_corr': list(partial_correlations.values()),
    'p_value': [p_values[col] for col in partial_correlations.keys()],
    'r_squared': [r_squared_values[col] for col in partial_correlations.keys()]
})

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='variable', y='partial_corr', data=plot_data, marker='o', color='blue')

# Customize plot
plt.title('Partial Correlation with civic_organizations_zip controlling for proportion of people in poverty')
plt.xlabel('Variable')
plt.ylabel('Partial Correlation Coefficient')

# Display the partial correlation values, p-values, and R-squared values on the data points
for i, row in plot_data.iterrows():
    plt.annotate(f'{row["partial_corr"]:.2f}\n(p-value: {row["p_value"]:.4f}, R-squared: {row["r_squared"]:.4f})',
                 (i, row['partial_corr']),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.xticks(range(len(plot_data)), plot_data['variable'], rotation=45, ha='right')  # Use range(len(plot_data)) for custom x-axis ticks

plt.tight_layout()
plt.show()

# Continue with the rest of your code for plotting
"""



