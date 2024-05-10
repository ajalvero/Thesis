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

#CNO Proportion of Total Pop

df['Mexican Proportion of Total Pop'] = df['B03001_004E'] / df['B03001_001E']

print(df['Mexican Proportion of Total Pop'])

df['PR Proportion of Total Pop'] = df['B03001_005E'] / df['B03001_001E']

print(df['PR Proportion of Total Pop'])

df['Cuban Proportion of Total Pop'] = df['B03001_006E'] / df['B03001_001E']

print(df['Cuban Proportion of Total Pop'])

df['Dominican Proportion of Total Pop'] = df['B03001_007E'] / df['B03001_001E']

print(df['Dominican Proportion of Total Pop'])

df['Central American Proportion of Total Pop'] = df['B03001_008E'] / df['B03001_001E']

print(df['Central American Proportion of Total Pop'])

df['Costa Rican Proportion of Total Pop'] = df['B03001_009E'] / df['B03001_001E']

print(df['Costa Rican Proportion of Total Pop'])

df['Guatemalan Proportion of Total Pop'] = df['B03001_010E'] / df['B03001_001E']

print(df['Guatemalan Proportion of Total Pop'])

df['Honduran Proportion of Total Pop'] = df['B03001_011E'] / df['B03001_001E']

print(df['Honduran Proportion of Total Pop'])

df['Nicaraguan Proportion of Total Pop'] = df['B03001_012E'] / df['B03001_001E']

print(df['Nicaraguan Proportion of Total Pop'])

df['Panamanian Proportion of Total Pop'] = df['B03001_013E'] / df['B03001_001E']

print(df['Panamanian Proportion of Total Pop'])

df['Salvadoran Proportion of Total Pop'] = df['B03001_014E'] / df['B03001_001E']

print(df['Salvadoran Proportion of Total Pop'])

df['Other CA Proportion of Total Pop'] = df['B03001_015E'] / df['B03001_001E']

print(df['Other CA Proportion of Total Pop'])

df['South American Proportion of Total Pop'] = df['B03001_016E'] / df['B03001_001E']

print(df['South American Proportion of Total Pop'])

df['Argentinean Proportion of Total Pop'] = df['B03001_017E'] / df['B03001_001E']

print(df['Argentinean Proportion of Total Pop'])

df['Bolivian Proportion of Total Pop'] = df['B03001_018E'] / df['B03001_001E']

print(df['Bolivian Proportion of Total Pop'])

df['Chilean Proportion of Total Pop'] = df['B03001_019E'] / df['B03001_001E']

print(df['Chilean Proportion of Total Pop'])

df['Colombian Proportion of Total Pop'] = df['B03001_020E'] / df['B03001_001E']

print(df['Colombian Proportion of Total Pop'])

df['Ecuadorian Proportion of Total Pop'] = df['B03001_021E'] / df['B03001_001E']

print(df['Ecuadorian Proportion of Total Pop'])

df['Paraguayan Proportion of Total Pop'] = df['B03001_022E'] / df['B03001_001E']

print(df['Paraguayan Proportion of Total Pop'])

df['Peruvian Proportion of Total Pop'] = df['B03001_023E'] / df['B03001_001E']

print(df['Peruvian Proportion of Total Pop'])

df['Uruguayan Proportion of Total Pop'] = df['B03001_024E'] / df['B03001_001E']

print(df['Uruguayan Proportion of Total Pop'])

df['Venezuelan Proportion of Total Pop'] = df['B03001_025E'] / df['B03001_001E']

print(df['Venezuelan Proportion of Total Pop'])

df['Other SA Proportion of Total Pop'] = df['B03001_026E'] / df['B03001_001E']

print(df['Other SA Proportion of Total Pop'])

df['Other HL Proportion of Total Pop'] = df['B03001_027E'] / df['B03001_001E']

print(df['Other HL Proportion of Total Pop'])

df['Spaniard Proportion of Total Pop'] = df['B03001_028E'] / df['B03001_001E']

print(df['Spaniard Proportion of Total Pop'])

df['Spanish Proportion of Total Pop'] = df['B03001_029E'] / df['B03001_001E']

print(df['Spanish Proportion of Total Pop'])

df['Spanish American Proportion of Total Pop'] = df['B03001_030E'] / df['B03001_001E']

print(df['Spanish American Proportion of Total Pop'])

df['Other Proportion of Total Pop'] = df['B03001_031E'] / df['B03001_001E']

print(df['Other Proportion of Total Pop'])



 # Assuming df is your DataFrame

# Filter the DataFrame where "B03001_001E" is greater than or equal to 1000
df_filtered = df[(df['B03001_001E'] >= 1000) & (df['Hispanic Proportion of Total Pop'] >=0)]

# Specify the columns of interest
columns_of_interest = df_filtered.columns[df_filtered.columns.get_loc("Mexican Proportion of Total Pop"):
df_filtered.columns.get_loc("Other Proportion of Total Pop") + 1] 

# Add "ec_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "ec_zip")

# Filter the "Hispanic Proportion of Total Pop" column based on the conditions of df_filtered
hispanic_proportion_filtered = df_filtered['Hispanic Proportion of Total Pop']

# Reindex the filtered "Hispanic Proportion of Total Pop" column to align with df_filtered
hispanic_proportion_filtered = hispanic_proportion_filtered.reindex(df_filtered.index)

# Insert the filtered "Hispanic Proportion of Total Pop" column into columns_of_interest
columns_of_interest = columns_of_interest.insert(1, "Hispanic Proportion of Total Pop")

poverty_filtered = df_filtered["poverty"]

poverty_filtered = poverty_filtered.reindex(df_filtered.index)

columns_of_interest = columns_of_interest.insert(1, "poverty")

bach_filtered = df_filtered["bachelors"]

bach_filtered = bach_filtered.reindex(df_filtered.index)

columns_of_interest = columns_of_interest.insert(1, "bachelors")

# Select only the columns of interest
df_selected = df_filtered[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()

# Display the correlation coefficients for each column with "ec_grp_mem_high_zip"
for column in columns_of_interest:
    correlation_with_ec_zip = correlation_matrix.loc[column, "ec_zip"]
    print(f"Correlation between {column} and ec_zip: {correlation_with_ec_zip}")

# Assuming correlation_matrix is the correlation matrix you calculated
# columns_of_interest includes "ec_zip" and other relevant columns
sns.set(style="whitegrid")

# Sort correlation coefficients by absolute value in descending order
sorted_correlations = correlation_matrix['ec_zip'].sort_values(ascending=False)

# Remove "ec_zip", 'Other Proportion of Hispanic Pop', and 'Spanish Proportion of Hispanic Pop' from the list of columns to plot
columns_to_plot = sorted_correlations.drop(['ec_zip', 'Other Proportion of Total Pop', 'Spanish Proportion of Total Pop', 'bachelors', 'poverty']).index

modified_column_names = [col.replace(' Proportion of Total Pop', '') for col in columns_to_plot]

# Replace "Hispanic Proportion of Total Pop" with "Total Hispanic" in the modified column names
modified_column_names = [col.replace('Hispanic Proportion of Total Pop', 'Total Hispanic') for col in modified_column_names]

# Replace "PR" with "Puerto Rican" in the modified column names
modified_column_names = [col.replace('PR', 'Puerto Rican') for col in modified_column_names]

# Replace "Other HL" with "Other Hispanic/Latinx" in the modified column names
modified_column_names = [col.replace('Other HL', 'Other Hispanic/Latinx') for col in modified_column_names]

# Replace "Other SA" with "Other South American" in the modified column names
modified_column_names = [col.replace('Other SA', 'Other S. American') for col in modified_column_names]

# Replace "Other CA" with "Other Central American" in the modified column names
modified_column_names = [col.replace('Other CA', 'Other C. American') for col in modified_column_names]

# Replace "South American" with "Total South American" in the modified column names
modified_column_names = [col.replace('South American', 'Total S. American') for col in modified_column_names]

# Replace "Central American" with "Total Central American" in the modified column names
modified_column_names = [col.replace('Central American', 'Total C. American') for col in modified_column_names]

# Plotting the two-sided horizontal bar graph
plt.figure(figsize=(10, 8))
sns.barplot(data=correlation_matrix.loc[columns_to_plot].reset_index(), x='ec_zip', y='index', palette='coolwarm')

# Customize plot
plt.title('Correlation with Economic Connectedness')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Countries of National Origin')
#plt.grid(axis='x')

plt.xlim(-0.6, 0.6)

# Display the correlation values on the bars
for index, value in enumerate(correlation_matrix.loc[columns_to_plot, 'ec_zip']):
    # Offset the negative numbers to the left of the bars
    if value < 0:
        plt.text(value - 0.06, index, f'{value:.2f}', va='center')
    else:
        plt.text(value + 0.01, index, f'{value:.2f}', va='center')

# Set modified column names as y-axis tick labels
plt.yticks(range(len(modified_column_names)), modified_column_names)

plt.tight_layout()
plt.show()

"""
# Initialize an empty list to store results
chi2_results = []

# Loop through each column (excluding 'ec_zip')
for column in df_selected.columns:
    if column != 'ec_zip':
        # Create a contingency table
        contingency_table = pd.crosstab(df_selected[column], df_selected['ec_zip'])
        
        # Perform chi-squared test
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Append results to the list
        chi2_results.append({'Variable': column, 'Chi-squared': chi2_stat, 'P-value': p_value})

# Convert the list of dictionaries to a DataFrame
chi2_results_df = pd.DataFrame(chi2_results)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Hide axes
ax.axis('off')

# Create the table
table = ax.table(cellText=chi2_results_df.values,
                 colLabels=chi2_results_df.columns,
                 cellLoc='center',
                 loc='upper center')

# Set font size
table.set_fontsize(14)

# Adjust layout
table.scale(1.2, 1.2)

# Save the table as an image
plt.savefig('chi2_results_table.png', bbox_inches='tight', pad_inches=0.1)

# Show the table
plt.show()
"""

# Initialize an empty list to store results
results = []

# Loop through each identity column
for column in df_selected.columns[1:]:  # Start from index 1 to skip 'ec_zip'
    # Select the independent variable (X) and dependent variable (y)
    X = df_selected[['poverty', 'bachelors', column]]
    y = df_selected['ec_zip']
    
    # Concatenate X and y and drop rows with NaN values
    df_cleaned = pd.concat([X, y], axis=1).dropna()
    
    # Assign cleaned independent and dependent variables
    X_cleaned = df_cleaned[['poverty', 'bachelors', column]]
    y_cleaned = df_cleaned['ec_zip']
    
    # Add a constant term to the independent variable (X) to fit the intercept
    X_cleaned = sm.add_constant(X_cleaned)
    
    # Create and fit the linear regression model
    model = sm.OLS(y_cleaned, X_cleaned).fit()
    
    # Get the p-value and correlation
    p_value = model.pvalues[-1]  # Index 1 for the p-value of the independent variable
    correlation = df_cleaned.corr().loc[column, 'ec_zip']
    
    # Append results to the list
    results.append({'Identity': column, 'P-value': p_value, 'Correlation': correlation})

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# Define your independent variable (X) and dependent variable (y)
X = df_selected[['Argentinean Proportion of Total Pop']]
y = df_selected['ec_zip']

# Concatenate X and y and drop rows with NaN values
df_cleaned = pd.concat([X, y], axis=1).dropna()

# Assign cleaned independent and dependent variables
X_cleaned = df_cleaned[['Argentinean Proportion of Total Pop']]
y_cleaned = df_cleaned['ec_zip']

# Add a constant term to the independent variable (X) to fit the intercept
X_cleaned = sm.add_constant(X_cleaned)

# Create and fit the linear regression model
model = sm.OLS(y_cleaned, X_cleaned).fit()

# Get the confidence intervals at a 95% confidence level
conf_int_95 = model.conf_int(alpha=0.05)

# Print the confidence intervals
print("95% Confidence Intervals:")
print(conf_int_95)

# Print the summary of the regression model
print(model.summary())


"""
#mexican_corr = df['Mexican Proportion of Hispanic Pop'].corr(df['ec_zip'])
#cuban_corr = df['Cuban Proportion of Hispanic Pop'].corr(df['ec_zip'])

#print(f"Correlation between Mexican Proportion and ec_zip: {mexican_corr}")
#print(f"Correlation between Cuban Proportion and ec_zip: {cuban_corr}")

# Assuming mexican_corr and cuban_corr are your correlation coefficients
#z_stat, p_value = fisher_exact([[1, len(df) - 3], [1, len(df) - 3]], alternative='two-sided')
#
#print(f"Fisher's z-test p-value: {p_value}")

# Assuming df is your DataFrame
X = df[['Mexican Proportion of Total Pop', 'Cuban Proportion of Total Pop']]
y = df['ec_zip']

# Drop any missing values if necessary
df_cleaned = pd.concat([X, y], axis=1).dropna()

# Perform linear regression
result = linregress(df_cleaned['Mexican Proportion of Total Pop'], df_cleaned['ec_zip'])

# Print the regression results
print(f"Slope (m): {result.slope}")
print(f"Intercept (b): {result.intercept}")
print(f"R-squared: {result.rvalue**2}")
print(f"P-value: {result.pvalue}")
print(f"Standard error: {result.stderr}")

print(mpl.__version__)
"""
"""
# Filter the DataFrame where "B03001_001E" is greater than or equal to 4500
df_filtered = df[(df['B03001_001E'] >= 1000) & (df['Hispanic Proportion of Total Pop'] >= 0.05)]

# Specify the columns of interest
columns_of_interest = df_filtered.columns[df_filtered.columns.get_loc("Hispanic Proportion of Total Pop"):
df_filtered.columns.get_loc("Other Proportion of Hispanic Pop") + 1]

# Add "ec_zip" to the list of columns
columns_of_interest = columns_of_interest.insert(0, "ec_zip")

# Select only the columns of interest
df_selected = df_filtered[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()

# Calculate partial correlation coefficients for each column
partial_correlations = {}
for column in correlation_matrix[1:]:
    partial_corr = pg.partial_corr(df, x='ec_zip', y=column, covar='poverty')['r'].values[0]
    partial_correlations[column] = partial_corr

# Display partial correlation coefficients for each column
for column, partial_corr in partial_correlations.items():
    print(f"Partial correlation between {column} and ec_zip controlling for proportion of people in poverty: {partial_corr:.2f}")

# Display the correlation coefficients for each column with "ec_grp_mem_high_zip"
for column in columns_of_interest:
    correlation_with_ec_zip = correlation_matrix.loc[column, "ec_zip"]
    print(f"Correlation between {column} and ec_zip: {correlation_with_ec_zip}")

# Assuming correlation_matrix is the correlation matrix you calculated
# columns_of_interest includes "ec_zip" and other relevant columns
sns.set(style="whitegrid")

# Create a DataFrame with the partial correlation coefficients
plot_data = pd.DataFrame(list(partial_correlations.items()), columns=['variable', 'partial_corr'])

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='variable', y='partial_corr', data=plot_data, marker='o', color='blue')

# Plotting the two-sided horizontal bar graph
plt.figure(figsize=(10, 8))
sns.barplot(data=correlation_matrix.loc[columns_to_plot].reset_index(), x='ec_zip', y='index', palette='coolwarm')

# Customize plot
plt.title('Partial Correlation with ec_zip controlling for proportion of people in poverty')
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
"""
# Assuming your DataFrame is named df

# Select rows where "Mexican Proportion of Hispanic Pop" is exactly 1.0 and exclude NaN values
mexican_proportion_1_no_nan = df[df["Mexican Proportion of Hispanic Pop"].eq(1.0)].dropna(subset=["Mexican Proportion of Hispanic Pop"])

# Print the values from the selected columns
print(mexican_proportion_1_no_nan[["Mexican Proportion of Hispanic Pop", "NAME"]])

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = mexican_proportion_1_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'Mexican Proportion of Hispanic Pop' is 1.0: {average_ec_grp_mem_high_zip:.2f}")


# Select rows where "Mexican Proportion of Hispanic Pop" is exactly 1.0 and exclude NaN values
sa_proportion_1_no_nan = df[df["South American Proportion of Hispanic Pop"].eq(1.0)].dropna(subset=["South American Proportion of Hispanic Pop"])

# Print the values from the selected columns
print(sa_proportion_1_no_nan[["South American Proportion of Hispanic Pop", "NAME"]])

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = sa_proportion_1_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'South American Proportion of Hispanic Pop' is 1.0: {average_ec_grp_mem_high_zip:.2f}")

# Assuming df is your DataFrame
mean_value = np.mean(df['ec_grp_mem_high_zip'])
print("Mean:", mean_value)

# Select rows where "South American Proportion of Hispanic Pop" is greater than or equal to 0.5 and exclude NaN values
sa_proportion_0_5_no_nan = df[df["South American Proportion of Hispanic Pop"] >= 0.5].dropna(subset=["South American Proportion of Hispanic Pop"])

# Print the values from the selected columns
print(sa_proportion_0_5_no_nan[["South American Proportion of Hispanic Pop", "NAME"]])

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = sa_proportion_0_5_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'South American Proportion of Hispanic Pop' is greater than or equal to 0.5: {average_ec_grp_mem_high_zip:.2f}")

# Select rows where "Mexican Proportion of Hispanic Pop" is greater than or equal to 0.5 and exclude NaN values
mexican_proportion_0_5_no_nan = df[df["Mexican Proportion of Hispanic Pop"] >= 0.5].dropna(subset=["Mexican Proportion of Hispanic Pop"])

# Print the values from the selected columns
print(mexican_proportion_0_5_no_nan[["Mexican Proportion of Hispanic Pop", "NAME"]])

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = mexican_proportion_0_5_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'Mexican Proportion of Hispanic Pop' is greater than or equal to 0.5: {average_ec_grp_mem_high_zip:.2f}")

# Select rows where "Mexican Proportion of Hispanic Pop" is greater than or equal to 0.25 and exclude NaN values
mexican_proportion_0_25_no_nan = df[df["Mexican Proportion of Hispanic Pop"] >= 0.25].dropna(subset=["Mexican Proportion of Hispanic Pop"])

# Print the values from the selected columns
print(mexican_proportion_0_25_no_nan[["Mexican Proportion of Hispanic Pop", "NAME"]])

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = mexican_proportion_0_25_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'Mexican Proportion of Hispanic Pop' is greater than or equal to 0.25: {average_ec_grp_mem_high_zip:.2f}")

# Calculate the average value for "ec_grp_mem_high_zip" within the group
average_ec_grp_mem_high_zip = sa_proportion_0_5_no_nan["ec_grp_mem_high_zip"].mean()

# Print the average value
print(f"Average ec_grp_mem_high_zip for rows where 'South American Proportion of Hispanic Pop' is greater than or equal to 0.5: {average_ec_grp_mem_high_zip:.2f}")
"""