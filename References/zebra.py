#Defining a Series

import numpy as np
import pandas as pd

# Since 'data' is the first location parameter, you can also ignore 'data=' and directly specify the value
s1 = pd.Series(data=np.array([-1, 3, 8]))    # Numpy array
s2 = pd.Series(data=[-1, 3, 8])              # List
s3 = pd.Series(data={'a':-1, 'b':3, 'c':8})  # Dictionary
s4 = pd.Series(data=-1)                      # Scalar
print(s1)
print(s2)
print(s3)
print(s4)

s5 = pd.Series([-1, 3, 8], index=['x', 'y', 'z'])  # specify the index by 'index' option
print(s5)

arr = np.array([100, 20, -3])

s6 = pd.Series(arr, index=['x', 'y', 'z'] )
print(arr)
print(s6)

arr[2] = -40
print('\n')
print(arr)
print("After reassigning value, the value of series \nalso will be updated automatically:")
print(s6)

print(s6.values)
print(s6.index)

#Selecting the Elements

s = pd.Series(data={'a':-1, 'b':3, 'c':8})
s.iloc[1]

s['a']

print(s[0:2])
print(s['a':'c'])

#Assigning Values to Elements

s['a'] = 100
s.iloc[2] = 80
print(s)

#Filtering Values

s = pd.Series([1, 3, 5, 2, 10])
idx_greater = s > 4
# According to Boolean value to filter
print(idx_greater)
print(s[idx_greater])  # greater than 4

# you cannot use `or` `and` or `not` and only can use corresponding operator symbol(|, & and ~)
filter_condition = (s.isin([2, 5]) | (s<4)) & ~(s>2) 
print(filter_condition)
print(s[filter_condition])

#Operations and Mathematical Functions

s * 2.5
print(s)
np.exp(s)

#Nan Value

# Declaring a 'Series' including the NaN value
s = pd.Series([1, np.NaN, 10, 9, -2, np.NaN])
print(s)

print(s.isnull())
print(s.notnull())

print(s[s.isnull()])
print(s[s.notnull()])

#Operations of Multiple Series

s = pd.Series({"Singapore": 30, "Malaysia": 23, "Vietnam": 36, "Cambodia": 41})
s1 = pd.Series({"China": 51, "Japan": 73, "Vietnam": 36, "Laos": 31})
print(s * s1)

#DataFrame and Defining a DataFrame

df = pd.DataFrame([["Malaysia", "Kuala Lumpur", 'daodao', False],
                   ["Singapore", "Singapore", 5850342, True],
                   ["Vietnam", "Hanoi", 97338579, True]],
                  columns=["Country", "Capital", "Population", "Isdeveloped"],
                  index=["a", "b", "c"])
print(df)

# Array
df = pd.DataFrame(
    np.array([[14, 35, 35, 35], [19, 34, 57, 34], [42, 74, 49, 59]]))
print(df)

# List,  use 'columns' and 'index' parameters to specify the column and index of generated dataframe.
df = pd.DataFrame([["Malaysia", "Kuala Lumpur", 32365999, False],
                   ["Singapore", "Singapore", 5850342, True],
                   ["Vietnam", "Hanoi", 97338579, True]],
                  columns=["Country", "Capital", "Population", "Isdeveloped"],
                  index=["a", "b", "c"])
print(df)

# dict
df = pd.DataFrame({"Country": ["Malaysia", "Singapore", "Vietnam"],
                   "Capital": ["Kuala Lumpur", "Singapore", "Hanoi"],
                   "Population": [32365999, 5850342, 97338579],
                   "Isdeveloped": [False, True, True]},
                  index=["a", "b", "c"])
print(df)

#Selecting the Internal Elements

# use ':' to represent select all
df.iloc[:, 0:2]

df.loc[:, "Country":"Population"]

df.loc["a", ["Country", "Population"]]

df.iloc[[0, 1]] # If you omit number of columns, all columns will be selected 

print(df.index)

df.columns

df.values

print(df["Country"])

df[["Country", "Population"]] # Use list to select multiple columns  

df.Country # Also support as atrribute to select

#Assigning Value

df1 = df.copy(True)
df1.loc["c", "Country"] = "Japan"
df1.loc["c", "Capital"] = "Tokyo"
df1.loc["c", "Population"] = 126476461
df1.loc["c", "Isdeveloped"] = True
print(df1)

df1.loc["c"] = ["Japan", "Tokyo", 126476461, True]
print(df)

#Assigning index, columns, and name of index and columns
df1.index = ["e", "f", "g"]
df1.index.name = "label"
df1.columns.name = "attributes"
df1.columns = ["Coun", "Cap", "Pop", "ID"]
print(df1)

#Delete columns from dataframe
del df1["ID"]
# axis = 1 or columns represents delete columns
df1.drop(["Coun"], axis='columns', inplace=True)

# axis = 0 or rows represents delete columns
df1.drop(["e"], axis='rows', inplace=True)

print(df1)

# inplace=True
df2 = df.copy(True)
print(df2)
print("----")
df2_return = df2.drop(["Country"], axis='columns', inplace=True)
print(df2)
print("----")
print(df2_return)

# inplace=False
df3 = df.copy(True)
print(df3)
print("----")
df3_return = df3.drop(["Country"], axis='columns', inplace=False)
print(df3)
print("----")
print(df3_return)

#Filtering

df2 = pd.DataFrame(np.array([[14, 35, 35, 35],
                             [19, 34, 57, 34],
                             [42, 74, 49, 59]]))
# filtering lesser than 30
print(df2[df2 < 30])

# Filtering accroding to conditions of one column
print(df[df["Population"] < 50000000])

df[(df["Population"] < 50000000) & (df["Isdeveloped"] == True)]

#Transposition of a Dataframe

df1 = df.T
print(df1)

print(df1.index)

print(df1.columns)

#Merge of Dataframe

df1 = pd.DataFrame(np.random.rand(3,4))
df2 = pd.DataFrame(np.random.rand(3,4))
df3 = pd.DataFrame(np.random.rand(6,4))
df4 = pd.DataFrame(np.random.rand(3,6))

# Vertical merging by default.
print(pd.concat([df1, df2, df3, df4]))

print(pd.concat([df1, df2, df3, df4], axis='columns'))

#View Data
df1 = pd.DataFrame(np.random.rand(100,4))
print(df1.head(2))

print(df1.tail(3))

#Computational Tools
df1 = pd.DataFrame(np.random.rand(5, 5), index=['i1', 'i2', 'i3', 'i4', 'i5'],
                   columns=['c1', 'c2', 'c3', 'c4', 'c5'])
print(df1.cov())

df1.corr() # method = pearson (default), optional: kendall, spearman

df1.corr(method='kendall')

# compute average value of each column by default.
df1.mean()

#compute the sum of each row by specifying the axis argument as ‘columns’ or 1.
df1.sum(axis=1)

#display a summary of the characteristics of the dataframe
df1.describe()

#Data Ranking

df1 = df.copy(deep=True)
df1.sort_values(by=['Population', 'Country'], ascending=False, na_position='first')

#NAN value
df1 = pd.DataFrame(np.random.rand(5, 5), index=['i1', 'i2', 'i3', 'i4', 'i5'],
                   columns=['c1', 'c2', 'c3', 'c4', 'c5'])
df1.iloc[0, 1] = np.nan
df1.iloc[2, 2] = np.nan
df1.iloc[3, 1] = np.nan
df1.iloc[3, 3] = np.nan
df1

# detecting nan value
print(df1.isnull())
print(df1.notnull())
print(df1.isna())

# False:0, True:1
df1.isnull().sum(axis=1)

# fill NaN value using a specific value
df1.fillna(value=0)

# delete NaN value
# ‘any’ : If any NA values are present, drop that row or column.
# ‘all’ : If all values are NA, drop that row or column.

# 0, or ‘index’ : Drop rows which contain missing values.
# 1, or ‘columns’ : Drop columns which contain missing value.
df1.dropna(axis="index", how="any")

#Date index
dti = pd.date_range("2018-01-01", periods=3, freq="H")
print(dti)
dti = pd.date_range(start="2021-09-28", end="2021-09-30", freq="10H")
print(dti)

dti = pd.date_range(start="2021-09-28", end="2021-09-30", freq="10H")
dti = dti.tz_localize("UTC")
dti

dti = pd.date_range(start="2021-09-28", end="2021-09-30", freq="10H")
dti = dti.tz_localize("Asia/Singapore")
dti

pd.to_datetime([100, 101, 102], unit="h", origin=pd.Timestamp("1900-01-01 00:00:00"))

#Upsampling and Downsampling
# prepare data, this section will be introduced in the next tutorial
# Data Source: http://www.weather.gov.sg/climate-historical-daily/
data = pd.read_csv('rainfall.csv', index_col=0, header=0, parse_dates=True)
data.head()

"""#Downsampling: Convert monthly data to yearly data by sum or max
df = data.copy(deep=True)
dfsum = df.resample("Y").sum()
dfsum.columns = ["Yearly Rainfall Total (mm)"]
dfsum.head()

dfmax = df.resample("Y").max()
dfmax.columns = ["Yearly Rainfall Maximum (mm)"]
dfmax.head()

# Upsampling: Convert monthly data to 10 days' data 
# by directly return (asfreq) or forward filling (pad/ffill)
dfmax.resample('10D').asfreq()[0:5]

dfmax.resample('10D').ffill()[0:5]

dfmax.resample('D').ffill(limit=2)[0:5]

#Group DataFrame
#You can group DataFrame using a mapper or by a Series of columns via the groupby function.

# Calculate average and maximum wind speed of each station
df = pd.DataFrame({'Station': ['Changi', 'Changi',
                              'Climenti', 'Climenti'],
                   'Wind Speed': [10., 5., 6., 16.]})
print(df.groupby(['Station']).mean())
print(df.groupby(['Station']).max())

# custom function
df = pd.DataFrame({'Station': ['Changi', 'Changi', 'Changi',
                              'Climenti', 'Climenti', 'Climenti'],
                   'Wind Speed': [10., 5., 12, 6., 16., 20]})
def groupNormalization(w):
    return (w-w.min())/(w.max()-w.min())
df.groupby(['Station']).apply(groupNormalization)


#Input/Output of Data

df = pd.read_csv('rainfall.csv', index_col=0, header=0, 
                 parse_dates=True)
print(df.head())

df = pd.read_csv('rainfall.csv', index_col=0, header=0, 
                 parse_dates=True, date_format='%Y-%m-%d')
df.head()

df = pd.read_csv('../../assets/data/Changi_daily_rainfall.csv', index_col=0, header=0, 
                 parse_dates=[0])
df.head()



"""







