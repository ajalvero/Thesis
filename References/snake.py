import numpy as np
from scipy import stats
print(stats.__name__)

#Descriptive Statistics
A = np.array([[10, 14, 11, 7, 9.5], [8, 9, 17, 14.5, 12],
              [15, 7, 11.5, 10, 10.5], [11, 11, 9, 12, 14]])
print(A)

# Mean (Location Parameter)
print("Mean: ", np.mean(A, axis=0))

# Median (Location Parameter)
print("Median: ", np.median(A, axis=0))

# Variance (Scale Parameter)
print("Variance: ", np.var(A, axis=0, ddof=1))  #ddof=1 provides an unbiased estimator of the variance

# IQR (Scale Parameter)
print("IQR: ", stats.iqr(A, axis=0))

# Skewness (Shape Parameter)
print("Skewness: ", stats.skew(A, axis=0))

# Kurtosis (Shape Parameter)
print("Kurtosis: ", stats.kurtosis(A, axis=0, bias=False))

# You can also quickly get descriptive statistics with a single function
print("Descriptive statistics:\n", stats.describe(A, axis=0))

import pandas as pd

dr = pd.read_csv('../../assets/data/rainfall.csv', index_col=0, header=0, parse_dates=True)
print(dr.head())
print(dr.describe())

# Mean (Location Parameter)
print(dr.mean())

# Median (Location Parameter)
print(dr.median())

# Variance (Scale Parameter)
print(dr.var())

# Skewness (Shape Parameter)
print(dr.skew())

# Kurtosis (Shape Parameter)
print(dr.kurtosis())

#Distributions
from scipy.stats import norm

bins = np.arange(5 - 3 * 2, 5 + 3 * 2, 0.01)

PDF = norm.pdf(bins, loc=5, scale=2)  # generate PDF in bins
CDF = norm.cdf(bins, loc=5, scale=2)  # generate CDF in bins
SF = norm.sf(bins, loc=5, scale=2)  # generate survival function (1-CDF)
PPF = norm.ppf(0.5, loc=5, scale=2)  # obtain percent point (inverse of CDF)
RVS = norm.rvs(loc=5, scale=2, size=1000)  # generate 1000 random variates
MMS = norm.stats(loc=5, scale=2, moments='mvsk')  # obtain the four moments

