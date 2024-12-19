import numpy as np

"""
Interpretation of the data

*   each NDVI measurement reflects a different time in the growing season (ranging from planting to peak and harvesting stages)
*   avg NDVI value over the season reflects the general health of the crop
*   higher NDVI value means healthy crop
*   12 NDVI measurements shows vegetation health over a six month time period
*   NDVI01 could represent early-stage vegetation health
*   NDVI06 could represent mid-season growth
*   NDVI12 could represent late-season growth
"""

columns = ['NDVI01', 'NDVI02', 'NDVI03', 'NDVI04', 'NDVI05', 'NDVI06',
'NDVI07', 'NDVI08', 'NDVI09', 'NDVI10', 'NDVI11', 'NDVI12']

"""
Statistical Features - to capture the overall characteristics of NDVI values
"""

# """
# - mean
# - median: the middle NDVI value in the time series is useful for reducing the influence of outliers
# - standard deviation: measures the variability of NDVI values over the season.
#                     : a higher value means sharp growth or decline phases.
# - min and max NDVI
# - range: showing the extent of NDVI variation throughout the season
# """

# function to measure statistical features
def measure_statistical_features(dataset):
  # calculate the mean of NDVI measurements in each time series
  ndvi_columns = dataset[columns]
  ndvi_means = ndvi_columns.mean(axis=1)

  # calculate the median of NDVI measurements in each time series
  ndvi_medians = ndvi_columns.median(axis=1)

  # calculate the standard deviation of NDVI measurements in each time series
  ndvi_std = ndvi_columns.std(axis=1)

  # calculate the min of NDVI measurements in each time series
  ndvi_min = ndvi_columns.min(axis=1)

  # calculate the max of NDVI measurements in each time series
  ndvi_max = ndvi_columns.max(axis=1)

  # calculate the range of NDVI measurements in each time series
  ndvi_range = ndvi_max - ndvi_min

  print(f"Stats for NDVI measurements for year 2021:")
  print(f"Mean:\n{ndvi_means}")
  print(f"\nMedian:\n{ndvi_medians}")
  print(f"\nStandard Deviation:\n{ndvi_std}")
  print(f"\nMin value:\n{ndvi_min}\n\nMax value:\n{ndvi_max}\n")
  print(f"Range:\n{ndvi_range}")


"""
Temporal features - to highlight growth trends and transitions
"""

# function to calculate temporal features
def measure_temporal_features(dataset):
  ndvi_columns = dataset[columns]
  # calculate the time to peak NDVI value
  ndvi_peak_time = ndvi_columns.idxmax(axis=1)

  # each month has two ndvi measurements so map over 6 month time period
  month_to_ndvi_dict = {
      'NDVI01': 'May-1', 'NDVI02': 'May-30', 'NDVI03': 'June-1', 'NDVI04': 'June-31', 'NDVI05': 'July-1',
      'NDVI06': 'July-30', 'NDVI07': 'Aug-1', 'NDVI08': 'Aug-30', 'NDVI09': 'Sep-1', 'NDVI10': 'Sep-31',
      'NDVI11': 'Oct-1', 'NDVI12': 'Oct-30'
  }
  print(f"Time to peak NDVI value:\n{ndvi_peak_time}  {month_to_ndvi_dict}")
  # calculate the rate of change
  # -> difference between executive NDVI values
  # -> subtracts the curr from the prev one
  # -> to show how quickly vegetation is growing or declining at each time step
  ndvi_rate_of_change = ndvi_columns.diff(axis=1)

  print(f"\nRate of change:\n{ndvi_rate_of_change}")

  # calculate the cumulative ndvi representing the overall vegetation productivity
  # -> add up all the prev values to the curr one
  ndvi_cumulative = ndvi_columns.cumsum(axis=1)

  print(f"\nCumulative NDVI:\n{ndvi_cumulative}")


"""
Area under the NDVI curve
"""
# to represent the total productivity or cumulative vegetation health during the entire growing season
def measure_auc(dataset):
  ndvi_columns = dataset[columns]
  auc_values = ndvi_columns.apply(lambda row: np.trapz(row), axis=1)
  print(f"Area under the NDVI curve (AUC):\n{auc_values}")