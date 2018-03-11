[Documentation Editing Guide](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet "Github Markdown Cheatsheet")

# Contents

## Metrics:
+ [Mean Error](#hydrostatsme)
+
+

## Data Management:
+ Merging Data
+ Finding Seasonal Periods
+ Finding Daily Averages
+ Finding Daily Standard Error
+ Finding Monthly Averages
+ Finding Monthly Standard Error

## Visualization
+ Hydrograph
+ Histogram
+ Scatter Plot
+ Quantile-Quantile Plot


# Metrics

### hydrostats.me

#### class hydrostats.me(forecasted_array, observed_array) 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/__init__.py#L19 "Source Code")

#### Mean Error (ME) 
The ME measures the difference between the simulated data and the observed data (Fisher, 1920).  For the mean error, a smaller number indicates a better fit to the original data. Note that if the error is in the form of random noise, the mean error will be very small, which can skew the accuracy of this metric. ME is cumulative and will be small even if there are large positive and negative errors that balance.  

| Parameters      |               |
| :-------------   |:-------------|
| forecasted_array| A 1D array of forecasted data from the time series. |
| observed_array| A 1D array of observed data from the time series.|

#### Example

```python
import hydrostats as hs
import numpy as np

sim = np.arange(10)
obs = np.random.rand(10)

hs.me(sim, obs)
```

# Data Management

### hydrostats.visual.plot

#### class hydrostats.visual.plot(merged_data_df, legend=None, metrics=None, grid=False, title=None, force_x=None, labels=None, savefigure=None): 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/visual.py#L10 "Source Code")

#### Mean Error (ME) 
The ME measures the difference between the simulated data and the observed data (Fisher, 1920).  For the mean error, a smaller number indicates a better fit to the original data. Note that if the error is in the form of random noise, the mean error will be very small, which can skew the accuracy of this metric. ME is cumulative and will be small even if there are large positive and negative errors that balance.  

| Parameters       |               |
| :-------------   |:-------------|
| merged_data_df   | A dataframe with a datetime type index and floating point type numbers in the two columns. The columns must be Simulated Data and Observed Data going from left to right. |
| observed_array   | A 1D array of observed data from the time series.|

#### Example

```python
import hydrostats as hs
import numpy as np

sim = np.arange(10)
obs = np.random.rand(10)

hs.me(sim, obs)
```

# Visualization


