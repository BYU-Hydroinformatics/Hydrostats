[Documentation Editing Guide](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet "Github Markdown Cheatsheet")

# Contents

## Metrics:
+ [Mean Error](#hydrostatsme)
+
+

## Data Management:
+ [Merging Data](#hydrostatsdatamerge_data)
+ [Finding Seasonal Periods](#hydrostatsdataseasonal_period)
+ [Finding Daily Averages](#hydrostatsdatadaily_average)
+ Finding Daily Standard Error
+ Finding Monthly Averages
+ Finding Monthly Standard Error

## Visualization
+ [Hydrograph](#hydrostatsvisualplot)
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

### hydrostats.data.merge_data

#### class hydrostats.data.merge_data(predicted_file_path, recorded_file_path, column_names=['Simulated', 'Observed']): 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/data.py#L5 "Source Code")

#### Merge Data 
The merge data function takes two csv input files and merges them together based on their datetime index. 

| Parameters       |               |
| :-------------   |:------------- |
| predicted_file_path (Required Input) | A string type input that has the path to the predicted data file. |
| recorded_file_path (Required Input) | A string type input that has the path to the recorded data file. |
| column_names (Default=['Simulated', 'Observed']) | A list of two string type inputs specifying the column names of the two columns in the dataframe created.|

#### Example

```python
In  [1]: import pandas as pd
        
         df_merged = hd.merge_data(r'/path/to/predicted.csv/', r'/path/to/recorded.csv/')

         df_merged
Out [1]: 
```
### hydrostats.data.seasonal_period

#### class hydrostats.data.seasonal_period(merged_dataframe, daily_period, time_range=None, numpy=False): 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/data.py#L73 "Source Code")

#### Seasonal Period 
The merge data function takes a pandas dataframe with two columns of predicted data and observed data and returns a seasonal period of data based on user input. 

| Parameters       |               |
| :-------------   |:-------------|
| merged_dataframe (Required Input) | A pandas dataframe input that has two columns of predicted data and observed data with a dateime index. |
| daily_period (Required Input) | A list of strings that contains the seasonal period that the user wants to capture (e.g. ['05-01', '07-01']). |
| time_range (Default=None) | An optional input of a list of two string type options specifying a date-range for the seasonal period. (e.g. ['1980-01-01', '1987-12-31'])|
| Numpy (Default=False) | An optional input of boolean type. If specified as true the function will return two numpy arrays rather than a pandas dataframe.|
#### Example

```python
In  [1]: import hydrostats.data as hd
         import pandas as pd
         import numpy as np
         
         example_df = pd.DataFrame(np.random.rand(1000, 2), index=pd.date_range('1990-01-01', periods=1000), freq='1D')
       
         df_seasonal = hd.seasonal_period(example_df, daily_period=['05-01', '05-04'])
         
         df_seasonal
Out [1]:
```

### hydrostats.data.daily_average

#### class hydrostats.data.daily_average(merged_data): 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/data.py#L22 "Source Code")

#### Daily Average
The daily average function takes a pandas dataframe and calculates daily averages of the time series. If the frequency of the time series is not daily, then it will still take the daily averages of all of the values that day.  

| Parameters       |               |
| :-------------   |:-------------|
| merged_data (Required Input) | A pandas dataframe input that has two columns of predicted data and observed data with a dateime index. |

#### Example

```python
In  [1]: import hydrostats.data as hd
         import pandas as pd
         import numpy as np
         
         example_df = pd.DataFrame(np.random.rand(1000, 2), index=pd.date_range('1990-01-01', periods=1000), freq='1D')
       
         df_daily_avg = hd.daily_average(example_df)
         
         df_seasonal
Out [1]:
```

# Visualization

### hydrostats.visual.plot

#### class hydrostats.visual.plot(merged_data_df, legend=None, metrics=None, grid=False, title=None, force_x=None, labels=None, savefigure=None): 
[source](https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/visual.py#L10 "Source Code")

#### Mean Error (ME) 
The ME measures the difference between the simulated data and the observed data (Fisher, 1920).  For the mean error, a smaller number indicates a better fit to the original data. Note that if the error is in the form of random noise, the mean error will be very small, which can skew the accuracy of this metric. ME is cumulative and will be small even if there are large positive and negative errors that balance.  

| Parameters       |               |
| :-------------   |:-------------|
| merged_data_df (Required Input) | A dataframe with a datetime type index and floating point type numbers in the two columns. The columns must be Simulated Data and Observed Data going from left to right. |
| legend (Default=None) | Adds a Legend in the 'best' location determined by the software. The entries must be in the form of a list. (e.g. ['Simulated Data', 'Predicted Data']|
| metrics (Default=None)  | Adds Metrics to the left side of the Plot. Any Metric from the Hydrostats Library can be added to the plot as the name of the function. The entries must be in a list. (e.g. |['me', 'mae'] would plot the Mean Error and the Mean Absolute Error on the left side of the plot.| 
| grid (Default=False) | Takes a boolean type input and adds a grid to the plot. |
| title (Default=None) | Takes a string type input and adds a title to the hydrograph. |
| force_x (Default=None) | Takes a boolean type input. If True, the x-axis ticks will be staggered every 20 ticks. This is a useful feature when plotting daily averages. |
| labels (Default=None) | Takes a list of two string type objects and adds them as x-axis labels and y-axis labels, respectively.|
| savefigure (Default=None) | Takes a string type input and will save the plot the the specified path as a filetype specified by the user. | 

Available Filetypes with savefig: 
- Postscript (.ps) 
- Encapsulated Postscript (.eps)
- Portable Document Format (.pdf)
- PGF code for LaTeX (.pgf
- Portable Network Graphics (.png)
- Raw RGBA bitmap (.raw)
- Raw RGBA bitmap (.rgba)
- Scalable Vector Graphics (.svg) 
- Scalable Vector Graphics (.svgz)
- Joint Photographic Experts Group (.jpg, .jpeg)
- Tagged Image File Format (.tif, .tiff)




#### Example

```python
import hydrostats as hs
import numpy as np

sim = np.arange(10)
obs = np.random.rand(10)

hs.me(sim, obs)
```

