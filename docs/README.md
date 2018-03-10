# Welcome to Hydrostats

[Documentation Editing Guide](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet "Github Markdown Cheatsheet")

# Contents

## Metrics:
[mean_error](###hydrostats.me)

### hydrostats.me

#### class hydrostats.me(forecasted_array, observed_array) [source] (https://github.com/waderoberts123/Hydrostats/blob/master/hydrostats/__init__.py#L19 "Source Code")

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
