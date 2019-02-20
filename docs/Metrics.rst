Metrics of Hydrological Skill (hydrostats.metrics)
==================================================

The metrics in Hydrostats are available through the HydroErr package. Below is a link to the HydroErr
documentation page that lists all of the metrics contained in the package.

Link
^^^^
https://byu-hydroinformatics.github.io/HydroErr/list_of_metrics.html

Examples
^^^^^^^^
The metrics can be imported into your scripts as part of the metrics module. An example is provided below showing how to
use the metrics included in the package from HydroErr (if you would prefer to only use the Hydrostats package and not
just import the HydroErr package).

.. code-block:: python
    :emphasize-lines: 1

    import hydrostats.metrics as hm
    import numpy as np

    sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])

    mean_error = hm.me(sim, obs)
    print(mean_error)
