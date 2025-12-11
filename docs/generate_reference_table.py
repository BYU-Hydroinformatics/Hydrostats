import numpy as np
import pypandoc as pdoc
from HydroErr.HydroErr import function_list, metric_abbr, metric_names

function_names = [i.__name__ for i in function_list]

# Make the lists into numpy object arrays
metric_names_array = np.array(metric_names)
metric_abbr_array = np.array(metric_abbr)
function_names_array = np.array(function_names)

# Sorting the metric names and the corresponding abbreviations and function names
sorting_indices = np.argsort(metric_names_array)

metric_names_array = metric_names_array[sorting_indices]
metric_abbr_array = metric_abbr_array[sorting_indices]
function_names_array = function_names_array[sorting_indices]

markdown_table = (
    "|Full Metric Name|Abbreviation|Function Name|\n|----------------|------------|-------------|\n"
)

for i in range(metric_names_array.shape[0]):
    table_row = f"|{metric_names_array[i]}|{metric_abbr_array[i]}|{function_names_array[i]}|\n"

    markdown_table += table_row

rst_table = pdoc.convert_text(markdown_table, "rst", "md")

intro_text = """Quick Reference Table
=====================

*****************************************************
Metrics, Abbreviations, and Functions Quick Reference
*****************************************************

This table contains a list of the metrics names, abbreviations, and the name of the functions that
are associated with them. It is a good reference when creating tables or plots to be able to see
what metrics are available for use and their abbreviation name.

"""

with open("ref_table.rst", "w") as outfile:
    outfile.write(intro_text)
    outfile.write(rst_table)
    outfile.write("\n")
