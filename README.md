# Lake Area Distribution (LAD): A Python package to extrapolate the lake-area distribution to small lakes and calculate their vegetation coverage and methane emissions

## Basic Usage

This package defines two classes: `LAD` (Lake-area distribution) and `BinnedLAD`(Binned LAD). Both are based on the `pandas.DataFrame` class but have additional attributes and methods. 

An `LAD` has entries for individual lakes and columns for area, temperature, vegetation, wetland double-counting, methane emissions, and an index for traceability to the source dataset. It is typically loaded from an ESRI shapefile as follows:

```python
#...
```

A `BinnedLAD` has entries for size bins that would comprise the histogram of lake areas. As such, it does not include individual lakes, because its primary purpose is to estimate, through extrapolation, the areas of bins below a measurement threshold. It contains many of the same attributes as a `LAD`, but only aggregated to bins. The class is typically created from an existing `LAD` object as follows:

```python
#...
```

### Loading sample data

I have included some small, sample datasets, which can be used to ensure your installation is working and to demo its functionality.

...

```python
#...
```

## Requirements

To use the jupyter notebooks, you will also need to install the requirements in requirements_optional.txt

## Installation

I suggest using the `mamba` package manager (install into base environment). You can use simple `conda` commands interchangably, if you do not want to install new software. First, create a new conda/mamba environment and activate it. Then:

```shell
$ mamba install --file requirements.txt
```

Access the classes and functions by importing the package. Enter the `LAD` project directory and run:

```python
from LAD import LAD, BinnedLAD
```

To run other functions, you can import them individually or import everything with `from LAD import *`.

### Optional: Use pip to install LAD to your environment 

Once you have installed your dependicies with `mamba`, you can use pip to install `LAD`. This will ensure `LAD` is on your path no matter which directory you call it from. (Using pip to install all dependencies will likely not work due to dependency conflict among geospatial packages like geopandas).

```shell
pip install -e .
```

If you want to install the package in editable mode (meaning you can make changes to the code and see the changes immediately without reinstalling), you can use the -e flag above. This will help you debug any inevitable errors you catch.
