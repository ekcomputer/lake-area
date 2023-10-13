# Lake Area Distribution (LAD): A Python package to extrapolate the lake-area distribution to small lakes and calculate their vegetation coverage and methane emissions

## Basic Usage


## Requirements

To use the jupyter notebooks, you will also need to install the requirements in requirements_optional.txt

## Installation

I suggest using the `mamba` package manager (install into base environment). First, create a new conda/mamba environment and activate it. Then:

```shell
mamba install --file requirements.txt
```

Access the classes and functions by importing the package. Enter the `LAD` directory and run:

```python
$ from LAD import LAD, BinnedLAD
```

### Optional: Use pip to install LAD to your environment 

Using pip to install all dependencies will likely not work due to dependency conflict among geospatial packages like geopandas. However, if you want to ensure LAD is on your path, no matter which directory you call it from, run:

```shell
pip install -e .
```

If you want to install the package in editable mode (meaning you can make changes to the code and see the changes immediately without reinstalling), you can use the -e flag above. This will help you debug any inevitable errors you catch.
