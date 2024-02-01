import matplotlib.patches as mpatches
from seaborn import objects as so
import os
from pathlib import Path
# import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import binned_statistic

from retry import retry
# import geopandas as gpd
# import pandas as pd
# import dask.dataframe as dd
# import ee
# import geemap
# from matplotlib import pyplot as plt
# import seaborn as sns
# import pyogrio
# from tqdm import tqdm
from util import *
from runLAD_PLD import output_dir
## I/O
# modN = 300000
# analysis_dir = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/vtest/'
analysis_dir = os.path.join(output_dir, 'Zonal-hist')
# index_file = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
# ee_zones_pth = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
# 'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na1'
ee_zones_pths = [
    # 'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    # 'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    # 'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    # 'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN'
]
regions = [
    # 'SWOT_PLD_v103_beta_1simpl_40',
    # 'SWOT_PLD_v103_beta_1simpl_46',
    # 'SWOT_PLD_v103_beta_1simpl_52',
    # 'SWOT_PLD_v103_beta_1simpl_58',
    'SWOT_PLD_v103_beta_1simpl_64',
    'SWOT_PLD_v103_beta_1simpl_70',
]

ee_value_raster_pth = "JRC/GSW1_4/GlobalSurfaceWater"
nWorkers = 30
# crs_str = 'PROJCS["Lambert_Azimuthal_Equal_Area",GEOGCS["Unknown",DATUM["D_unknown",SPHEROID["Unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",45.5],PARAMETER["central_meridian",-114.125],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
crs_wkt = 'PROJCS["ProjWiz_Custom_Lambert_Azimuthal", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984",6378137.0,298.257223563]], PRIMEM["Greenwich",0.0], UNIT["Degree",0.0174532925199433]], PROJECTION["Lambert_Azimuthal_Equal_Area"], PARAMETER["False_Easting",0.0], PARAMETER["False_Northing",0.0], PARAMETER["Central_Meridian",0], PARAMETER["Latitude_Of_Origin",65], UNIT["Meter",1.0]]'

# name_lat = 'Pour_lat'
# name_lon = 'Pour_long'
name_lat = 'lat'
name_lon = 'lon'
lat_ranges = [
    # [40, 46],
    # [46, 52],
    # [52, 58],
    # [58, 64],
    [64, 70],
    [72, 78]
]
lon_ranges = [
    # [-180, 180],
    # [-180, 180],
    # [-180, 180],
    # [-180, 180],
    [-180, 180],
    [-180, 180]
]
# lat_range = [62, 64.5]  # for testing
# lon_range = [-105, -103]
step = 0.5
offset_lower = 0  # 0.25

## Geemap zonal histogram parameters (note: start small and only increase them if API is hitting memory limits and not returning a CSV file)
scale = 30  # None  # 30
tile_scale = 4  # 12  # 2

## I/O for reading csvs
id_var = 'lake_num'  # Hylak_id
area_var = 'Shape_Area'  # Lake_area # km2
# '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
lake_inventory_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta.gdb'
loadJoined = False

# Auto I/O
offset_upper = step

## Testing
# vect = ee.FeatureCollection("projects/sat-io/open-datasets/HydroLakes/lake_poly_v10").map(addMod)
# print(vect.filter("Hylak_id < 500").filter("Lake_area < 1000").size().getInfo())
# print('Number of features in chunk: ', vect.filter("Hylak_id < 1000").size())
# vect.first().get('mod50')
# vect.propertyNames()
# vect.first().propertyNames() # to actually print the result!
# vect.get('mod50')

## Test on single (Error: property 'element' is required means some filter returned zero. )
# getResult(3, 1)
# getResult(0, np.array([-104.25, 51.25]))

######################
#### Operations
######################
runGlakesByRegion(ee_zones_pths, lat_ranges, lon_ranges, step, analysis_dir, name_lat, name_lon,
                  offset_upper, offset_lower, crs_wkt, scale, tile_scale, ee_value_raster_pth, nWorkers, regions)

# loadJoined = True
# CombineProcessGlakes(analysis_dir, ee_zones_pths, loadJoined, id_var)
