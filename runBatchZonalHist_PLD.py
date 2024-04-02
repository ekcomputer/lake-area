import pandas as pd
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
from LAD.util import CombineProcessLakes, AddReanalysisTemps, downloadERA5
from runLAD_PLD import output_dir

## I/O
# modN = 300000
# analysis_dir = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/vtest/'
analysis_dir = os.path.join(output_dir, 'Zonal-hist')
# index_file = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
# ee_zones_pth = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
# 'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na1'
ee_zones_pths = [
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN',
    'projects/ee-ekyzivat/assets/Shapes/SWOT_PLD_v103_beta_1simpl_40degN'
]
regions = [
    'SWOT_PLD_v103_beta_1simpl_40degN',
    'SWOT_PLD_v103_beta_1simpl_46degN',
    'SWOT_PLD_v103_beta_1simpl_52degN',
    'SWOT_PLD_v103_beta_1simpl_58degN',
    'SWOT_PLD_v103_beta_1simpl_64degN',
    'SWOT_PLD_v103_beta_1simpl_70degN',
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
    [40, 46],
    [46, 52],
    [52, 58],
    [58, 64],
    [64, 70],
    [70, 78]
]
lon_ranges = [
    [-180, 180],
    [-180, 180],
    [-180, 180],
    [-180, 180],
    [-180, 180],
    [-180, 180]
]
# lat_range = [62, 64.5]  # for testing
# lon_range = [-105, -103]
step = 0.5
offset_lower = 0  # 0.25

## Geemap zonal histogram parameters (note: start small and only increase them if API is hitting memory limits and not returning a CSV file)
scale = 360  # None  # 30
tile_scale = 8  # 12  # 2

## I/O for reading csvs
id_var = ['lat', 'lon', 'lake_num']  # 'lake_num'  # Hylak_id
area_var = 'Shape_Area'  # Lake_area # km2
# '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
# lake_inventory_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta.gdb'
lake_inventory_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/edk_out/SWOT_PLD_v103_beta_1simpl_40degN.shp'
loadJoined = False
cds_dir = '/Volumes/thebe/Ch4/ERA5/cds'
era5_output_name = 'ERA5_stl1_2022_global.nc'

# Auto I/O
offset_upper = step
era5_pth = os.path.join(cds_dir, era5_output_name
                        )
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
## Will likely need to iterate running runLakesByRegion and cleanCSVs, because geemap sometimes writes errors to a csv file instead of a proper csv
######################

# cleanCSVs(analysis_dir)

# runLakesByRegion(ee_zones_pths, lat_ranges, lon_ranges, step, analysis_dir, name_lat, name_lon,
#                   offset_upper, offset_lower, crs_wkt, scale, tile_scale, ee_value_raster_pth, nWorkers, regions)

# loadJoined = True
# CombineProcessLakes(analysis_dir, lake_inventory_pth,
#                     ee_zones_pths, loadJoined, id_var, join_how='right', lat_var='lat', regions=regions)  # right join because each subset dataset is identical and the joined ds varies based on lat

# downloadERA5(
#     cds_dir
#     2022,
#     'soil_temperature_level_1',
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#     [0.25, 0.25],
#     [78, -180, -56, 180],
#     output_name=era5_output_name
# )

## Add temperatures to reference emissions dataset (Johnson dataset)
# AddReanalysisTemps('/Volumes/thebe/Other/JohnsonGlobalMethane/edk_out/2022jg006793-sup-0002-data set si-s01_normalized.csv',
#                    era5_pth,
#                    lat_var='Latitude',
#                    long_var='Longitude',
#                    year_var='Obs. Year',
#                    month_var='Obs. Month',
#                    )

## Rosentreter methane dataset instead
AddReanalysisTemps('/Volumes/thebe/Other/Rosentreter2021/edk_out/Rosentreter_Aquatic_Ecosystems_lakes.csv',
                   era5_pth,
                   lat_var='lat',
                   long_var='long',
                   year_var=None,
                   month_var=None,
                   year=2022,
                   )

## Add temperatures to lakes database
# AddReanalysisTemps('/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta.gdb',  # '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/edk_out/CH4/output/Zonal-hist/lake_zstats_Oc_binned.shp', # '/Volumes/thebe/Ch4/ERA5/cds/temperatures.nc'
#                    era5_pth,  # '/Volumes/thebe/Ch4/ERA5/cds/temperatures.nc'
#                    fields_to_read=['Shape', 'lake_id', 'lake_num', 'lon', 'lat', 'ref_area', 'pekel_water_frac'],
#                    # extension='.gdb',
#                    year=2022,
#                    driver='OpenFileGDB',
#                    )
