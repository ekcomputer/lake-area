
'''__Batch run for zonal stats__ \
Tips from: https://gorelick.medium.com/fast-er-downloads-a2abd512aa26 \
Overlays Pekel GSW Occurrence values over HydroLAKES and computes zonal histogram for each lake in Google Earth Engine.

First, authenticate to ee using:
`earthengine authenticate`

TODO
* Remove original HL attributes before download
* Check that all features are present in downloads, after merging
* 0-pad "Class_n" in output
* Add kwd args for batchZonalHist and threadPoolExecutor
* add test for batchZonalHist using HydroLakes data
* Use dask for more steps
* Put last bits in functions
* Test on HydroLAKES.
'''

import matplotlib.patches as mpatches
from seaborn import objects as so
import os
from pathlib import Path
# import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import binned_statistic

from retry import retry
import geopandas as gpd
import pandas as pd
import dask.dataframe as dd
import ee
import geemap
from matplotlib import pyplot as plt
import seaborn as sns
import pyogrio
from tqdm import tqdm

## Register with ee using high-valume (and high-latency) endpoint
# NOT 'https://earthengine.googleapis.com'
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# def getRequests(index_file):
#     ''' Based on unique lat/long indexes in BAWLD'''

#     ## Load shapefile to join
#     index = gpd.read_file(index_file, engine='pyogrio')

#     ## For testing
#     index = index[:5]  # uncomment to test on only 5 features

#     ## For test run: filter only a few tiles
#     # gdf_bawld.query("(Lat > 59) and (Lat < 60) and (Long > -109) and (Long < -102)", inplace=True) # comment out

#     return index[['Long', 'Lat']].to_numpy()


def getRequests(lat_range, lon_range, step=0.5):
    ''' 
    Returns an N x 2 array of longitude, latitude pairs for all permutations of pairs from lat_range, lon_range.
    Arguments:
        lat_range: array-like
            [min lat, max lat]
        lon_range: array-like
            [min long, max long]
        step: float
            grid spacing
    Returns:
        coord_list: numpy.Array
    '''

    # Create a meshgrid of all possible latitude and longitude values
    lats, lons = np.meshgrid(
        np.arange(lat_range[0], lat_range[1], step), np.arange(lon_range[0], lon_range[1], step))

    # Reshape the arrays into a single array of latitude, longitude pairs
    coord_list = np.vstack([lons.ravel(), lats.ravel()]).T

    return coord_list

## testing
# foo = getRequests()
# for f in foo:
#     print(f)

# foo

# (tries=10, delay=1, backoff=2) # 7,1,3 causes max delay of 12 min, hopefully enough to clear "service unavailable errors."
# (tries=7, delay=1, backoff=3)


@retry(tries=3, delay=2, backoff=10)
def batchZonalHist(index, coords, name_lat, name_lon, offset_lower, offset_upper, crs, scale, tile_scale, ee_zones_pth, ee_value_raster_pth, out_dir):
    '''
    getResult _summary_

    Parameters
    ----------
    index : int
        _description_
    coords : list(float)
        _description_
    name_lat : str
        variable name for latitude in ee zones FeatureCollection
    name_lon : str
        variable name for longitude in ee zones FeatureCollection
    offset_lower : float
        subtract from values in lat/lon range to determine lower bound for each range. offset_lower + offset_upper should sum to the step size used to make the coords list, or else there will be gapas in the domain.
    offset_upper : float
        add from values in lat/lon range to determine upper bound for each range.
    crs : str
        crs WKT to define projection
    scale : float
        Passed to geemap.zonal_statistics_by_group(): A nominal scale in meters of the projection to work in. Use None to set default.
    tile_scale : float
        Passed to geemap.zonal_statistics_by_group(): A scaling factor used to reduce aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable computations that run out of memory with the default. Use 1.0 for default.
    ee_zones_pth : str
        GEE zones FeatureCollection path 
    ee_value_raster_pth : str
        _description_
    out_dir : str
        _description_
    '''
    """
    Handle the HTTP requests to download one result. index is python index and long is longitude, used for aggregation.
    index is placeholder
    group is an object that represents a unique value within a grouping (e.g. country name, grid cell longitude), and is not related to "group" in function geemap.zonal_statistics_by_group
    """
    ''' TODO: for real, filter to only Arctic X, change scale and tile scale X, change load gdf BB'''

    ## I/O
    out_pth = os.path.join(
        out_dir, f'GL_zStats_Oc_Long{coords[0]}_Lat{coords[1]}.csv')

    ## Don't overwrite if starting again
    if os.path.exists(out_pth + '.txt'):
        return
    # check if an error message was downloaded instead and thus renders the @retry useless
    if os.path.exists(out_pth):
        with open(out_pth, 'r') as file:
            first_line = file.readline()
            if first_line.startswith('{'):
                pass  # proceed
            else:
                return  # skip file

    ## Load vect and compute mod of ID variable to use for grouping, filtering to high latitudes
    # .filter("Pour_lat > 45.0") #.map(addMod)
    vect = ee.FeatureCollection(ee_zones_pth)  # ee_zones_pth_input

    # For testing: Filter  to reduce size of operation
    # vectF = vectF.filter("Pour_lat > 59.55").filter("Pour_lat < 59.56") #.filter("Long == -126.25")
    # vect = vect.filter("Hylak_id < 500").filter("Lake_area < 1000")

    ## Load GSW
    gsw = ee.Image(ee_value_raster_pth)
    occurrence = gsw.select('occurrence').unmask()

    ## Filter based on bawld cell geometry (note: cells are unequal area)
    # vectF = vect.filter(ee.Filter.eq(modstr, group))
    # groupEE = [ee.Number.float(group[0]) , ee.Number.float(group[1])] # list(map(ee.Number.float, group)) # convert to server object
    vectF = vect.filter(ee.Filter.And(ee.Filter.expression(f"({name_lon} > {coords[0]-offset_lower}) && ({name_lon} <= {coords[0]+offset_upper})"),
                                      ee.Filter.And(ee.Filter.expression(f"({name_lat} > {coords[1]-offset_lower}) && ({name_lat} <= {coords[1]+offset_upper})"))))
    nFeats = vectF.size().getInfo()
    print(f'Number of features in chunk: {nFeats}')
    # print(vect.size())
    # print(vectF.size())
    # statistics_type can be either 'SUM' or 'PERCENTAGE'
    # denominator can be used to convert square meters to other areal units, such as square kilometers
    if nFeats != 0:
        geemap.zonal_statistics_by_group(
            occurrence,
            vectF,
            out_pth,
            statistics_type='SUM',
            denominator=1000000,
            decimal_places=3,
            crs=crs,
            # meters, specifiy to compute at native res (default would be 300m)
            scale=scale,
            # default is 1, increase number to reduce chunking tile size (it won't affect results, but will take longer and use less mem)
            tile_scale=tile_scale
        )
        print(f'Done with group {index}: {coords}')
    else:
        print('No features within region filtered by group.')
        Path(out_pth + '.txt').touch()


def genStarmap(coord_list, name_lat, name_lon, offset_lower, offset_upper, crs_wkt, scale, tile_scale, ee_zones_pth, ee_value_raster_pth, out_dir):
    '''Helper function to prepare a list with all the required arguments to run batchZonalHist() in parallel. See batchZonalHist for arguments docstring.'''
    data_for_starmap = [(i,
                        coord_list[i],
                        name_lat,
                        name_lon,
                        offset_lower,
                        offset_upper,
                        crs_wkt,
                        scale,
                        tile_scale,
                        ee_zones_pth,
                        ee_value_raster_pth,
                        out_dir)
                        for i in range(len(coord_list))]
    return data_for_starmap


def ensure_unique_ids(df: pd.DataFrame, id_var: str) -> pd.DataFrame:
    """
    Ensure that the DataFrame has unique values for the specified ID variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check and modify.
    id_var : str
        The name of the column in `df` to be checked for unique values.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame, potentially with duplicates removed based on `id_var`.

    Raises
    ------
    AssertionError
        If duplicate values are found for `id_var`.
    
    Example usage:
    ------
    df = pd.DataFrame(...)
    df = ensure_unique_ids(df, 'your_id_column_name')

    """
    len0 = len(df)
    len1 = len(df.drop_duplicates(subset=id_var))
    dups_exist = len0 - len1 != 0

    if dups_exist:
        df = df.drop_duplicates(subset=id_var)
        print(
            f'Found {len0 - len1} duplicate id values and removed duplicates arbitrarily.')

    return df


def runGlakesByRegion():
    '''Custom I/O operations to load four GLAKES files in .gdb format, clipping by 40 degN latitude.
    Calls functions via GEE in parallel using geemap toolbox.'''
    for j, ee_zones_pth in enumerate(ee_zones_pths):
        lat_range, lon_range = lat_ranges[j], lon_ranges[j]
        region = os.path.basename(ee_zones_pth).split('/')[-1]
        table_dir = os.path.join(analysis_dir, region, 'tables')
        tile_dir = os.path.join(analysis_dir, region, 'tiles')
        for dir in [analysis_dir, table_dir, tile_dir]:
            os.makedirs(dir, exist_ok=True)
        ## View expected number of results
        coord_list = getRequests(lat_range, lon_range, step)  # index_file
        print(f'Number of items: {len(coord_list)}')

        ## Run function
        print(
            f'Sending request in {len(coord_list)} chunks...\n----------------------------------\n')

        # Prepare enumerate-like object for starmap, instead of  # pool.starmap(getResult, enumerate(coord_list))
        data_for_starmap = genStarmap(coord_list,
                                      name_lat,
                                      name_lon,
                                      offset_lower,
                                      offset_upper,
                                      crs_wkt,
                                      scale,
                                      tile_scale,
                                      ee_zones_pth,
                                      ee_value_raster_pth,
                                      tile_dir)

        ## Multiprocessing
        # pool = multiprocessing.Pool(nWorkers)
        # pool.starmap(batchZonalHist, data_for_starmap)
        # pool.close()
        # pool.join()

        ## Multithreading
        # Could also use ProcessPoolExecutor for multiprocessing
        with ThreadPoolExecutor(max_workers=nWorkers) as executor:
            # Submit tasks with keyword arguments
            # futures = [executor.submit(batchZonalHist, **args)
            #            for args in data_for_starmap]
            # Submit tasks with standard arguments
            # futures = executor.submit(batchZonalHist, data_for_starmap)
            futures = [executor.submit(batchZonalHist, *args)
                       for args in data_for_starmap]

            # Wrap as_completed with tqdm for a progress bar
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass  # Each iteration represents one completed task

        print(
            f'\nFinished region: {region}.\n---------------------------------')
    print('\nFinished all regions.\n---------------------------------')


def CombineProcessGlakes():
    '''Load and piece together with dask, write out .gdb files with new binned Occurrence attributes. (START HERE if not running GEE part).'''
    # latter argument suggested by dask error and it fixes it! # usecols=[id_var]
    gdf_join_binned_pth = os.path.join(
        analysis_dir, 'GL_zStats_Oc_binned.gdb')
    if not loadJoined:
        gdfs = []  # init

        ## Load shapefile to join
        # lake_inventory = gpd.read_file(lake_inventory_pth,
        #                                engine='pyogrio')  # bbox=(-180, 40, 180, 90)) # bbox can speed loading

        for j, ee_zones_pth in enumerate(ee_zones_pths):
            region = os.path.basename(ee_zones_pth).split('/')[-1]
            print(f'Loading region: {region}.')

            lake_inventory = gpd.read_file(f"/Volumes/metis/Datasets/GLAKES/GLAKES/GLAKES_{region.replace('GLAKES_','')}.shp",
                                           engine='pyogrio', bbox=(-180, 40, 180, 80))
            table_dir = os.path.join(analysis_dir, region, 'tables')
            tile_dir = os.path.join(analysis_dir, region, 'tiles')
            ddf = dd.read_csv(f"{tile_dir}/*.csv", assume_missing=True,
                              on_bad_lines='skip', dtype={'system:index': 'object'})

            ## convert to pandas df
            df = ddf.compute()
            df = df.drop_duplicates(subset=id_var).reset_index().drop(
                ['index', 'system:index'], axis=1)

            ## ensure df has unique Hylak_id keys
            df = ensure_unique_ids(df, id_var)

            ## Mask in occurence columns and change values to int
            # occurrence columns positive mask. use map function, rather than for loop, for practice!
            oc_columns = list(map(lambda c: ('Class_' in c)
                                  and ('sum' not in c), df.columns))
            # all relevant occurance fields converted to ints, as a list
            oc_column_vals = list(
                map(lambda c: int(c.replace('Class_', '')), df.columns[oc_columns]))
            # oc_column_vals

            ## Bin occurrence
            bStat = binned_statistic(
                oc_column_vals, values=df.iloc[:, oc_columns], statistic=np.nansum, bins=[0, 5, 50, 95, 100])
            bStat
            bin_labels = ['Oc_0_5', 'Oc_5_50', 'Oc_50_95', 'Oc_95_100']
            dfB = pd.DataFrame(bStat.statistic, columns=bin_labels) / pd.DataFrame(
                df.loc[:, 'Class_sum']).values * 100  # , index=df.index) # df binned
            dfB[id_var] = df[id_var]
            dfB['Class_sum'] = df.Class_sum
            dfB = ensure_unique_ids(dfB, id_var)

            ## Filter columns
            cols_to_keep = df.columns[[('Class' in c) or (
                id_var in c) for c in df.columns]]

            ## Join files
            # gdf_join_full = lake_inventory.merge(df[cols_to_keep], left_on='Hylak_id',
            #                           right_on='Hylak_id', how='inner', validate='one_to_one')

            # Merge the GLAKES data with the dataframe 'df' based on the common attribute 'id_var'
            # gdf_join_full = lake_inventory.merge(df[cols_to_keep], on=id_var,
            #                                     how='inner', validate='one_to_one')
            gdf_join_binned = lake_inventory.merge(dfB, on=id_var,
                                                   how='left', validate='one_to_one')
            gdf_join_binned.query('Lat > 40', inplace=True)

            ## Write out full shapefile (slowww...52 minutes, 3.4 GB [without pyogrio])
            # gdf_join_full_pth = os.path.join(analysis_dir, 'GL_zStats_Oc_full.shp')
            # gdf_join_full.to_file(gdf_join_full_pth, engine='pyogrio')

            # Save the merged data to a new geodatabase in the same location
            # gdf_join_full_pth = os.path.join(analysis_dir, 'GL_zStats_Oc_full.gdb')
            # gdf_join_full.to_file(
            #     gdf_join_full_pth, driver='OpenFileGDB', engine='pyogrio')

            gdf_join_binned_pth = os.path.join(
                table_dir, f"GL_zStats_Oc_binned_{region.replace('GLAKES_','')}.gdb")
            gdf_join_binned.to_file(gdf_join_binned_pth,
                                    driver='OpenFileGDB', engine='pyogrio')
            gdfs.append(gdf_join_binned)

        # write out combined
        gdf_join_binned = pd.concat(gdfs)
        del gdfs  # save mem
        gdf_join_binned.crs = gdf_join_binned.crs
        gdf_join_binned.to_file(gdf_join_binned_pth,
                                driver='OpenFileGDB', engine='pyogrio')

    ## Go straight to loading # HERE
    if loadJoined:
        gdf_join_binned = gpd.read_file(
            gdf_join_binned_pth, engine='pyogrio', read_geometry=False)

    ## Filter in only lakes > 40 N
    nanfilter = np.isnan(gdf_join_binned['Oc_5_50'])
    print(f"Contains {np.sum(nanfilter)} Na\'s.")
    nanindex = gdf_join_binned[nanfilter].index
    input_dict = {'Class_sum': 82155, 'Oc_0_5': 0.05,
                  'Oc_5_50': 0.05, 'Oc_50_95': 10.0, 'Oc_95_100': 89.9}
    for key, value in input_dict.items():
        gdf_join_binned.loc[nanindex, key] = value

    means = np.average(gdf_join_binned[['Oc_0_5', 'Oc_5_50', 'Oc_50_95', 'Oc_95_100']],
                       weights=gdf_join_binned.Class_sum, axis=0)
    print(f"Mean double-counting: {means[:2].sum():0.3} %")
    pass


if __name__ == '__main__':
    ## I/O
    # modN = 300000
    # analysis_dir = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/vtest/'
    analysis_dir = '/Volumes/metis/Datasets/Liu_aq_veg/Zonal-hist'
    # index_file = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    # ee_zones_pth = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
    ee_zones_pths = ['projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na2',
                     'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_as',
                     'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_eu',
                     'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na1']  # 'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na1'
    ee_value_raster_pth = "JRC/GSW1_4/GlobalSurfaceWater"
    nWorkers = 30
    # crs_str = 'PROJCS["Lambert_Azimuthal_Equal_Area",GEOGCS["Unknown",DATUM["D_unknown",SPHEROID["Unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",45.5],PARAMETER["central_meridian",-114.125],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
    crs_wkt = 'PROJCS["ProjWiz_Custom_Lambert_Azimuthal", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984",6378137.0,298.257223563]], PRIMEM["Greenwich",0.0], UNIT["Degree",0.0174532925199433]], PROJECTION["Lambert_Azimuthal_Equal_Area"], PARAMETER["False_Easting",0.0], PARAMETER["False_Northing",0.0], PARAMETER["Central_Meridian",0], PARAMETER["Latitude_Of_Origin",65], UNIT["Meter",1.0]]'

    # name_lat = 'Pour_lat'
    # name_lon = 'Pour_long'
    name_lat = 'Lat'
    name_lon = 'Lon'
    lat_ranges = [[40, 78.0], [40, 78.0], [40.0, 77.0], [40, 78.0]]
    lon_ranges = [[-180, 180], [-180, 180], [-24.5, 69.0], [-180, 180]]
    # lat_range = [62, 64.5]  # for testing
    # lon_range = [-105, -103]
    step = 0.5
    offset_lower = 0  # 0.25

    ## Geemap zonal histogram parameters (note: start small and only increase them if API is hitting memory limits and not returning a CSV file)
    scale = None  # 30
    tile_scale = 12  # 2

    ## I/O for reading csvs
    id_var = 'Lake_id'  # Hylak_id
    area_var = 'Area_PW'  # Lake_area # km2
    # '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
    lake_inventory_pth = '/Volumes/metis/Datasets/GLAKES/GLAKES.gdb'
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
    # runGlakesByRegion()

    CombineProcessGlakes()
