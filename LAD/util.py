
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
* Write test for HydroLAKES.
* Auto re-run when GEE downloads have error message
* Check for corrupted error csvs after each download instead of seperately in own function.
* Run in dask instead of using binned_statistic
'''

import matplotlib.patches as mpatches
from seaborn import objects as so
import os
from pathlib import Path
# import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import binned_statistic
import xarray as xr

from retry import retry
# import timeout_decorator
import geopandas as gpd
import pandas as pd
import dask.dataframe as dd
import ee
import geemap
from matplotlib import pyplot as plt
import seaborn as sns
import pyogrio
from warnings import warn
from tqdm import tqdm
try:  # if no internet connection
    ## Register with ee using high-valume (and high-latency) endpoint
    # NOT 'https://earthengine.googleapis.com'
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
except:
    warn(UserWarning("EE not initialized."))
    pass

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


# @timeout_decorator.timeout(6, use_signals=False)
@retry(tries=3, delay=2, backoff=10)
# Skip files that take too long to run. Re-run later wiwth different scale.
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
        out_dir, f'lake_zstats_Oc_Long{coords[0]}_Lat{coords[1]}.csv')

    ## Don't overwrite if starting again
    if os.path.exists(out_pth + '.txt'):
        return

    # check if an error message was downloaded instead and thus renders the file useless
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


def runLakesByRegion(ee_zones_pths, lat_ranges, lon_ranges, step, analysis_dir, name_lat, name_lon, offset_upper, offset_lower, crs_wkt, scale, tile_scale, ee_value_raster_pth, nWorkers, regions=None):
    '''Custom I/O operations to load mutliple PLD files in .gdb format, clipping by 40 degN latitude.
    Calls functions via GEE in parallel using geemap toolbox.'''
    for j, ee_zones_pth in enumerate(ee_zones_pths):
        lat_range, lon_range = lat_ranges[j], lon_ranges[j]
        if regions == None:
            region = os.path.basename(ee_zones_pth).split('/')[-1]
        else:
            region = regions[j]
        table_dir = os.path.join(analysis_dir, region, 'tables')
        tile_dir = os.path.join(analysis_dir, region, 'tiles')
        for dir in [analysis_dir, table_dir, tile_dir]:
            os.makedirs(dir, exist_ok=True)
        ## View expected number of results
        coord_list = getRequests(lat_range, lon_range, step)  # index_file

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


def cleanCSVs(analysis_dir):
    """
    Clean CSV files in a directory.

    This function walks through the specified directory and its subdirectories, and attempts to open and read each CSV file using pandas.
    If an error occurs while opening a file, the filename is printed.
    If the first line of a file starts with '{', it indicates that the file is not in a valid CSV format and it is deleted.

    Parameters:
    analysis_dir: str
        Root directory (with possible subdirs) to check

    Returns:
    None
    """
    for root, dirs, files in os.walk(analysis_dir):
        for file in files:
            if file.endswith('.csv'):
                try:
                    # Try opening the file using pandas
                    pth = os.path.join(root, file)
                    pd.read_csv(pth)
                except Exception as e:
                    # Print the filename if there is an error
                    with open(pth, 'r') as file:
                        first_line = file.readline()
                    if first_line.startswith('{'):  # 401 error in JSON format
                        os.remove(file.name)
                        print(
                            f"Deleted error file: {pth}")
                    else:
                        print(
                            f"Error opening file: {pth}")


def MakeUniqueIndex(df, idxs):
    '''Pandas can only merge on a single columns'''


def CombineProcessLakes(analysis_dir, lake_inventory_pth, ee_zones_pths, loadJoined, id_var, lat_min=40, join_how='left', lat_var='Lat', regions=None):
    '''
    CombineProcessLakes: Load and piece together with dask individual csv outputs from GEE zonal histogram, write out .gdb files with new binned Occurrence attributes. (START HERE if not running GEE part).

    Saves memory by loading in different regions at a time.
    Parameters
    ----------
    analysis_dir : str
        _description_
    lake_inventory_pth : str
        A priori lake inventory in OGR-readable geospatial format (e.g. .shp, .gdb)
    ee_zones_pths : str or array of str
        Directory containing gee output as csvs.
    loadJoined : bool
        Whether to bypass the merging process and load the final product
    id_var : str
        ID variable used in lake dataset.
    lat_min : int, optional
       only process north of this latitude, by default 40
    join_how : str, optional
        gpd join method, by default 'left'
    lat_var : str, optional
        Name of latitude variable, by default 'Lat'
    '''

    # latter argument suggested by dask error and it fixes it! # usecols=[id_var]
    gdf_join_binned_pth = os.path.join(
        analysis_dir, 'lake_zstats_Oc_binned.gdb')  # final merged output path
    if not loadJoined:
        gdfs = []  # init
        for j, ee_zones_pth in enumerate(ee_zones_pths):
            if regions is not None:
                assert len(ee_zones_pths) == len(
                    regions), "ee_zones_pths must have same length as regions."
                region = regions[j]
            else:
                region = os.path.basename(ee_zones_pth).split(
                    '/')[-1]  # assumes ee_zones_pths have unique names
            print(f'Loading region: {region}.')
            lake_inventory = gpd.read_file(lake_inventory_pth,
                                           engine='pyogrio', bbox=(-180, lat_min, 180, 80))
            table_dir = os.path.join(analysis_dir, region, 'tables')
            tile_dir = os.path.join(analysis_dir, region, 'tiles')
            print(f'Loading tiles...')
            ddf = dd.read_csv(f"{tile_dir}/*.csv", assume_missing=True,
                              on_bad_lines='skip', dtype={'system:index': 'object'})
            ## For testing:
            # ddf = dd.read_csv(f"{tile_dir}/*Lat45.0.csv", assume_missing=True,
            #                   on_bad_lines='skip', dtype={'system:index': 'object'})  # Testing

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
            print('Rebinning...')
            bStat = binned_statistic(
                oc_column_vals, values=df.iloc[:, oc_columns], statistic=np.nansum, bins=[0, 5, 50, 95, 100])
            bStat
            bin_labels = ['Oc_0_5', 'Oc_5_50', 'Oc_50_95', 'Oc_95_100']
            dfB = pd.DataFrame(bStat.statistic, columns=bin_labels) / pd.DataFrame(
                df.loc[:, 'Class_sum']).values * 100  # , index=df.index) # df binned
            dfB[id_var] = df[id_var]
            dfB['Class_sum'] = df.Class_sum
            if isinstance(id_var, list):
                for var in id_var:
                    dfB[var] = df[var]
            dfB = ensure_unique_ids(dfB, id_var)

            ## Filter columns
            cols_to_keep = df.columns[[('Class' in c) or (
                c in id_var) for c in df.columns]]

            ## Join files
            # gdf_join_full = lake_inventory.merge(df[cols_to_keep], left_on='Hylak_id',
            #                           right_on='Hylak_id', how='inner', validate='one_to_one')

            # Merge the PLD data with the dataframe 'df' based on the common attribute 'id_var'
            # gdf_join_full = lake_inventory.merge(df[cols_to_keep], on=id_var,
            #                                     how='inner', validate='one_to_one')
            gdf_join_binned = lake_inventory.merge(dfB, on=id_var,
                                                   how=join_how, validate='one_to_one')
            gdf_join_binned.query(f'{lat_var} > {lat_min}', inplace=True)

            ## Write out full shapefile (slowww...52 minutes, 3.4 GB [without pyogrio])
            # gdf_join_full_pth = os.path.join(analysis_dir, 'lake_zstats_Oc_full.shp')
            # gdf_join_full.to_file(gdf_join_full_pth, engine='pyogrio')

            # Save the merged data to a new geodatabase in the same location
            # gdf_join_full_pth = os.path.join(analysis_dir, 'lake_zstats_Oc_full.gdb')
            # gdf_join_full.to_file(
            #     gdf_join_full_pth, driver='OpenFileGDB', engine='pyogrio')

            gdf_join_binned_tmp_pth = os.path.join(
                table_dir, f"lake_zstats_Oc_binned_{region.replace('_1simpl','')}.shp")
            gdf_join_binned.to_file(gdf_join_binned_tmp_pth,
                                    engine='pyogrio')  # driver='OpenFileGDB',
            print(f'Saved region to: {gdf_join_binned_tmp_pth}')
            gdfs.append(gdf_join_binned)

        ## Combine
        gdf_join_binned = pd.concat(gdfs)  # re-use variable name
        del gdfs  # save mem

        ## Check for duplicates in combined file
        gdf_join_binned.drop_duplicates(
            subset=id_var, keep='first', inplace=True)

        ## Write out combined
        gdf_join_binned.crs = gdf_join_binned.crs
        try:
            gdf_join_binned.to_file(gdf_join_binned_pth,
                                    driver='OpenFileGDB', engine='pyogrio')
        except:
            gdf_join_binned.to_file(gdf_join_binned_pth.replace('.gdb', '.shp'),
                                    engine='pyogrio')
        print(f'Saved merged file to: {gdf_join_binned_pth}')

    ## Go straight to loading
    else:
        print('Loading existing joined zonal stats...')
        gdf_join_binned = gpd.read_file(
            gdf_join_binned_pth, engine='pyogrio', read_geometry=False)

    ## Filter in only lakes > 40 N (and no NaNs)
    nanfilter = np.isnan(gdf_join_binned['Oc_5_50'])
    print(f"Contains {np.sum(nanfilter)} Na\'s.")
    nanindex = gdf_join_binned[nanfilter].index

    ## Manually add Caspian Sea which is often too large to process in GEE: only runs for lakes with NaN
    input_dict = {'Class_sum': 82155, 'Oc_0_5': 0.05,
                  'Oc_5_50': 0.05, 'Oc_50_95': 10.0, 'Oc_95_100': 89.9}
    for key, value in input_dict.items():
        gdf_join_binned.loc[nanindex, key] = value

    means = np.average(gdf_join_binned[['Oc_0_5', 'Oc_5_50', 'Oc_50_95', 'Oc_95_100']],
                       weights=gdf_join_binned.Class_sum, axis=0)
    print(means)
    print(f"Mean double-counting: {means[:2].sum():0.3} %")
    pass


def parseYearsMonths(years=None, months=None):
    ''' Helper function that returns a list of years and months (as ints) based on the input format. Years are 4-digit numeric, and months are spelled out. Both can accept ranges or ','or'/'-sep lists.'''

    ## pre-parse months
    # years=years.replace('/', ',')

    years_list = []
    month_dict = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04',
        'May': '05', 'June': '06', 'July': '07', 'August': '08',
        'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }

    if years is not None:
        for item in years.split(','):
            if '-' in item:
                start, end = list(map(int, item.split('-')))
                years_list.extend([i for i in range(start, end + 1)])
            else:
                years_list.append(int(item.strip()))
    else:
        years_list = None

    if months is not None:
        months = months.replace('/', ',')  # pre-parse
        months_list = []
        for item in months.split(','):
            if '-' in item:
                start, end = item.split('-')
                start_month = int(month_dict[start.strip()])
                end_month = int(month_dict[end.strip()])
                months_list.extend(
                    [i for i in range(start_month, end_month + 1)])
            else:
                months_list.append(int(month_dict[item.strip()]))
    else:
        months_list = None

    ## Convert to int
    # years_list = list(map(int, years_list))
    # months_list = list(map(int, months_list))

    return years_list, months_list


def pullTemp(df, da, lat_var='LAT', long_var='LONG', year_var='YEAR.S', month_var='MONTH', var='lblt', year=None):
    '''
    Merges in ERA5 temperature to an array based on 'LAT' and 'LONG' fields.
    
    Parameters
    ----------
    df : pd.DataFrame 
        df with lat, long, year, month fields as named below
    lat : str
        Field name to use for latitude (called "latitude" in ERA5)
    long : str
        Field name to use for longitude (called "longitude" in ERA5)
    year : str
        Field name to use for year of observation (comes from "time" in ERA5)
    month : str
        Field namae to use for month of observation (comes from "time" in ERA5)
    da : xarray.DataSet
        ERA5 data
    var : str ('lblt)
        One of: 'lblt' 'skt', 'stl4', 'stl1', 't2m'
    
    Returns
    -------
    None

    Function appends new temp field onto dataset fields. If 'year' is given, 'year_var' and 'month_var' are ignored and all months are used for average.
    
    '''
    lt = df[lat_var]
    ln = df[long_var]
    if year is not None:  # manual year supplied
        yr_list = [year]
        mth_list = np.arange(1, 13)
    else:
        # Get latitude, longitude, year, and month from the row
        # yr_list = parseYears(df[year_var]) # in case a range or list of years
        # mth_list = parseTimes(df[month_var]) # in case a range of months
        yr_list, mth_list = parseYearsMonths(df[year_var], df[month_var])

    # Select the temperature data from ERA5 using the given coordinates and time
    # da_tmp = da[var].where(da['time.year'].isin(yr), drop=True).sel(latitude=lt, longitude=ln, method='nearest')

    temprs = da[var].sel(latitude=lt, longitude=ln, method='nearest').sel(
        time=(da['time.year'].isin(yr_list)) & (da['time.month'].isin(mth_list)))

    ## reduce
    tempr = temprs.mean(dim='time')

    return tempr.values


def AddReanalysisTemps(ds_pth, temps_pth, id_var, lat_var='lat', long_var='lon', tvar='stl1'):
    '''
    AddReanalysisTemps Adds temperatures to ds_pth from temps_pth and writes out to input directory as new shapefile.

    Parameters
    ----------
    ds_pth : str
        geospatial dataset
    temps_pth : str
        .nc dataset
    '''
    print('Loading files...')
    gdf_lakes = gpd.read_file(ds_pth,
                              engine='pyogrio', read_geometry=True)  # , columns=[id_var, lat_var, lon_var])
    da = xr.load_dataset(temps_pth)

    ## Fill NaNs in climate data (for coastal measurements) # TODO clean this up
    da_filled = da.interpolate_na(dim='longitude', method='nearest')
    da_sorted = da.sortby('latitude')
    da_filled_lat = da_sorted.interpolate_na(dim='latitude', method='nearest')
    da_filled = da_filled.combine_first(da_filled_lat)

    print('Adding temperatures...')
    temps = gdf_lakes.apply(lambda row: pullTemp(row, da_filled, lat_var=lat_var,
                                                 long_var=long_var, var=tvar, year=2022), axis=1)  # .astype('float')
    gdf_lakes[f'ERA5_{tvar}'] = temps.astype('float')

    ## Fill nans with mean
    gdf_lakes[f'ERA5_{tvar}'].fillna(
        gdf_lakes[f'ERA5_{tvar}'].mean(), inplace=True)

    ## Write out
    pth_out = ds_pth.replace('.shp', '_temp.shp')
    gdf_lakes.to_file(pth_out, engine='pyogrio')
    print(f'Saved temps file to: {pth_out}')
