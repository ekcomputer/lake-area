
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
def batchZonalHist(index, coords, name_lat, name_lon, offset_lower, offset_upper, crs, ee_zones_pth, ee_value_raster_pth, out_dir):
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
        out_dir, f'HL_zStats_Oc_Long{coords[0]}_Lat{coords[1]}.csv')

    ## Don't overwrite if starting again
    if os.path.exists(out_pth) or os.path.exists(out_pth + '.txt'):
        return

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
            scale=30,
            # default is 1, increase number to reduce chunking tile size (it won't affect results, but will take longer and use less mem)
            tile_scale=2
        )
        print(f'Done with group {index}: {coords}')
    else:
        print('No features within region filtered by group.')
        Path(out_pth + '.txt').touch()


def genStarmap(coord_list, name_lat, name_lon, offset_lower, offset_upper, crs_wkt, ee_zones_pth, ee_value_raster_pth, out_dir):
    '''Helper function to prepare a list with all the required arguments to run batchZonalHist() in parallel. See batchZonalHist for arguments docstring.'''
    data_for_starmap = [(i,
                        coord_list[i],
                        name_lat,
                        name_lon,
                        offset_lower,
                        offset_upper,
                        crs_wkt,
                        ee_zones_pth,
                        ee_value_raster_pth,
                        out_dir)
                        for i in range(len(coord_list))]
    return data_for_starmap

########### Apply functions via GEE calls in parallel

if __name__ == '__main__':
    ## I/O
    # modN = 300000
    analysis_dir = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/vtest/'
    index_file = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    # ee_zones_pth = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
    ee_zones_pth = 'projects/ee-ekyzivat/assets/Shapes/GLAKES/GLAKES_na1'
    ee_value_raster_pth = "JRC/GSW1_4/GlobalSurfaceWater"
    nWorkers = 18
    # crs_str = 'PROJCS["Lambert_Azimuthal_Equal_Area",GEOGCS["Unknown",DATUM["D_unknown",SPHEROID["Unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",45.5],PARAMETER["central_meridian",-114.125],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
    crs_wkt = 'PROJCS["ProjWiz_Custom_Lambert_Azimuthal", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984",6378137.0,298.257223563]], PRIMEM["Greenwich",0.0], UNIT["Degree",0.0174532925199433]], PROJECTION["Lambert_Azimuthal_Equal_Area"], PARAMETER["False_Easting",0.0], PARAMETER["False_Northing",0.0], PARAMETER["Central_Meridian",0], PARAMETER["Latitude_Of_Origin",65], UNIT["Meter",1.0]]'

    # name_lat = 'Pour_lat'
    # name_lon = 'Pour_long'
    name_lat = 'Lat'
    name_lon = 'Lon'
    # lat_range = [40, 84]
    # lon_range = [-180, 180]
    lat_range = [62, 64.5]  # for testing
    lon_range = [-105, -103]
    step = 0.5
    offset_lower = 0  # 0.25

    # Auto I/O
    table_dir = os.path.join(analysis_dir, 'tables')
    out_dir = os.path.join(analysis_dir, 'tiles')
    for dir in [analysis_dir, table_dir, out_dir]:
        os.makedirs(dir, exist_ok=True)
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

    ## View expected number of results
    coord_list = getRequests(lat_range, lon_range, step)  # index_file
    print(f'Number of items: {len(coord_list)}')

    ## Run function
    print(f'Sending request in {len(coord_list)} chunks...')

    # Prepare enumerate-like object for starmap, instead of  # pool.starmap(getResult, enumerate(coord_list))
    data_for_starmap = genStarmap(coord_list,
                                  name_lat,
                                  name_lon,
                                  offset_lower,
                                  offset_upper,
                                  crs_wkt,
                                  ee_zones_pth,
                                  ee_value_raster_pth,
                                  out_dir)

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

    pass
# ## ............................
# # %% [markdown]
# # ## Load and piece together

# # %%
# # Load files using dask
# # from https://mungingdata.com/pandas/read-multiple-csv-pandas-dataframe/
# tile_dir = os.path.join(analysis_dir, 'tiles')
# ddf = dd.read_csv(f"{tile_dir}/*.csv", assume_missing=True, on_bad_lines='skip', dtype={'system:index': 'object',
#                   'Lake_name': 'object'})  # latter argument suggested by dask error and it fixes it!

# # %%
# ## convert to pandas df
# df = ddf.compute()
# df = df.drop_duplicates(subset='Hylak_id').reset_index().drop('index', axis=1)
# df

# # %%
# ## Debugging LAD.py
# np.any(df.Lake_area == 0)

# # %%
# ## ensure df has unique Hylak_id keys
# df = df.drop_duplicates(subset='Hylak_id')

# ## ensure df has unique Hylak_id keys
# assert len(df) - len(df.drop_duplicates(subset='Hylak_id')) == 0

# # %%
# ## Save as excel as intermediate step
# df_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.csv.gz')
# df.to_csv(df_pth, compression='gzip')

# # %%
# ## START HERE if not running GEE part
# ## Load df
# df_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.csv.gz')
# df = pd.read_csv(df_pth)

# # %% [markdown]
# # ## Bin GSW in 4 bins

# # %%
# ## Mask in occurence columns and change values to int
# # occurrence columns positive mask. use map function, rather than for loop, for practice!
# oc_columns = list(map(lambda c: ('Class_' in c)
#                   and ('sum' not in c), df.columns))
# # all relevant occurance fields converted to ints, as a list
# oc_column_vals = list(
#     map(lambda c: int(c.replace('Class_', '')), df.columns[oc_columns]))
# # oc_column_vals

# # %%
# bStat = binned_statistic(
#     oc_column_vals, values=df.iloc[:, oc_columns], statistic=np.nansum, bins=[0, 5, 50, 95, 100])
# bStat

# # %%
# bin_labels = ['0-5', '5-50', '50-95', '95-100']
# dfB = pd.DataFrame(bStat.statistic, columns=bin_labels) / pd.DataFrame(
#     df.loc[:, 'Class_sum']).values * 100  # , index=df.index) # df binned
# dfB['Hylak_id'] = df.Hylak_id
# dfB['Class_sum'] = df.Class_sum
# dfB

# # %%
# ## ensure dfB has unique Hylak_id keys
# dfB = dfB.drop_duplicates(subset='Hylak_id')

# ## ensure dfB has unique Hylak_id keys
# len(dfB) - len(dfB.drop_duplicates(subset='Hylak_id'))

# # %% [markdown]
# # ## Load shapefile and join in GSW values (full and binned)

# # %%
# ## Load shapefile to join
# gdf = gpd.read_file('/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp',
#                     engine='pyogrio')  # bbox=(-180, 40, 180, 90)) # bbox can speed loading


# # %%
# ## Filter columns
# cols_to_keep = df.columns[[('Class' in c) or (
#     'Hylak_id' in c) for c in df.columns]]

# # %%

# ## Merge files
# gdf_join_full = gdf.merge(df[cols_to_keep], left_on='Hylak_id',
#                           right_on='Hylak_id', how='inner', validate='one_to_one')

# # %%
# ## view
# gdf_join_full.head(2)

# # %%
# ## Write out full shapefile (slowww...52 minutes, 3.4 GB [without pyogrio])
# gdf_join_full_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.shp')
# gdf_join_full.to_file(gdf_join_full_pth, engine='pyogrio')

# # %%
# gdf.columns

# # %%
# ## Merge binned file to bawld gdf (only keep a few original attributes)
# # ['Cell_ID', 'Long', 'Lat', 'Area_Pct', 'Shp_Area', 'WETSCAPE', 'geometry']
# gdf_join_binned = gdf.merge(
#     dfB, left_on='Hylak_id', right_on='Hylak_id', how='inner', validate='one_to_one')
# gdf_join_binned

# # %%
# gdf_join_binned.columns
# # gdf_join_binned.columns[-6:]
# # gdf_join_binned[[-6:]]
# # gdf_join_binned.iloc[:,[0, -6:]]
# gdf_join_binned[['Hylak_id', 'geometry', '0-5',
#                  '5-50', '50-95', '95-100', 'Class_sum']]

# # %%
# ## Write out binned shapefile (can join in remaining attributes later)
# colsKeep = ['Hylak_id', 'geometry', '0-5',
#             '5-50', '50-95', '95-100', 'Class_sum']
# gdf_join_binned_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_binned.shp')
# gdf_join_binned[colsKeep].to_file(gdf_join_binned_pth)

# # %% [markdown]
# # ## Histogram plots

# # %%
# ## load if necessary (previously defined vars)
# # print('Loading OC_full...')
# # gdf_join_full_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.shp')
# # gdf_join_full = pyogrio.read_dataframe(gdf_join_full_pth, use_arrow=True)

# print('Loading OC_binned...')
# gdf_join_binned_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_binned.shp')
# gdf_join_binned = pyogrio.read_dataframe(gdf_join_binned_pth, use_arrow=True)

# bin_labels = ['0-5', '5-50', '50-95', '95-100']


# # %%
# ## Preprocess to remove any nan's in important columns
# gdf_join_binnedF = gdf_join_binned.dropna(subset=bin_labels)  # filtered
# print(
#     f'Dropped {gdf_join_binned.shape[0] - gdf_join_binnedF.shape[0]} rows with nans.')

# ## Averaging method 1: Take weighted average
# try:
#     weightAvg = np.average(
#         gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Lake_area'], axis=0)
# except:
#     try:
#         weightAvg = np.average(
#             gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Shp_Area'], axis=0)
#     except:
#         gdf_join_binnedF = gdf_join_binnedF.merge(gdf[['Hylak_id', 'Lake_area']], left_on='Hylak_id',
#                                                   right_on='Hylak_id', how='left', validate='one_to_one')  # Add in HL lake area if not present
#         # weightAvg = np.average(gdf_join_binnedF[bin_labels], weights = gdf_join_binnedF['Class_sum'], axis=0) # If I was sloppy and didn't save HL area
#         weightAvg = np.average(
#             gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Lake_area'], axis=0)
# # weightAvg = np.average(gdf_join_binned[bin_labels], axis=0)

# weightAvg

# # %%
# # Add Area <50% Oc
# gdf_join_binnedF['Area_lt_50'] = (
#     gdf_join_binnedF['0-5'] + gdf_join_binnedF['5-50']) / 100 * gdf_join_binnedF.Lake_area  # Units: km2
# gdf_join_binnedF.head(3)

# # %%
# ## Averaging option B: Sum and then average
# dfS = gdf_join_binnedF[bin_labels] / 100 * pd.DataFrame(
#     gdf_join_binnedF.loc[:, 'Class_sum']).values  # convert percentages back to sums
# dfS['Hylak_id'] = gdf_join_binnedF.Hylak_id

# ## Add area bin
# # gdf_join_binnedF['area_bin'] = pd.cut(gdf_join_binnedF.Class_sum, area_bins, labels=area_bins_labels)

# ## Melt for plotting
# # 'area_bin' # melted data frame where Occurrence bins represent unweighted MEANS
# dfsM = dfS.melt(id_vars=['Hylak_id'], var_name='Occurrence bin')

# ## Get sums for normalizing second axis
# # areaSum = gdf_join_binnedF[bin_labels].sum()

# ## view
# dfS.head(2)

# # %%
# ## Reshape (melt) and plot as grouped bar plot (very slow to plot)
# "Within all of one occurrence bin, what was the contribution of LEV values"
# g = sns.catplot(dfsM,  # .iloc[:1000,:],
#                 hue='Occurrence bin', y='value', x='Occurrence bin', kind='bar', palette='cividis_r', errorbar=('ci', 95))
# # Weighted percentage of pixels within bin (%)
# g.set_axis_labels('Occurrence bin', 'Proportion')
# g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

# ## Add second y-scale
# # ax2 = g.ax.twinx()
# # ax2.set_yticklabels()

# # %% [markdown]
# # ## Stacked bar plot showing lake size

# # %%
# # 3 Very Hydrolakes minimum lake size
# assert not (np.any(gdf_join_binnedF.Lake_area < 0.1))
# assert np.all(gdf_join_binnedF['Area_lt_50'] <= gdf_join_binnedF.Lake_area)

# # %%
# ## Bin data by lake area
# area_bins = pd.IntervalIndex.from_breaks(
#     [0.1, 1, 10, 100, 1e3, 1e4, 1e5], closed='left')
# area_bins_labels = ['0.1-1', '1-10', '10-100',
#                     '100-1000', '1000-10000', '10000-100000']
# gdf_join_binnedF['area_bin'] = pd.cut(
#     gdf_join_binnedF.Lake_area, area_bins, right=False, labels=area_bins_labels)
# gdf_join_binnedF.head(3)

# # %%
# ## Melt for plotting
# # Melted data frame where occurrence categories are MEAN, not SUM
# dfsM2 = gdf_join_binnedF.drop(columns='geometry').melt(
#     id_vars=['Hylak_id', 'area_bin'], var_name='Occurrence bin')

# ## group for later on
# grouped = dfsM2.groupby(['area_bin', 'Occurrence bin']).mean().reset_index()

# ## view
# # easier to view than the variable grouped, due to og index
# table = dfsM2.groupby(['area_bin', 'Occurrence bin']).sum()
# table

# # %%
# ## Save this table
# table.to_csv(os.path.join(table_dir, 'breakdown-size-oc.csv'))

# # %%
# ## Plot lakes by double breakdown

# ## plot colors
# plot_colors = ['r', 'g', 'b', 'orange']

# ## Try stacked bar plots... problem is that I can't add averages, and I can't easily divide dataframes with sum columns and also categorical columns...
# # bar3 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100','50-95','5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[3]) # .query("`Occurrence bin` == '0-5'" #"@np.isin(`Occurrence bin`, ['0-5', '5-50'])"
# # bar2 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['50-95','5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[2])
# # bar1 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[1])
# # bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[0])

# ## Try again (I verified each group sums to 1), can also use barplot or catplot(..., kind='bar')
# bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])], x="area_bin",
#                    y="value", hue='Occurrence bin', errorbar=('ci', 95), color=plot_colors[0], palette='cividis_r')
# # bar0.set_axis_labels('Proportion (%)', 'Area ($km^2$) bin')
# bar0.axes.set_xlabel('Lake area ($km^2$) bin')
# bar0.axes.set_ylabel('Proportion (%)')
# bar0.axes.set_title('HydroLAKES breakdown')
# plt.show()

# # %%
# ## Actual stacked plot using new sns objects API.
# sns.set_palette('cividis_r')  # doesn't st color map... not sure why...
# g = so.Plot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])],
#             x="area_bin", y="value", color='Occurrence bin').add(so.Bar(), so.Agg(), so.Stack())
# # g. ax.set_xlabels(bin_labels)
# g.label(x='Area bin ($km^2$)', y='Percentage')
# # g.show()

# # %% [markdown]
# # ## Scrap

# # %%
# ## Plot
# ## Reshape (melt) and plot as grouped bar plot
# "Within all of one occurrence bin, what was the contribution of LEV values"
# g = sns.catplot(gdf_join_binned[['Hylak_id'] + bin_labels].melt(id_vars='Hylak_id', var_name='Occurrence bin'),
#                 hue='Occurrence bin', y='value', x='Occurrence bin', kind='bar', palette='cividis_r', errorbar=('ci', 95))
# g.set_axis_labels('', 'Unweighted percentage of pixels within bin (%)')
# g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

# # %%
# ## Weighted average histogram/barplot without conf intervals
# dfWA = pd.DataFrame([weightAvg, bin_labels], index=['value', 'bin']).T
# g = sns.catplot(dfWA, hue='bin', y='value', x='bin',
#                 kind='bar', palette='cividis_r')
# g.set_axis_labels('', 'Area-weighted percentage of pixels within bin (%)')
# g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

# # %%
# ## Weighted average histogram/barplot (alternate using MPL)

# plt.bar(x=np.arange(4), height=weightAvg)
# plt.ylabel('Unweighted percentage of pixels within bin (%)')


# # %%
# ## Now plot as stacked bar plot (from https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot)

# # set plot style: grey grid in the background:
# # sns.set(style="darkgrid")

# # set the figure size
# # plt.figure(figsize=(14, 14))

# ## plot colors
# plot_colors = ['r', 'g', 'b', 'orange']

# ## small dataset for testing
# dfsM2_sub = dfsM2  # .iloc[1::80,:]

# # top bar -> sum all values(smoker=No and smoker=Yes) to find y position of the bars
# total = dfsM2_sub.groupby('area_bin')['value'].mean().reset_index()

# # bar chart 1 -> top bars (group of 'smoker=No')
# # bar_total = sns.barplot(x="area_bin",  y="value", data=total, color=plot_colors[0])
# bar_total = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'], [
#                               '95-100', '50-95', '5-50', '0-5'])]

# # bottom bar ->  take only smoker=Yes values from the data
# bin1 = dfsM2_sub[dfsM2_sub['Occurrence bin'] == '0-5']
# bin2 = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'], ['5-50', '0-5'])]
# bin3 = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'],
#                          ['50-95', '5-50', '0-5'])]
# # bin4 = dfsM2_sub[dfsM2_sub['Occurrence bin']=='95-100'] # not needed

# # bar chart 2 -> bottom bars (group of 'smoker=Yes')
# # bar2 = sns.barplot(x="area_bin", y="value", data=bin1, estimator='mean', errorbar=None,  color=plot_colors[1])
# # bar3 = sns.barplot(x="area_bin", y="value", data=bin2, estimator='mean', errorbar=None,  color=plot_colors[2])
# # bar4 = sns.barplot(x="area_bin", y="value", data=bin3, estimator='mean', errorbar=None,  color=plot_colors[3])

# # simple way of computing remaining bars by addition
# total_bin1 = bin1.groupby('area_bin')['value'].mean().reset_index()
# total_bin2 = bin2.groupby('area_bin')['value'].mean().reset_index()
# total_bin3 = bin3.groupby('area_bin')['value'].mean().reset_index()

# # add bar plots for sub totals
# bar3 = sns.barplot(x="area_bin", y="value",
#                    data=total_bin3, color=plot_colors[2])
# bar2 = sns.barplot(x="area_bin", y="value",
#                    data=total_bin2, color=plot_colors[1])
# bar1 = sns.barplot(x="area_bin", y="value",
#                    data=total_bin1, color=plot_colors[0])

# # add legend
# bars = [mpatches.Patch(color=j, label=bin_labels[i])
#         for i, j in enumerate(plot_colors)]
# # top_bar = mpatches.Patch(color='darkblue', label=bin_labels[0])
# # bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
# plt.legend(handles=bars)

# # show the graph
# plt.show()

# # %%
# ## Now dubplicate unweighted mean
# bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])],
#                    x="Occurrence bin", y="value", errorbar=('ci', 95), color=plot_colors[0], palette='cividis_r')

# # %% [markdown]
# # ## Scrap functions

# # %%


# def getRequests():
#     """Generates a list of work items to be downloaded. Should be dquivalent to 'return modN', where modN is mod number.
#     """
#     ## Load vector dataset
#     vect = ee.FeatureCollection(
#         "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10").map(addMod)

#     # For testing: Filter  to reduce size of operation
#     # vectv = vect.filter("Pour_lat > 59.5").filter("Pour_lat < 59.6") #.filter("Long == -126.25")

#     ## Aggregate by Hylak_id mod
#     # return np.unique(vectF.aggregate_array('Country').getInfo()) # change to vect not vectF for real run
#     # change to vect not vectF for real run
#     return np.unique(vect.aggregate_array(modstr).getInfo())

# # %%


# def getRequests():
#     ''' shortcut function that doesn't take 2.5 minutes.'''
#     return range(modN)

# # %%


# def addMod(feature):
#     '''Adds a new mod[n] column to FeatureCollection'''
#     mod = modN  # defined at beginning
#     modComputed = ee.Number(feature.get('Hylak_id')
#                             ).mod(mod)  # ee.Number.parse(
#     return feature.set('mod' + str(mod), modComputed)  # .double()
