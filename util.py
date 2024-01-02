# %% [markdown]
# __Batch run for zonal stats__ \
# Tips from: https://gorelick.medium.com/fast-er-downloads-a2abd512aa26 \
# Overlays Pekel GSW Occurrence values over HydroLAKES and computes zonal histogram for each lake in Google Earth Engine.\
# \
# TODO
# * Remove original HL attributes before download from EE

# %% [markdown]
# ## I/O

# %%
import matplotlib.patches as mpatches
from seaborn import objects as so
import os
from pathlib import Path
import multiprocessing
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


# %%
## I/O
modN = 300000
analysis_dir = '/mnt/g/Ch4/GSW_zonal_stats/HL/v4/'
table_dir = os.path.join(analysis_dir, 'tables')
for dir in [analysis_dir, table_dir]:
    os.makedirs(dir, exist_ok=True)

## Derived
# modstr = 'mod'+str(modN)

# %%
## Register with ee using high-valume (and high-latency) endpoint
# NOT 'https://earthengine.googleapis.com'
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# %% [markdown]
# ## Functions

# %%


def getRequests():
    ''' Based on unique lat/long indexes in BAWLD'''

    ## Load shapefile to join
    gdf_bawld = gpd.read_file(
        '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip')

    ## For test run: filter only a few tiles
    # gdf_bawld.query("(Lat > 59) and (Lat < 60) and (Long > -109) and (Long < -102)", inplace=True) # comment out

    return gdf_bawld[['Long', 'Lat']].to_numpy()

## testing
# foo = getRequests()
# for f in foo:
#     print(f)

# foo

# %%


# (tries=10, delay=1, backoff=2) # 7,1,3 causes max delay of 12 min, hopefully enough to clear "service unavailable errors."
@retry(tries=7, delay=1, backoff=3)
def getResult(index, group):
    """
    Handle the HTTP requests to download one result. index is python index and long is longitude, used for aggregation.
    index is placeholder
    group is an object that represents a unique value within a grouping (e.g. country name, grid cell longitude), and is not related to "group" in function geemap.zonal_statistics_by_group
    """
    ''' TODO: for real, filter to only Arctic X, change scale and tile scale X, change load gdf BB'''
    ## I/O
    out_dir = os.path.join(analysis_dir, 'tiles')
    out_pth = os.path.join(
        out_dir, f'HL_zStats_Oc_Long{group[0]}_Lat{group[1]}.csv')

    ## Don't overwrite if starting again
    if os.path.exists(out_pth) or os.path.exists(out_pth + '.txt'):
        return

    ## CRS (ist there a smarter way to do this?)
    crs = 'PROJCS["Lambert_Azimuthal_Equal_Area",GEOGCS["Unknown",DATUM["D_unknown",SPHEROID["Unknown",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",45.5],PARAMETER["central_meridian",-114.125],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
    # crs = 'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",90],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","6931"]]'

    ## Load vect and compute mod of ID variable to use for grouping, filtering to high latitudes
    # .filter("Pour_lat > 45.0") #.map(addMod)
    vect = ee.FeatureCollection(
        "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10")

    # For testing: Filter  to reduce size of operation
    # vectF = vectF.filter("Pour_lat > 59.55").filter("Pour_lat < 59.56") #.filter("Long == -126.25")
    # vect = vect.filter("Hylak_id < 500").filter("Lake_area < 1000")

    ## Load GSW
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occurrence = gsw.select('occurrence').unmask()

    ## Filter based on bawld cell geometry (note: cells are unequal area)
    # vectF = vect.filter(ee.Filter.eq(modstr, group))
    # groupEE = [ee.Number.float(group[0]) , ee.Number.float(group[1])] # list(map(ee.Number.float, group)) # convert to server object
    vectF = vect.filter(ee.Filter.And(ee.Filter.expression(f"(Pour_long > {group[0]-0.25}) && (Pour_long <= {group[0]+0.25})"),
                                      ee.Filter.And(ee.Filter.expression(f"(Pour_lat > {group[1]-0.25}) && (Pour_lat <= {group[1]+0.25})"))))
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
        print("Done with group: ", group)
    else:
        print('No features within region filtered by group.')
        Path(out_pth + '.txt').touch()

# %% [markdown]
# ## Apply functions via GEE calls in parallel

# %%
## Testing
# vect = ee.FeatureCollection("projects/sat-io/open-datasets/HydroLakes/lake_poly_v10").map(addMod)
# print(vect.filter("Hylak_id < 500").filter("Lake_area < 1000").size().getInfo())
# print('Number of features in chunk: ', vect.filter("Hylak_id < 1000").size())
# vect.first().get('mod50')
# vect.propertyNames()
# vect.first().propertyNames() # to actually print the result!
# vect.get('mod50')

# %%
## Test on single (Error: property 'element' is required means some filter returned zero. )
# getResult(3, 1)
# getResult(0, np.array([-104.25, 51.25]))


# %%
## View expected number of results
items = getRequests()
# print(f'Number of items: {len(items)}')

# %%
print(f'Number of items: {len(items)}')

# %%
## Run function
# items = getRequests() # a list whose length is the number of groups to use for parallelizing # np.arange(modN) #
print(f'Sending request in {len(items)} chunks...')
pool = multiprocessing.Pool(30)  # 25
pool.starmap(getResult, enumerate(items))
pool.close()
pool.join()

# %% [markdown]
# ## Load and piece together

# %%
# Load files using dask
# from https://mungingdata.com/pandas/read-multiple-csv-pandas-dataframe/
tile_dir = os.path.join(analysis_dir, 'tiles')
ddf = dd.read_csv(f"{tile_dir}/*.csv", assume_missing=True, on_bad_lines='skip', dtype={'system:index': 'object',
                  'Lake_name': 'object'})  # latter argument suggested by dask error and it fixes it!

# %%
## convert to pandas df
df = ddf.compute()
df = df.drop_duplicates(subset='Hylak_id').reset_index().drop('index', axis=1)
df

# %%
## Debugging LAD.py
np.any(df.Lake_area == 0)

# %%
## ensure df has unique Hylak_id keys
df = df.drop_duplicates(subset='Hylak_id')

## ensure df has unique Hylak_id keys
assert len(df) - len(df.drop_duplicates(subset='Hylak_id')) == 0

# %%
## Save as excel as intermediate step
df_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.csv.gz')
df.to_csv(df_pth, compression='gzip')

# %%
## START HERE if not running GEE part
## Load df
df_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.csv.gz')
df = pd.read_csv(df_pth)

# %% [markdown]
# ## Bin GSW in 4 bins

# %%
## Mask in occurence columns and change values to int
# occurrence columns positive mask. use map function, rather than for loop, for practice!
oc_columns = list(map(lambda c: ('Class_' in c)
                  and ('sum' not in c), df.columns))
# all relevant occurance fields converted to ints, as a list
oc_column_vals = list(
    map(lambda c: int(c.replace('Class_', '')), df.columns[oc_columns]))
# oc_column_vals

# %%
bStat = binned_statistic(
    oc_column_vals, values=df.iloc[:, oc_columns], statistic=np.nansum, bins=[0, 5, 50, 95, 100])
bStat

# %%
bin_labels = ['0-5', '5-50', '50-95', '95-100']
dfB = pd.DataFrame(bStat.statistic, columns=bin_labels) / pd.DataFrame(
    df.loc[:, 'Class_sum']).values * 100  # , index=df.index) # df binned
dfB['Hylak_id'] = df.Hylak_id
dfB['Class_sum'] = df.Class_sum
dfB

# %%
## ensure dfB has unique Hylak_id keys
dfB = dfB.drop_duplicates(subset='Hylak_id')

## ensure dfB has unique Hylak_id keys
len(dfB) - len(dfB.drop_duplicates(subset='Hylak_id'))

# %% [markdown]
# ## Load shapefile and join in GSW values (full and binned)

# %%
## Load shapefile to join
gdf = gpd.read_file('/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp',
                    engine='pyogrio')  # bbox=(-180, 40, 180, 90)) # bbox can speed loading


# %%
## Filter columns
cols_to_keep = df.columns[[('Class' in c) or (
    'Hylak_id' in c) for c in df.columns]]

# %%

## Merge files
gdf_join_full = gdf.merge(df[cols_to_keep], left_on='Hylak_id',
                          right_on='Hylak_id', how='inner', validate='one_to_one')

# %%
## view
gdf_join_full.head(2)

# %%
## Write out full shapefile (slowww...52 minutes, 3.4 GB [without pyogrio])
gdf_join_full_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.shp')
gdf_join_full.to_file(gdf_join_full_pth, engine='pyogrio')

# %%
gdf.columns

# %%
## Merge binned file to bawld gdf (only keep a few original attributes)
# ['Cell_ID', 'Long', 'Lat', 'Area_Pct', 'Shp_Area', 'WETSCAPE', 'geometry']
gdf_join_binned = gdf.merge(
    dfB, left_on='Hylak_id', right_on='Hylak_id', how='inner', validate='one_to_one')
gdf_join_binned

# %%
gdf_join_binned.columns
# gdf_join_binned.columns[-6:]
# gdf_join_binned[[-6:]]
# gdf_join_binned.iloc[:,[0, -6:]]
gdf_join_binned[['Hylak_id', 'geometry', '0-5',
                 '5-50', '50-95', '95-100', 'Class_sum']]

# %%
## Write out binned shapefile (can join in remaining attributes later)
colsKeep = ['Hylak_id', 'geometry', '0-5',
            '5-50', '50-95', '95-100', 'Class_sum']
gdf_join_binned_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_binned.shp')
gdf_join_binned[colsKeep].to_file(gdf_join_binned_pth)

# %% [markdown]
# ## Histogram plots

# %%
## load if necessary (previously defined vars)
# print('Loading OC_full...')
# gdf_join_full_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_full.shp')
# gdf_join_full = pyogrio.read_dataframe(gdf_join_full_pth, use_arrow=True)

print('Loading OC_binned...')
gdf_join_binned_pth = os.path.join(analysis_dir, 'HL_zStats_Oc_binned.shp')
gdf_join_binned = pyogrio.read_dataframe(gdf_join_binned_pth, use_arrow=True)

bin_labels = ['0-5', '5-50', '50-95', '95-100']


# %%
## Preprocess to remove any nan's in important columns
gdf_join_binnedF = gdf_join_binned.dropna(subset=bin_labels)  # filtered
print(
    f'Dropped {gdf_join_binned.shape[0] - gdf_join_binnedF.shape[0]} rows with nans.')

## Averaging method 1: Take weighted average
try:
    weightAvg = np.average(
        gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Lake_area'], axis=0)
except:
    try:
        weightAvg = np.average(
            gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Shp_Area'], axis=0)
    except:
        gdf_join_binnedF = gdf_join_binnedF.merge(gdf[['Hylak_id', 'Lake_area']], left_on='Hylak_id',
                                                  right_on='Hylak_id', how='left', validate='one_to_one')  # Add in HL lake area if not present
        # weightAvg = np.average(gdf_join_binnedF[bin_labels], weights = gdf_join_binnedF['Class_sum'], axis=0) # If I was sloppy and didn't save HL area
        weightAvg = np.average(
            gdf_join_binnedF[bin_labels], weights=gdf_join_binnedF['Lake_area'], axis=0)
# weightAvg = np.average(gdf_join_binned[bin_labels], axis=0)

weightAvg

# %%
# Add Area <50% Oc
gdf_join_binnedF['Area_lt_50'] = (
    gdf_join_binnedF['0-5'] + gdf_join_binnedF['5-50']) / 100 * gdf_join_binnedF.Lake_area  # Units: km2
gdf_join_binnedF.head(3)

# %%
## Averaging option B: Sum and then average
dfS = gdf_join_binnedF[bin_labels] / 100 * pd.DataFrame(
    gdf_join_binnedF.loc[:, 'Class_sum']).values  # convert percentages back to sums
dfS['Hylak_id'] = gdf_join_binnedF.Hylak_id

## Add area bin
# gdf_join_binnedF['area_bin'] = pd.cut(gdf_join_binnedF.Class_sum, area_bins, labels=area_bins_labels)

## Melt for plotting
# 'area_bin' # melted data frame where Occurrence bins represent unweighted MEANS
dfsM = dfS.melt(id_vars=['Hylak_id'], var_name='Occurrence bin')

## Get sums for normalizing second axis
# areaSum = gdf_join_binnedF[bin_labels].sum()

## view
dfS.head(2)

# %%
## Reshape (melt) and plot as grouped bar plot (very slow to plot)
"Within all of one occurrence bin, what was the contribution of LEV values"
g = sns.catplot(dfsM,  # .iloc[:1000,:],
                hue='Occurrence bin', y='value', x='Occurrence bin', kind='bar', palette='cividis_r', errorbar=('ci', 95))
# Weighted percentage of pixels within bin (%)
g.set_axis_labels('Occurrence bin', 'Proportion')
g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

## Add second y-scale
# ax2 = g.ax.twinx()
# ax2.set_yticklabels()

# %% [markdown]
# ## Stacked bar plot showing lake size

# %%
# 3 Very Hydrolakes minimum lake size
assert not (np.any(gdf_join_binnedF.Lake_area < 0.1))
assert np.all(gdf_join_binnedF['Area_lt_50'] <= gdf_join_binnedF.Lake_area)

# %%
## Bin data by lake area
area_bins = pd.IntervalIndex.from_breaks(
    [0.1, 1, 10, 100, 1e3, 1e4, 1e5], closed='left')
area_bins_labels = ['0.1-1', '1-10', '10-100',
                    '100-1000', '1000-10000', '10000-100000']
gdf_join_binnedF['area_bin'] = pd.cut(
    gdf_join_binnedF.Lake_area, area_bins, right=False, labels=area_bins_labels)
gdf_join_binnedF.head(3)

# %%
## Melt for plotting
# Melted data frame where occurrence categories are MEAN, not SUM
dfsM2 = gdf_join_binnedF.drop(columns='geometry').melt(
    id_vars=['Hylak_id', 'area_bin'], var_name='Occurrence bin')

## group for later on
grouped = dfsM2.groupby(['area_bin', 'Occurrence bin']).mean().reset_index()

## view
# easier to view than the variable grouped, due to og index
table = dfsM2.groupby(['area_bin', 'Occurrence bin']).sum()
table

# %%
## Save this table
table.to_csv(os.path.join(table_dir, 'breakdown-size-oc.csv'))

# %%
## Plot lakes by double breakdown

## plot colors
plot_colors = ['r', 'g', 'b', 'orange']

## Try stacked bar plots... problem is that I can't add averages, and I can't easily divide dataframes with sum columns and also categorical columns...
# bar3 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100','50-95','5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[3]) # .query("`Occurrence bin` == '0-5'" #"@np.isin(`Occurrence bin`, ['0-5', '5-50'])"
# bar2 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['50-95','5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[2])
# bar1 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['5-50', '0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[1])
# bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['0-5'])], x="area_bin",  y="value", errorbar=None, color=plot_colors[0])

## Try again (I verified each group sums to 1), can also use barplot or catplot(..., kind='bar')
bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])], x="area_bin",
                   y="value", hue='Occurrence bin', errorbar=('ci', 95), color=plot_colors[0], palette='cividis_r')
# bar0.set_axis_labels('Proportion (%)', 'Area ($km^2$) bin')
bar0.axes.set_xlabel('Lake area ($km^2$) bin')
bar0.axes.set_ylabel('Proportion (%)')
bar0.axes.set_title('HydroLAKES breakdown')
plt.show()

# %%
## Actual stacked plot using new sns objects API.
sns.set_palette('cividis_r')  # doesn't st color map... not sure why...
g = so.Plot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])],
            x="area_bin", y="value", color='Occurrence bin').add(so.Bar(), so.Agg(), so.Stack())
# g. ax.set_xlabels(bin_labels)
g.label(x='Area bin ($km^2$)', y='Percentage')
# g.show()

# %% [markdown]
# ## Scrap

# %%
## Plot
## Reshape (melt) and plot as grouped bar plot
"Within all of one occurrence bin, what was the contribution of LEV values"
g = sns.catplot(gdf_join_binned[['Hylak_id'] + bin_labels].melt(id_vars='Hylak_id', var_name='Occurrence bin'),
                hue='Occurrence bin', y='value', x='Occurrence bin', kind='bar', palette='cividis_r', errorbar=('ci', 95))
g.set_axis_labels('', 'Unweighted percentage of pixels within bin (%)')
g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

# %%
## Weighted average histogram/barplot without conf intervals
dfWA = pd.DataFrame([weightAvg, bin_labels], index=['value', 'bin']).T
g = sns.catplot(dfWA, hue='bin', y='value', x='bin',
                kind='bar', palette='cividis_r')
g.set_axis_labels('', 'Area-weighted percentage of pixels within bin (%)')
g.set(title=f'Hydrolakes: GSW Occurrence breakdown for each bin')

# %%
## Weighted average histogram/barplot (alternate using MPL)

plt.bar(x=np.arange(4), height=weightAvg)
plt.ylabel('Unweighted percentage of pixels within bin (%)')


# %%
## Now plot as stacked bar plot (from https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot)

# set plot style: grey grid in the background:
# sns.set(style="darkgrid")

# set the figure size
# plt.figure(figsize=(14, 14))

## plot colors
plot_colors = ['r', 'g', 'b', 'orange']

## small dataset for testing
dfsM2_sub = dfsM2  # .iloc[1::80,:]

# top bar -> sum all values(smoker=No and smoker=Yes) to find y position of the bars
total = dfsM2_sub.groupby('area_bin')['value'].mean().reset_index()

# bar chart 1 -> top bars (group of 'smoker=No')
# bar_total = sns.barplot(x="area_bin",  y="value", data=total, color=plot_colors[0])
bar_total = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'], [
                              '95-100', '50-95', '5-50', '0-5'])]

# bottom bar ->  take only smoker=Yes values from the data
bin1 = dfsM2_sub[dfsM2_sub['Occurrence bin'] == '0-5']
bin2 = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'], ['5-50', '0-5'])]
bin3 = dfsM2_sub[np.isin(dfsM2_sub['Occurrence bin'],
                         ['50-95', '5-50', '0-5'])]
# bin4 = dfsM2_sub[dfsM2_sub['Occurrence bin']=='95-100'] # not needed

# bar chart 2 -> bottom bars (group of 'smoker=Yes')
# bar2 = sns.barplot(x="area_bin", y="value", data=bin1, estimator='mean', errorbar=None,  color=plot_colors[1])
# bar3 = sns.barplot(x="area_bin", y="value", data=bin2, estimator='mean', errorbar=None,  color=plot_colors[2])
# bar4 = sns.barplot(x="area_bin", y="value", data=bin3, estimator='mean', errorbar=None,  color=plot_colors[3])

# simple way of computing remaining bars by addition
total_bin1 = bin1.groupby('area_bin')['value'].mean().reset_index()
total_bin2 = bin2.groupby('area_bin')['value'].mean().reset_index()
total_bin3 = bin3.groupby('area_bin')['value'].mean().reset_index()

# add bar plots for sub totals
bar3 = sns.barplot(x="area_bin", y="value",
                   data=total_bin3, color=plot_colors[2])
bar2 = sns.barplot(x="area_bin", y="value",
                   data=total_bin2, color=plot_colors[1])
bar1 = sns.barplot(x="area_bin", y="value",
                   data=total_bin1, color=plot_colors[0])

# add legend
bars = [mpatches.Patch(color=j, label=bin_labels[i])
        for i, j in enumerate(plot_colors)]
# top_bar = mpatches.Patch(color='darkblue', label=bin_labels[0])
# bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
plt.legend(handles=bars)

# show the graph
plt.show()

# %%
## Now dubplicate unweighted mean
bar0 = sns.barplot(grouped[np.isin(grouped['Occurrence bin'], ['95-100', '50-95', '5-50', '0-5'])],
                   x="Occurrence bin", y="value", errorbar=('ci', 95), color=plot_colors[0], palette='cividis_r')

# %% [markdown]
# ## Scrap functions

# %%


def getRequests():
    """Generates a list of work items to be downloaded. Should be dquivalent to 'return modN', where modN is mod number.
    """
    ## Load vector dataset
    vect = ee.FeatureCollection(
        "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10").map(addMod)

    # For testing: Filter  to reduce size of operation
    # vectv = vect.filter("Pour_lat > 59.5").filter("Pour_lat < 59.6") #.filter("Long == -126.25")

    ## Aggregate by Hylak_id mod
    # return np.unique(vectF.aggregate_array('Country').getInfo()) # change to vect not vectF for real run
    # change to vect not vectF for real run
    return np.unique(vect.aggregate_array(modstr).getInfo())

# %%


def getRequests():
    ''' shortcut function that doesn't take 2.5 minutes.'''
    return range(modN)

# %%


def addMod(feature):
    '''Adds a new mod[n] column to FeatureCollection'''
    mod = modN  # defined at beginning
    modComputed = ee.Number(feature.get('Hylak_id')
                            ).mod(mod)  # ee.Number.parse(
    return feature.set('mod' + str(mod), modComputed)  # .double()

# %% [markdown]
# ## ERRORS

# %% [markdown]
# ```python
# Computing (Hi Ethan!)...
# Output exceeds the size limit. Open the full output data in a text editor
# ---------------------------------------------------------------------------
# HttpError                                 Traceback (most recent call last)
# File ~/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/data.py:328, in _execute_cloud_call(call, num_retries)
#     327 try:
# --> 328   return call.execute(num_retries=num_retries)
#     329 except googleapiclient.errors.HttpError as e:
#
# File ~/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/_helpers.py:130, in positional.<locals>.positional_decorator.<locals>.positional_wrapper(*args, **kwargs)
#     129         logger.warning(message)
# --> 130 return wrapped(*args, **kwargs)
#
# File ~/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/http.py:938, in HttpRequest.execute(self, http, num_retries)
#     937 if resp.status >= 300:
# --> 938     raise HttpError(resp, content, uri=self.uri)
#     939 return self.postproc(resp, content)
#
# HttpError: <HttpError 400 when requesting https://earthengine-highvolume.googleapis.com/v1alpha/projects/earthengine-legacy/value:compute?prettyPrint=false&alt=json returned "User memory limit exceeded.". Details: "User memory limit exceeded.">
#
# During handling of the above exception, another exception occurred:
#
# EEException                               Traceback (most recent call last)
# File ~/mambaforge/envs/geospatial/lib/python3.10/site-packages/geemap/common.py:6961, in zonal_stats_by_group(in_value_raster, in_zone_vector, out_file_path, statistics_type, decimal_places, denominator, scale, crs, tile_scale, return_fc, verbose, timeout, proxies, **kwargs)
#    6960     else:
# -> 6961         ee_export_vector(final_result, filename, timeout=timeout, proxies=proxies)
#    6963 except Exception as e:
# ...
#    6961         ee_export_vector(final_result, filename, timeout=timeout, proxies=proxies)
#    6963 except Exception as e:
# -> 6964     raise Exception(e)
#
# Exception: User memory limit exceeded.

# %% [markdown]
# ```python
# Output exceeds the size limit. Open the full output data in a text editor
# ---------------------------------------------------------------------------
# RemoteTraceback                           Traceback (most recent call last)
# RemoteTraceback:
# """
# Traceback (most recent call last):
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/data.py", line 328, in _execute_cloud_call
#     return call.execute(num_retries=num_retries)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/_helpers.py", line 130, in positional_wrapper
#     return wrapped(*args, **kwargs)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/http.py", line 938, in execute
#     raise HttpError(resp, content, uri=self.uri)
# googleapiclient.errors.HttpError: <HttpError 429 when requesting https://earthengine-highvolume.googleapis.com/v1alpha/projects/earthengine-legacy/value:compute?prettyPrint=false&alt=json returned "Too Many Requests: Request was rejected because the request rate or concurrency limit was exceeded.". Details: "Too Many Requests: Request was rejected because the request rate or concurrency limit was exceeded.">
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/multiprocessing/pool.py", line 125, in worker
#     result = (True, func(*args, **kwds))
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/multiprocessing/pool.py", line 51, in starmapstar
#     return list(itertools.starmap(args[0], args[1]))
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/decorator.py", line 232, in fun
#     return caller(func, *(extras + args), **kw)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/retry/api.py", line 73, in retry_decorator
#     return __retry_internal(partial(f, *args, **kwargs), exceptions, tries, delay, max_delay, backoff, jitter,
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/retry/api.py", line 33, in __retry_internal
# ...
#     772     return self._value
#     773 else:
# --> 774     raise self._value
#
# EEException: Too Many Requests: Request was rejected because the request rate or concurrency limit was exceeded.

# %% [markdown]
# Some error about element has a missing property. I treated it like an error caused by returning an empty table/feature collection and it seemed to resolve it.

# %% [markdown]
# This happened after 7 hrs of running (with 22,065 files generated)
# ```python
# Downloading data from https://earthengine-highvolume.googleapis.com/v1alpha/projects/earthengine-legacy/tables/356e1d94dc202d39f9c68cc224730aa3-c7c5ab24b3d58dd6aa199d1a90b902e0:getFeatures
# Please wait ...
# Data downloaded to /mnt/g/Ch4/GSW_zonal_stats/HL/v2/tiles/HL_zStats_Oc_Long-159.75_Lat55.75.csv
# Done with group:  [-159.75   55.75]
# Output exceeds the size limit. Open the full output data in a text editor
# ---------------------------------------------------------------------------
# RemoteTraceback                           Traceback (most recent call last)
# RemoteTraceback:
# """
# Traceback (most recent call last):
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/data.py", line 328, in _execute_cloud_call
#     return call.execute(num_retries=num_retries)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/_helpers.py", line 130, in positional_wrapper
#     return wrapped(*args, **kwargs)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/googleapiclient/http.py", line 938, in execute
#     raise HttpError(resp, content, uri=self.uri)
# googleapiclient.errors.HttpError: <HttpError 503 when requesting https://earthengine-highvolume.googleapis.com/v1alpha/projects/earthengine-legacy/value:compute?prettyPrint=false&alt=json returned "The service is currently unavailable.". Details: "The service is currently unavailable.">
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/geemap/common.py", line 6962, in zonal_stats_by_group
#     ee_export_vector(final_result, filename, timeout=timeout, proxies=proxies)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/geemap/common.py", line 1523, in ee_export_vector
#     selectors = ee_object.first().propertyNames().getInfo()
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/computedobject.py", line 98, in getInfo
#     return data.computeValue(self)
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/data.py", line 764, in computeValue
#     return _execute_cloud_call(
#   File "/home/ekyzivat/mambaforge/envs/geospatial/lib/python3.10/site-packages/ee/data.py", line 330, in _execute_cloud_call
# ...
#    6962         ee_export_vector(final_result, filename, timeout=timeout, proxies=proxies)
#    6964 except Exception as e:
# -> 6965     raise Exception(e)
#
# Exception: The service is currently unavailable.
