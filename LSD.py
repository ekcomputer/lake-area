#!/home/ekyzivat/mambaforge/envs/geospatial/bin/python
# # Lake-size distribution (LSD) scale comparison.
# 
# Goal: to load lake maps from the same region at two scale (HR and LR) and predict the small water body coverage (defined as area < 0.001 or 0.01 km2) from the LR dataset and physiographic region (with uncertainty).
# 
# Steps:
# 1. plot both LSD as survivor functions in log-log space (see functions from TGRS paper)

## Imports
from warnings import warn
import argparse
import os
import math
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
import seaborn as sns
import scipy.ndimage as ndi 
from sklearn.metrics import mean_squared_error
import pandas as pd
import geopandas as gpd
import pyogrio
import powerlaw
from tqdm import tqdm

## Plotting params
sns.set_theme('notebook', font='Ariel')
sns.set_style('ticks')
# mpl.rc('text', usetex=True)
mpl.rcParams['pdf.fonttype'] = 42

# ## Functions and classes
# ### Plotting functions

def findNearest(arr, val):
    ''' Function to find index of value nearest to target value'''
    # calculate the difference array
    difference_array = np.absolute(arr-val)
    
    # find the index of minimum element from the array
    index = difference_array.argmin()
    return index

def ECDFByValue(values, values_for_sum=None, reverse=True):
    """
    Returns sorted values and their cumsum.
    Called by plotECDFByValue.

    Parameters
    ----------
    values (array-like) : Values for histogram

    Returns
    -------
    X : sorted array
    S : cumsum of sorted array
    values_for_sum : (optional) an associated value to use for summing, for case of summing fluxes by area.

    """
    X = np.sort(values)
    if reverse:
        X = X[-1::-1] # highest comes first bc I reversed order
    
    if values_for_sum is None:
        S = np.cumsum(X) # cumulative sum, starting with highest [lowest, if reverse=False] values
    else:
        if isinstance(values_for_sum, pd.Series): # convert to pandas, since X is now pandas DF
            values_for_sum = values_for_sum.values
        sorted_indices = np.argsort(values)
        if reverse:
            values = values[-1::-1]
        S = np.cumsum(values_for_sum[sorted_indices])
    return X, S

def plotECDFByValue(values=None, reverse=True, ax=None, normalized=True, X=None, S=None, **kwargs):
    '''
    Cumulative histogram by value (lake area), not count.
    Creates, but doesn't return fig, ax if they are not provided. By default, CDF order is reversed to emphasize addition of small lakes (flips over ahorizontal axis).
    Required to provide either values or X and S. 'reverse' has no effect if X and S are provided.
    
    Parameters
    ----------
    values (array-like) : Values for histogram
    reverse (True) : Plot ascending from right
    ax (optional) : matplotlib axis for plot
    normalized (True) : Whether y-intercept should be 1
    X
    S
    Returns
    -------
    X : sorted array
    X : cumsum of sorted array    
    '''
    if values is None and (X is None or S is None):
        raise ValueError("Must provide either 'values' or 'X' and 'S'.")
    if X is not None and values is not None:
        raise ValueError("Both 'values' and 'X' were provided.")
    if X is None and S is None: # values is given
        X, S = ECDFByValue(values, reverse=reverse) # compute, returns np arrays
    else:    # X and S given
        if isinstance(S, pd.Series): # convert to pandas, since X is now pandas DF 
            S = S.values
        if isinstance(X, pd.Series): # sloppy repeat
            X = X.values
    if normalized:
        S = S/S[-1] # S/np.sum(X) has unintended result when using for extrapolated, when S is not entirely derived from X
        ylabel = 'Cumulative fraction of total area'
    else:
        ylabel = 'Cumulative area (million $km^2$)'
    if not ax:
        _, ax = plt.subplots()
    ax.plot(X, S, **kwargs) 

    ## Viz params
    ax.set_xscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Lake area ($km^2$)')
    return

def plotEPDFByValue(values, ax=None, bins=100, **kwargs):
    '''Cumulative histogram by value (lake area), not count. Creates, but doesn't return fig, ax if they are not provided. No binning used.'''
    X = np.sort(values)
    S = np.cumsum(X) # cumulative sum, starting with lowest values
    if not ax:
        _, ax = plt.subplots()
    ax.plot(X, X/np.sum(X), **kwargs) 

    ## Viz params
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylabel('Fraction of total area')
    ax.set_xlabel('Lake area')
    return  # S

def weightedStd(x, w):
    '''Computes standard deviation of values given as group means x, with weights w'''
    return np.sqrt((np.average((x-np.average(x, weights=w, axis=0))**2, weights=w, axis=0)).astype('float'))

def confidence_interval(x):
    '''
    A function to compute the confidence interval for each region group of summed lake areas in a size bin.
    
    Parameters
    ----------
    x : array-like (np.Array, pandas.DataFrame, or pd.Series)
        Array whose rows will act as input into function
    '''
    n = len(x)
    m = x.mean(numeric_only=True)
    se = x.sem(numeric_only=True)
    h = se * stats.t.ppf((1 + 0.95) / 2, n - 1)
    out = pd.Series({'mean': m, 'lower': m - h, 'upper': m + h})

    ## Set greater than 0
    return np.maximum(out, 0)

def confidence_interval_from_sem(group_sums, group_counts, group_sem):
    '''
    Confidence interval from std error of mean. Different format function than confidence_interval
    
    Output format is a df with a multi-index (0: size  bins, 1: categorical of either 'mean', 'lower', or 'upper'). Confidence intervals are values of the interval bounds, not anomalies (as would be accepted by pyplot error_bars).
    '''
    idx = pd.MultiIndex.from_tuples([(label, stat) for label in group_sums.index for stat in ['mean', 'lower', 'upper']], names=['size_bin', 'stat'])
    n = group_counts.sum()
    h = group_counts * group_sem * stats.t.ppf((1 + 0.95) / 2, n - 1)
    lower = np.maximum(group_sums - h, 0)
    upper = group_sums + h
    ds = pd.Series(index=idx, name='Area_km2', dtype=float)
    ds.loc[ds.index.get_level_values(1) == 'lower'] = lower.values
    ds.loc[ds.index.get_level_values(1) == 'upper'] = upper.values
    ds.loc[ds.index.get_level_values(1) == 'mean'] = group_sums.values
    # ds = pd.Series({'mean': group_sums, 'lower': np.maximum(group_sums - h, 0), 'upper': group_sums + h})

    ## Set greater than 0
    return ds  

def confidence_interval_from_extreme_regions(group_means, group_low, group_high, name='LEV_frac'):
    '''
    Instead of using within-group stats like sem, define CI as means of extreme groups
    
    Output format is a df with a multi-index (0: size  bins, 1: categorical of either 'mean', 'lower', or 'upper'). Confidence intervals are values of the interval bounds, not anomalies (as would be accepted by pyplot error_bars).
    '''
    idx = pd.MultiIndex.from_tuples([(label, stat) for label in group_means.index for stat in ['mean', 'lower', 'upper']], names=['size_bin', 'stat'])
    ds = pd.Series(index=idx, name=name, dtype=float)

    ## fille with nan if no CI
    if group_low is None:
        group_low = pd.Series(np.full_like(group_means, np.nan), index=group_means.index)
    if group_high is None:
        group_high = pd.Series(np.full_like(group_means, np.nan), index=group_means.index)

    ## set values
    ds.loc[ds.index.get_level_values(1) == 'lower'] = group_low.values
    ds.loc[ds.index.get_level_values(1) == 'upper'] = group_high.values
    ds.loc[ds.index.get_level_values(1) == 'mean'] = group_means.values
    return ds

def binnedVals2Error(binned_areas, n):
    '''Convert BinnedLSD.binnedAreas to error bars for plotting. May need to subtract 1 from n.'''
    ci = binned_areas.loc[:, ['lower', 'upper']].to_numpy().reshape((n,2)).T
    mean = binned_areas.loc[:, ['mean']].to_numpy()
    yerr = np.abs(ci-mean)
    return yerr

def public_attrs(self):
    public_attrs_dict = {}
    for attr, value in self.__dict__.items():
        if not attr.startswith('_'):
            public_attrs_dict[attr] = value
    return public_attrs_dict 

def interval_geometric_mean(interval):
    '''calculate the geometric mean of an interval'''
    return math.sqrt(interval.left * interval.right)

def loadBAWLD_CH4():
    ## Load
    df = pd.read_csv('/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD-CH4/data/ek_out/archive/BAWLD_CH4_Aquatic.csv', 
        encoding = "ISO-8859-1", dtype={'CH4.E.FLUX ':'float'}, na_values='-')
    len0 = len(df)

    ## Add total open water flux column
    df['CH4.E.FLUX'].fillna(df['CH4.D.FLUX']*1.2, inplace=True)
    df['CH4.D.FLUX'].fillna(df['CH4.E.FLUX']/1.2, inplace=True)
    df['CH4.DE.FLUX'] = df['CH4.D.FLUX'] + df['CH4.E.FLUX']

    ## Filter and pre-process
    df.query("SEASON == 'Icefree' ", inplace=True) # and `D.METHOD` == 'CH'
    df.dropna(subset=['SA', 'CH4.DE.FLUX', 'TEMP'], inplace=True)

    ## if I want transformed y as its own var
    # df['CH4.DE.FLUX.LOG'] = np.log10(df['CH4.DE.FLUX']+1) 

    ## print filtering
    len1 = len(df)
    print(f'Filtered out {len0-len1} BAWLD-CH4 values ({len1} remaining).')
    # print(f'Variables: {df.columns}')

    ## Linear models (regression)
    formula = "np.log10(Q('CH4.DE.FLUX')+0.01) ~ np.log10(SA) + TEMP" # 'Seasonal.Diff.Flux' 'CH4.D.FLUX'
    model = ols(formula=formula, data=df).fit()

    return model

def computeLEV(df: pd.DataFrame, ref_dfs: list, names: list, extreme_regions_lev=None) -> True:
    """
    Uses Bayes' law and reference Lake Emergent Vegetation (LEV) distribution to estimate the LEV in a given df, based on water Occurrence.

    Parameters
    ----------
    df (pd.DataFrame) : A dataframe with 101 water occurrence (Pekel 2016) classes ranging from 0-100%, named as 'Class_0',... 'Class_100'.
    
    ref_dfs (list) : where each item is a dataframe with format: Index: (LEV, dry land, invalid, water, SUM), Columns: ('HISTO_0', ... 'HISTO_100')
    
    names (list) : list of strings with dataset/region names in same order as ref_dfs
    
    extreme_regions_lev : array-like
        List of two region names to use for min/max LEV. Note, here region names must match entries in 'names,' so must be acronyms (e.g. CSD), unlike in BinnedLSD.
    Returns
    -------
    lev : pd.DataFrame with same index as df and a column for each reference LEV distribution with name from names. Units are unitless (fraction)

    """
    ## Rename columns in ref_df to match df
    func = lambda x: x.replace('HISTO_', 'Class_')
    ref_dfs = [ref_df.rename(columns=func) for ref_df in ref_dfs]

    # Multiply the dataframes element-wise based on common columns
    cols = ['LEV_' + name for name in names]
    df_lev = pd.DataFrame(columns = cols) # will have same length as df, which can become a LSD
    for i, ref_df in enumerate(ref_dfs):
        ''' df is in units of km2, ref_df is in units of px'''
        common_cols = df.columns.intersection(ref_df.columns.drop('Class_sum'))
        assert len(common_cols) == 101, f"{len(common_cols)} common columns found bw datasets. 101 desired."
        df_tmp = df[common_cols].reindex(columns=common_cols) # change order and only keep common (hist class) columns [repeats unnecessesarily ev time...]
        Class_sum = ref_df['Class_sum'] # used for update
        ref_df = ref_df[common_cols].reindex(columns=common_cols) # change order permanently (note: these orders are actually not numerical- no matter, as long as consistent bw two dfs)
        result = df_tmp /  df.Class_sum.values[:, None] * ref_df.loc['LEV', :] / ref_df.loc['CLASS_sum', :] # Mult Oc fraction of each oc bin by LEV fraction of each Oc bin.Broadcast ref_df over number of lakes
        df_lev['LEV_' + names[i]] = np.nansum(result, axis=1)

        # ## matrix version
        # C = (df_tmp.fillna(0) /  df.Class_sum.values[:, None]).values
        # D =  (ref_df.loc[['LEV'], :].values / ref_df.loc[['CLASS_sum'], :].values).T
        # df_lev['LEV_' + names[i]] = np.matmul(C,D)

        # ## matrix version w third term
        # C = (df_tmp.fillna(0) /  df.Class_sum.values[:, None]).values
        # D =  (ref_df.loc[['LEV'], :].values / ref_df.loc[['CLASS_sum'], :].values).T
        # df_lev['LEV_' + names[i]] = Class_sum.loc['LEV'] / Class_sum.loc['CLASS_sum'] * np.matmul(C,D)

    ## Summary stats
    if extreme_regions_lev is not None:
        assert np.all(['LEV_'+region_col in df_lev.columns for region_col in extreme_regions_lev]), "One region name in extreme_regions_lev is not present in lsd."
        df_lev['LEV_MEAN'] = df_lev[cols].mean(axis=1)
        df_lev['LEV_MIN'] = df_lev['LEV_'+extreme_regions_lev[0]]
        df_lev['LEV_MAX'] = df_lev['LEV_'+extreme_regions_lev[1]]
    else:
        df_lev['LEV_MEAN'] = df_lev[cols].mean(axis=1) # df_lev[cols].mean(axis=1)
        df_lev['LEV_MIN'] = df_lev[cols].min(axis=1) # df_lev[] #
        df_lev['LEV_MAX'] = df_lev[cols].max(axis=1) # df_lev[] #

    ## Join and return
    df_lev = pd.concat((df.drop(columns=np.concatenate((common_cols.values, ['Class_sum']))), df_lev), axis=1)

    return df_lev

def produceRefDs(ref_df_pth: str) -> True:
    """
    Pre-process raw dataframe in prep for computeLEV function.

    Parameters
    ----------
    ref_df (str) : Path to a csv file where each name is a region name and the values are dataframes with format: Index: (LEV, dry land, invalid, water, SUM), Columns: ('HISTO_0', ... 'HISTO_100')

    Returns: 
    -------
    df_out (pd.DataFrame): df with re-normalized and re-named columns

    """
    df = pd.read_csv(ref_df_pth, index_col='Broad_class').drop(index=['invalid', 'dry land', 'SUM'])

    ## Add missing HISTO_100 column if needed
    for i in range(101):
        if not f'HISTO_{i}' in df.columns:
            df[f'HISTO_{i}'] = 0

    df['HISTO_sum'] = df.sum(axis=1)
    df.loc['CLASS_sum', :] = df.sum(axis=0)

    return df

def loadUAVSAR(ref_names = ['CSB', 'CSD', 'PAD', 'YF']):
    '''
    Uses the regions in ref_names for averaging and regions in extreme_regions_lev_for_extrap for confidence interval.

    Returns:
    lsd_lev_cat: LSD
        LSD for reference regions. Used as LEV for non-inventoried lakes.
    ref_dfs: pd.DataFrame
        Giving Occurrence values by land cover class (e.g. LEV, Water) for reference UAVSAR regions. Used to estimate LEV for inventoried lakes.
    '''

    ## Load LEV/LSD from UAVSAR
    print('Loading UAVSAR and Pekel overlay...')
    default_ref_names = ['CSB', 'CSD', 'PAD', 'YF']
    pths_shp = ['/mnt/f/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/bakerc_16008_19059_012_190904_L090_CX_01_Freeman-inc_rcls_lakes.shp',
        '/mnt/f/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/daring_21405_17094_010_170909_L090_CX_01_LUT-Freeman_rcls_lakes.shp',
        '/mnt/f/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/padelE_36000_19059_003_190904_L090_CX_01_Freeman-inc_rcls_lakes.shp',
        '/mnt/f/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/YFLATS_190914_mosaic_rcls_lakes.shp']
    
    pths_csv = [ # CSV
        '/mnt/g/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay/bakerc_16008_19059_012_190904_L090_CX_01_Freeman-inc_rcls_brn_zHist_Oc_LEV_s.csv',
        '/mnt/g/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay/daring_21405_17094_010_170909_L090_CX_01_LUT-Freeman_rcls_brn_zHist_Oc_LEV_s.csv',
        '/mnt/g/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay/padelE_36000_19059_003_190904_L090_CX_01_Freeman-inc_rcls_brn_zHist_Oc_LEV_s.csv',
        '/mnt/g/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay/YFLATS_190914_mosaic_rcls_brn_train_zHist_Oc_LEV_s.csv' # Note YF is split into train/holdout XX vs XX %
        ]
    values = [{ 'pths_shp': pths_shp[i], 'pths_csv': pths_csv[i] } for i in range(len(pths_shp))]
    pths_dict = {k: v for k, v in zip(default_ref_names, values)}

    ## Filter based on ref_names
    _ = [pths_dict.pop(key) for key in default_ref_names if not key in ref_names]

    lsd_levs = []
    for i, pth in enumerate(pths_dict.values()):
        lsd_lev_tmp = LSD.from_shapefile(pth['pths_shp'], name=ref_names[i], area_var='area_px_m2', lev_var='em_fractio', idx_var='label', _areaConversionFactor=1e6, other_vars = ['edge', 'cir_observ'])
        lsd_lev_tmp.query('edge==0 and cir_observ==1', inplace=True)
        lsd_levs.append(lsd_lev_tmp)

    ## Plots (only works for four regions)
    # fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
    # for i, ax in enumerate(axes.flatten()):
    #     lsd_levs[i].plot_lev_cdf_by_lake_area(error_bars=False, ax=ax, plotLegend=False)

    ref_dfs = list(map(produceRefDs, [value['pths_csv'] for value in pths_dict.values()])) # load ref dfs 
    lsd_lev_cat = LSD.concat(lsd_levs, broadcast_name=True)
    return lsd_lev_cat, ref_dfs
    
def regionStats(lsd) -> True:
    """
    Analyze regions to choose min and max for confidence interval. Creates a df where each row is a region, with attributes for stats.

    Parameters
    ----------
    lsd : LSD

    Returns
    -------
    df_regions : pd.DataFrame

    """
    cols = ['region', 'maxA','A_0.1', 'nGT0.5', 'nGT1','nGT5']
    df_regions = pd.DataFrame(columns=cols)
    for i, region in enumerate(np.unique(lsd.Region)):
        lsd_tmp = lsd.query(f"Region == '{region}'")
        df_dict = {
            'region': region, 
            'maxA': lsd_tmp.Area_km2.max(), # make sure region has data beyond index range
            'A_0.1': lsd_tmp.area_fraction(0.1),
            'nGT0.5': lsd_tmp.query("Area_km2 > 0.5").shape[0],
            'nGT1': lsd_tmp.query("Area_km2 > 1").shape[0],
            'nGT5': lsd_tmp.query("Area_km2 > 5").shape[0],
            }
        df_tmp = pd.DataFrame(df_dict, index=pd.Index([i]))
        df_regions = pd.concat((df_regions, df_tmp), ignore_index=True)
    return df_regions
    
## Class (using inheritance)
class LSD(pd.core.frame.DataFrame): # inherit from df? pd.DataFrame # 
    '''Lake size distribution'''
    def __init__(self, df, name=None, area_var=None, region_var=None, idx_var=None, name_var=None, lev_var=None, t_var=None, _areaConversionFactor=1, regions=None, computeArea=False, other_vars=None, **kwargs):
        '''
        Loads df or gdf and creates copy only with relevant var names.
        If called with additional arguments (e.g. attributes from LSD class that is having a pd operation applied to it), they will be added as attributes

        Parameters
        ----------
        df : pandas.DataFrame or geopandas.GeoDataframe
            Dataframe to convert to LSD
        name : str
            Name of dataset
        area_var : string, optional
            Indicate which variable refers to shape area. Default is to first attempt with area_var = 'Area_km2'.
        region_var : string, optional
            To indicate which region if multiple regions in dataset
        idx_var: string, optional. Default is to first attempt with idx_var = 'index'.
            Index variable
        name_var: string, optional
            Name of variable that gives name of dataset (e.g. CIR or perl). Defaults to 'unamed'.
        lev_var: string, optional
            Name of var for LEV fraction, if loading from a dataset that has obsered LEV>
        areaConversionFactor : float, optional, defaults to 1.
            Denominator for unit conversion. 1 for km2, 1e6 for m2
        regions : list, optional
            If provided, will transform numeric regions to text
        computeArea : Boolean, default:False
            If provided, will compute Area from geometry. Doesn't need a crs, but needs user input for 'areaConversionFactor.'
        other_vars : list, optional
            If provided, LSD will retain these columns.
        ''' 

        ## Add default name
        if name == None:
            name='unamed'

        ## Check if proper column headings exist (this might occur if I am initiating from a merged group of existing LSD objects)
        if 'idx_'+name in df.columns:
            idx_var = 'idx_'+name
        if 'Area_km2' in df.columns:
            area_var = 'Area_km2'
        if 'em_fractio' in df.columns:
            lev_var = 'em_fractio'
        if 'Region' in df.columns:
            region_var = 'Region'
        if 'geometry' in df.columns: # if I am computing geometry, etc.
            geometry_var = 'geometry'
        else: geometry_var = None
        if 'est_mg_m2_day' in df.columns: 
            mg_var = 'est_mg_m2_day'
        else: mg_var = None
        if 'est_g_day' in df.columns:
            g_var = 'est_g_day'
        else: g_var = None
        if 'Temp_C' in df.columns:
            t_var = 'Temp_C'
        else: t_var = None

        ## Choose which columns to keep (based on function arguments, or existing vars with default names that have become arguments)
        columns = [col for col in [idx_var, area_var, region_var, name_var, geometry_var, mg_var, g_var, lev_var, t_var] if col is not None] # allows 'region_var' to be None

        ## Retain LEV variables if they exist
        columns += [col for col in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX'] if col in df.columns] # 'LEV_CSB', 'LEV_CSD', 'LEV_PAD', 'LEV_YF',       'LEV_MEAN', 'LEV_MIN', 'LEV_MAX'
        
        ##  Retain other vars, if provided
        if other_vars is not None:
            columns += [col for col in other_vars]

        super().__init__(df[columns]) # This inititates the class as a DataFrame and sets self to be the output. By importing a slice, we avoid mutating the original var for 'df'. Problem here is that subsequent functions might not recognize the class as an LSD. CAn I re-write without using super()?

        ## Compute areas if they don't exist
        if computeArea == True:
            if area_var is None:
                gdf = gpd.GeoDataFrame(geometry=self.geometry)
                self['Area_km2'] = gdf.area
                self.drop(columns='geometry', inplace=True)
            else:
                raise ValueError('area_var is provided, but computeArea is set to True.')

        ## rename vars
        if region_var is not None:
            self.rename(columns={idx_var:'idx_'+name, area_var:'Area_km2', lev_var:'LEV_MEAN', region_var:'Region'}, inplace=True)
        else:
            self.rename(columns={idx_var:'idx_'+name, area_var:'Area_km2', lev_var:'LEV_MEAN'}, inplace=True)

        ## Assert
        assert np.all(self.Area_km2 > 0), "Not all lakes have area > 0."

        ## Add attributes from input variables that get used in this class def
        self.name = name
        self.orig_area_var = area_var
        self.orig_region_var = region_var
        self.orig_idx_var = idx_var
        self.name_var = name_var # important, because otherwise re-initiating won't know to retain this column

        ## Add default passthrough attributes that get used (or re-used) by methods. Put all remaining attributes here.
        self.regions_ = None # np.unique(self.Region) # retain, if present
        self.isTruncated = False
        self.truncationLimits = None
        self.isBinned = False
        self.bins = None
        self.refBinnedLSD = None # Not essential to pre-set this

        ## Add passthrough attributes if they are given as kwargs (e.g. after calling a Pandas function). This overwrites defaults defined above.
        ## Any new attributes I create in future methods: ensure they start with '_' if I don't want them passed out.
        ## Examples: is_binned, orig_area_var, orig_region_var, orig_idx_var
        for attr, val in kwargs.items():
            setattr(self, attr, val)

        ## Set new attributes (Ensure only executed first time upon definition...)
        if _areaConversionFactor !=1:
            self.Area_km2 = self.Area_km2/_areaConversionFactor
        if regions is not None:
            self.reindexregions_(regions)
        if idx_var is None: # if loaded from shapefile that didn't have an explicit index column
            self.reset_index(inplace=True)
        if region_var is None: # auto-name region from name if it's not given
            self['Region'] = name

    def get_public_attrs(self):
        return public_attrs(self)
    
    @classmethod
    def from_shapefile(cls, path, name=None, area_var=None, lev_var=None, region_var=None, idx_var=None, **kwargs): #**kwargs): #name='unamed', area_var='Area', region_var='NaN', idx_var='OID_'): # **kwargs
        ''' 
        Load from shapefile on disk.
        Accepts all arguments to LSD.
        '''
        columns = [col for col in [idx_var, area_var, lev_var, region_var] if col is not None] # allows 'region_var' to be None
        
        ##  Retain other vars, if provided
        if 'other_vars' in kwargs:
            columns += [col for col in kwargs['other_vars']]

        read_geometry = False
        if 'computeArea' in kwargs:
            if kwargs['computeArea']==True:
                read_geometry = True
        df = pyogrio.read_dataframe(path, read_geometry=read_geometry, use_arrow=True, columns=columns)
        if name is None:
            name = os.path.basename(path).replace('.shp','').replace('.zip','')
        return cls(df, name=name, area_var=area_var, region_var=region_var, idx_var=idx_var, **kwargs)
    
    @classmethod
    def from_paths(cls, file_pattern, name='unamed', area_var=None, lev_var=None, region_var=None, idx_var=None, exclude=None, **kwargs):
        '''Load in serial with my custom class, based on a glob file pattern
         (can be parallelized with multiprocessing Pool.map). Help from ChatGPT

         Exclude: array_like
            An array of filepaths or unique strings within files to skip loading.
         '''
        # Define the file pattern
        shapefiles = glob(file_pattern)
        dfs = [] # create an empty list to store the loaded shapefiles

        ## Filter out raw regions
        if exclude is not None:
            shapefiles = [file for file in shapefiles if not any(fname in file for fname in exclude)]

        # loop through the shapefiles and load each one using Geopandas
        for shpfile in shapefiles:
            lsd = cls.from_shapefile(shpfile, area_var=area_var, lev_var=lev_var, region_var=region_var, idx_var=idx_var, **kwargs)
            dfs.append(lsd)
        
        # merge all the loaded shapefiles into a single GeoDataFrame
        lsd = LSD.concat(dfs, ignore_index=True) #, crs=gdfs[0].crs)
        lsd.name=name
        return lsd

    def reindexregions_(self, regions):
        ''' Where regions is a list of region names corresponding to the numbers in the existing inedex. 2 is hard-coded in for now and refers to the 1-based indexing of the CIR shapefile and skips the final region (which is just the previous ones summed).'''
        self['Region'] = np.array(regions)[self['Region'].values - 2]
        
    def regions(self):
        ''' 
        Return unique regions names if not already generated.
        No need to call regions_ directly, which will re-compute every time.  
            '''
        if self.regions_ is None:
            self.regions_ = np.unique(self.Region)
        return self.regions_
        
    @classmethod # not sure if necessary
    def concat(cls, lsds, broadcast_name=False, **kwargs): 
        '''
        Concatenates multiple LSD instances.
        Don't call on an instance, but call from the base class. Takes the name of the first LSD
        lsds : arry_like
            Array of LSDS to concatenate.
        broadcast_name : boolean
            If true, add a column to each of the input LSDS with their name (name of dataset)
        
        Example: LSD.concat((lsd1, lsd2))
        '''
        if broadcast_name: # if combining multiple lsds from different sources
            for lsd in lsds:
                lsd['Name'] = lsd.name
                name_var = 'Name'
                name='multi'
        else: # if loading in lsds from same source, but different files
            name_var = None
            name=lsds[0].name
        return cls(pd.concat(lsds, **kwargs), name=name, name_var=name_var) # Need to re-init before returning because pd.DataFrame.concat is a function, not method and can't return in-place. Therefore, it returns a pd.DataFrame object that needs to be converted back to a LSD.

    def truncate(self, min:float, max:float=np.inf, inplace=False, **kwargs):
        '''
        Truncates LSD by keeping only lakes >= min threshold [and < max threshold].
        Always performed inplace.
        '''
        if inplace == True:
            pd.DataFrame.query(self, "(Area_km2 >= @min) and (Area_km2 < @max)", inplace=inplace, **kwargs) # true inplace
            self.isTruncated=True
            self.truncationLimits=(min, max)
        else:
            attrs = self.get_public_attrs()
            lsd = LSD(pd.DataFrame.query(self, "(Area_km2 >= @min) and (Area_km2 < @max)", inplace=inplace, **kwargs), **attrs) # false # TODO: can rewrie this without calling a new LSD._init_: just slice the rows of the LSD based on index...
            lsd.isTruncated=True
            lsd.truncationLimits=(min, max)           
            return lsd         
    
    def query(self, expr, inplace=False):
        '''
        Runs np.DataFrame.query and returns an LSD of results. 
        When inplace=False, output is re-generated as an LSD. When inplace=True, output LSD class is unchanged.
        '''
        if inplace == True:
            pd.DataFrame.query(self, expr, inplace=inplace) # true
        else:
            attrs = self.get_public_attrs()
            return LSD(pd.DataFrame.query(self, expr, inplace=inplace), **attrs) # false
        
    def area_fraction(self, limit):
        '''
        Compute the fraction of areas from lakes < area given by lim, only if not computed already.
        Creates attribute A_[n]_ where n=0.001, 0.01, etc.
        No need to call A_0.001_ directly, because it may not exist.
        Will not include extrapolated areas in estimate, because bin edges may not align. Use extrapolated_area_fraction() instead.

        TODO: add option to include extrapolation in estimate.
        '''
        if self.isTruncated:
            if self.truncationLimits[1]  is not np.inf:
                warn("Careful, you are computing an area fraction based on a top-truncated LSD, so the fraction may be too high.")
        attr = f'_A_{limit}' # dynamically-named attribute (_ prefix means it won't get copied over after a truncation or concat)
        if attr in self.get_public_attrs():
            return getattr(self, attr)
        else: # do computation
            area_fraction = self.truncate(0, limit).Area_km2.sum()/self.Area_km2.sum()
            setattr(self, attr, area_fraction)
            return area_fraction
    
    def extrapolated_area_fraction(self, ref_LSD, bottomLim, limit, emax=0.5, tmax=5):
        """
        Computes area fraction from lakes < limit in area, for limit < minimum observed lake area.

        Does this by calling extrapolate() twice, using different bottomLims to ensure the area thresholds in question exactly match the desired limit. Creates attribute A_[n]_ where n=0.001, 0.01, etc. Self should be truncated, ref_LSD shouldn't.

        TODO: add confidence interval
    
        Parameters
        ----------
        self : LSD (truncated)
        limit  : float
            threshold. Will be used for extrapolation (for numerator).
        ref_LSD : LSD (not truncated)
            Reference LSD to bin and then use for extrapolation.
        bottomLim : float
            Bottom size limit to which to extrapolate (for denominator).    
        emax : float, default 0.5
            Extrapolation limit max. emax defines the right bound of the extrapolation region (and the left bound of the index region). Should be set a little higher than the resolution of the dataset to be extrapolated to account for biases near its detection limit.
        tmax : float, default 5
            tmax defines the right bound of the index region (so it truncates the ref LSD). don't set it too high, which introduces variability between datasets in the index region fractional areas.
        Returns
        -------
        area_fraction (float): sum all areas < limit and divide

        """
        ## Checks
        if self is None:
            raise ValueError("self cannot be None")
        assert isinstance(ref_LSD, LSD), "ref_LSD must be type LSD"
        assert not ref_LSD.isTruncated, "ref_LSD shouldn't already be truncated, because it needs to happen inside this function."
        assert self.isTruncated, "To be explicit, truncate self, which indicates which region to extrapolate to."
        assert limit < self.truncationLimits[0], f"Limit ({limit}) is >= the lower truncation limit of LSD ({self.truncationLimits[0]}), so use area_fraction() method instead." # auto run area_fraction
        assert limit > bottomLim, f"'limit' ({limit}) must be > bottomLim ({bottomLim})."
        assert emax >= self.truncationLimits[0], f"emax ({emax}) should be >= the top truncation limit of self ({self.truncationLimits[1]})"
        if self.truncationLimits[1]  is not np.inf:
            warn("Careful, you are computing an extrapolated area fraction based on a top-truncated LSD, so the fraction may be too high.")
        
        ## Make copy for modification via extrapolate function
        attrs = self.get_public_attrs()
        lsd = LSD(self, **attrs)

        ## Computation
        binned_ref = BinnedLSD(ref_LSD.truncate(0, tmax), btm=limit, top=emax, compute_ci_lsd=False)
        lsd.truncate(emax, np.inf, inplace=True)
        lsd.extrapolate(binned_ref)
        num = lsd.sumAreas(includeExtrap=True)

        lsd = LSD(self, **attrs) # re-init
        binned_ref = BinnedLSD(ref_LSD.truncate(0, tmax), bottomLim, emax, compute_ci_lsd=False)
        lsd.truncate(emax, np.inf, inplace=True)
        lsd.extrapolate(binned_ref)
        denom = lsd.sumAreas(includeExtrap=True)
        area_fraction = 1 - num/denom


        # tmin, tmax = (0.0001,30) # Truncation limits for ref LSD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
        # emax = 0.5 # Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
        # binned_ref = BinnedLSD(lsd.truncate(tmin, tmax), tmin, emax, compute_ci_lsd=True) # reference distrib (try 5, 0.5 as second args)


        ## update area fract on original lsd and return
        attr = f'_A_{limit}'
        setattr(self, attr, area_fraction) # Save value within LSD structure
        return area_fraction
    
    def extrapolate(self, ref_BinnedLSD, ref_BinnedLEV=None):
        '''
        Extrapolate by filling empty bins below the dataset's resolution.

        Note: The limits of the extrapolation are defined by the btmEdge/topEdge of ref_BinnedLSD. The top limit of rererence distribution is defined by the top truncation of the reference binnedLSD. Function checks to make sure it is truncated <= 5 km2.
        
        Parameters
        ----------
        ref_BinnedLSD : binnedLSD
            Reference binnedLSD used for extrapolation.
        ref_BinnedLEV : binnedLSD that has LEV 
            Reference binnedLSD used for LEV extrapolation.This binned LSD probably comes from different data, hence the additional argument.

        ''' 
        # returns a binnedLSD   
        # give error if trying to extrapolate to a smaller area than is present in ref distrib

        # HERE truncate last bin

        ## Check validity
        # assert self.isTruncated # can correct for this
        assert ref_BinnedLSD.isTruncated, "Reference binnedLSD must be top truncated or its bin estimates will be highly variable."
        assert self.isTruncated, "LSD should be bottom-truncated when used for extrapolation to be explicit."
        assert self.truncationLimits[0] == ref_BinnedLSD.topEdge, f"Mismatch between LSD bottom truncation limit ({self.truncationLimits[0]}) and ref binned LSD top edge ({ref_BinnedLSD.topEdge})"
        assert ref_BinnedLSD.isNormalized
        if ref_BinnedLSD.truncationLimits[1] > 5:
            warn("Reference binned LSD should be top truncated to <= ~ 5 km2 to avoid too large of an index region (highly variable and HR lsd might be biased low in a large index region).")
        if ref_BinnedLEV is not None:
            assert hasattr(ref_BinnedLEV, 'binnedLEV'), "If ref_BinnedLEV is provided, it must have a a binnedLEV attribute."
            assert ref_BinnedLSD.nbins == ref_BinnedLEV.nbins, "nbins differs between ref_BinnedLSD and ref_BinnedLEV"
            assert ref_BinnedLSD.btmEdge == ref_BinnedLEV.btmEdge, f"Ref binned LSD btm edge ({ref_BinnedLSD.btmEdge}) doesn't equal ref binned LEV btm edge ({ref_BinnedLEV.btmEdge})"
            assert ref_BinnedLSD.topEdge == ref_BinnedLEV.topEdge, f"Ref binned LSD top edge ({ref_BinnedLSD.topEdge}) doesn't equal ref binned LEV top edge ({ref_BinnedLEV.topEdge})"
        ## Perform the extrapolation (bin self by its bottom truncation limit to the indexTopLim, and simply multiply it by the normalized refBinnedLSD to do the extrapolation)!
        
        index_region_sum = self.truncate(ref_BinnedLSD.topEdge, ref_BinnedLSD.truncationLimits[1]).Area_km2.sum() # Sum all area in LSD in the index region, defined by the topEdge and top-truncation limit of the ref LSD.
        last = ref_BinnedLSD.binnedAreas.index.get_level_values(0).max() # last index with infinity to drop (use as mask)
        binned_areas = ref_BinnedLSD.binnedAreas.drop(index=last) * index_region_sum # remove the last entries, which are mean, lower, upper for the bin that goes to np.inf
        
        ## Return in place a new attribute called extrapLSD which is an instance of a binnedLSD
        self.extrapLSD = BinnedLSD(btm=ref_BinnedLSD.btmEdge, top=ref_BinnedLSD.topEdge, nbins=ref_BinnedLSD.nbins, compute_ci_lsd=ref_BinnedLSD.hasCI_lsd, binned_areas=binned_areas)
       
        ## Add binnedLEV, if specified. Technically, not even extrapolating, just providing the reference distribution, but using same syntax as for LSD for compatibility.
        if ref_BinnedLEV is not None:
            last_lev = ref_BinnedLEV.binnedAreas.index.get_level_values(0).max()
            self.extrapLSD.binnedLEV = ref_BinnedLEV.binnedLEV.drop(index=last_lev) # will this work if no LEV_MIN/MAX?

        ## Manually update its attributes
        self.extrapLSD.indexTopLim = ref_BinnedLSD.truncationLimits[1] # since bottom of index region matches, ensure top does as well in definition.
        self.extrapLSD.isExtrapolated=True
        self.extrapLSD.isNormalized = False # units of km2, not fractions
        self.extrapLSD.bottomLim = ref_BinnedLSD.btmEdge
        self.extrapLSD.topLim = ref_BinnedLSD.topEdge
        self.extrapLSD.hasCI_lsd = ref_BinnedLSD.hasCI_lsd
        if hasattr(ref_BinnedLEV, 'hasCI_lev'):
            self.extrapLSD.hasCI_lev = ref_BinnedLEV.hasCI_lev
        self.extrapLSD.extreme_regions_lsd = ref_BinnedLSD.extreme_regions_lsd
        if hasattr(ref_BinnedLEV, 'extreme_regions_lev'):
            self.extrapLSD.extreme_regions_lev = ref_BinnedLEV.extreme_regions_lev
        ## Save reference binned LSD (which has any remaining attrs)
        self.refBinnedLSD = ref_BinnedLSD
        return
    
    def sumAreas(self, ci=False, includeExtrap=True):
        '''
        Sums all lake area in distribution.
        
        Parameters
        ----------
        ci : Boolean
            Whether to output the lower and upper confidence intervals.
        includeExtrap : Boolean
            Whether to include any extrapolated areas, if present
        '''
        self._observed_area = self.Area_km2.sum()
        self._extrap_area = None # init
        self._total_area = None
        if includeExtrap:
            assert hasattr(self, 'extrapLSD'), "LSD doesn't include an extrapLSD attribute."
            if ci==False:
                self._extrap_area = self.extrapLSD.sumAreas(ci=ci)
                out =  self._extrap_area + self._observed_area
            else:
                self._extrap_area = np.array(self.extrapLSD.sumAreas(ci=ci))
                out =  tuple(self._extrap_area + self._observed_area) # convert to tuple, as is common for python fxns to return
            self._total_area = self._observed_area + self._extrap_area
        else:
            out =  self._observed_area
        
        return out
    
    def sumLev(self, includeExtrap=True, asFraction=False):
        '''
        Sums all lev area in distribution. Requires LEV CI, even if including nans.
        
        Parameters
        ----------
        includeExtrap : Boolean (True)
            Whether to include any extrapolated areas, if present
        asFraction : Boolean (False)
            If True: computes weighted mean LEV (unitless, equal to total LEV / total lake area)
        '''
        lev_sum = confidence_interval_from_extreme_regions(*[(pd.Series((self[param]* self.Area_km2).sum())) for param in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']]).loc[0,:]
        # m = confidence_interval_from_extreme_regions(*[pd.Series(np.average(self[param], weights=self.Area_km2)) for param in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']]).loc[0,:]
        result = lev_sum
        total_area = self.sumAreas(includeExtrap=False, ci=False)
        if includeExtrap:
            if hasattr(self, 'extrapLSD'):
                if hasattr(self.extrapLSD, 'binnedLEV'):
                    lev_sum_extrap = self.extrapLSD.sumLev(asFraction=False)
                    result = lev_sum  + lev_sum_extrap
                    # result = lev_sum * self.sumAreas(includeExtrap=False) + lev_sum_extrap * self.extrapLSD.sumAreas(ci=False)
                    total_area = self.sumAreas(includeExtrap=True)
                    # np.average(pd.concat((m, m_extrap), axis=1), axis=1, weights=[self.sumAreas(includeExtrap=False), self.extrapLSD.sumAreas(ci=False)]) # Note:weights are scalers
                    pass
                else:
                    warn('No LEV in extrapolated LSD. Returning calculation based on inventoried lake LEV only.')
            else: 
                warn('No extrapolated LSD. Returning calculation based on inventoried lake LEV only.')
        else:
            pass
        if asFraction:
            result /= total_area
        return result
    
    def predictFlux(self, model, includeExtrap=True):
        '''
        Predict methane flux based on area bins and temperature.
        
        TODO: 
            * Use temp as a df variable, not common attribute
            * Lazy algorithm- only compute if self._Total_flux_Tg_yr not present
        Parameters
        ----------
        model : statsmodels
        coeff : array-like
            list of model coefficients
        returns: ax
        '''
        assert 'Temp_C' in self.columns, "LSD needs a Temp_C attribute in order to predict flux."
        if includeExtrap==True:
            assert hasattr(self, 'extrapLSD'), "includeExtrap was set to true, but no self.extrapLSD found."
            self.extrapLSD.Temp_C = np.average(self.Temp_C, weights=self.Area_km2) # Use lake-area-weighted average temp as temp for extrapolated lakes, since we don't know where they are located
            self.extrapLSD.predictFlux(model)
            binned_total_flux_Tg_yr = self.extrapLSD._total_flux_Tg_yr
        else:
            binned_total_flux_Tg_yr = 0 # Not NaN, because needs to be added
        
        ## Flux (areal, mgCH4/m2/day)
        self['est_mg_m2_day'] = 10**(model.params.Intercept +
        model.params['np.log10(SA)'] * np.log10(self.Area_km2) 
        + model.params['TEMP'] * self.Temp_C) - 0.01 # jja, ann, son, mam

        ## Flux (flux rate, gCH4/day)
        self['est_g_day'] = self.est_mg_m2_day * self.Area_km2 * 1e3 # * 1e6 / 1e3 # (convert km2 -> m2 and mg -> g)

        self._total_flux_Tg_yr = self['est_g_day'].sum() * 365.25 / 1e12 + binned_total_flux_Tg_yr# see Tg /yr
        # return self._Total_flux_Tg_yr
        return

    ## Plotting
    def plot_lsd(self, all=True, plotLegend=True, groupby_name=False, cdf=True, ax=None, plotLabels=False, **kwargs):
        '''
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False).

        Accepts kwargs to plotECDFByValue.
        
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        plotLabels : boolean (False)
            Label plots by region
        returns: ax
        '''
        ## Cumulative histogram by value (lake area), not count
        
        ## colors
        # rainbow_cycler = cycler
        sns.set_palette("colorblind", len(self.regions())) # colors from https://stackoverflow.com/a/46152327/7690975 Other option is: `from cycler import cycler; `# ax.set_prop_cycle(rainbow_cycler), plt(... prop_cycle=rainbow_cycler, )

        ## plot
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)

        if groupby_name==False: # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                X, S = ECDFByValue(pd.DataFrame.query(self, 'Region == @region').Area_km2, reverse=False)
                plotECDFByValue(X=X, S=S/1e6, ax=ax, alpha=0.4, label=region, **kwargs)

        else:
            assert 'Name' in self.columns, "LSD is missing 'Name' column."
            # cmap = colors.Colormap('Pastel1')
            names = np.unique(self['Name'])
            cmap = plt.cm.get_cmap('Paired', len(names))
            for j, name in enumerate(names):
                for i, region in enumerate(np.unique(pd.DataFrame.query(self, 'Name == @name').Region)): # can't use .regions() after using DataFrame.query because it returns a DataFrame
                    X, S = ECDFByValue(pd.DataFrame.query(self, 'Region == @region').Area_km2, reverse=False)
                    plotECDFByValue(X=X, S=S/1e6, ax=ax, alpha=0.6, label=name, color=cmap(j), **kwargs)
        
        ## repeat for all
        if all:
            X, S = ECDFByValue(self.Area_km2, reverse=False)
            plotECDFByValue(X=X, S=S/1e6, ax=ax, alpha=0.4, color='black', label='All', **kwargs)

        ## Legend and labels
        if plotLegend:
            ax.legend(loc= 'center left', bbox_to_anchor=(1.04, 0.5)) # legend on right (see https://stackoverflow.com/a/43439132/7690975)
        if plotLabels:
            labelLines(ax.get_lines())

        return ax

    def plot_flux(self, all=True, plotLegend=True, groupby_name=False, cdf=True, ax=None, normalized=True, reverse=True):
        '''
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False).
        
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        returns: ax
        '''
        ## Cumulative histogram by value (lake area), not count
        
        ## colors
        # rainbow_cycler = cycler
        assert 'est_g_day' in self.columns, "LSD doesn't have a flux estimate yet." 
        sns.set_palette("colorblind", len(self.regions())) # colors from https://stackoverflow.com/a/46152327/7690975 Other option is: `from cycler import cycler; `# ax.set_prop_cycle(rainbow_cycler), plt(... prop_cycle=rainbow_cycler, )

        ## plot
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)

        if groupby_name==False: # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                lsd_by_region = self.query(f'Region == "{region}"')
                X, S = ECDFByValue(lsd_by_region.Area_km2, values_for_sum=lsd_by_region.est_g_day * 365.25 / 1e12, reverse=reverse) # convert from g/day to Tg/yr
                plotECDFByValue(X=X, S=S, ax=ax, alpha=0.4, label=region, normalized=normalized, reverse=False) # if reverse, set this in ECDFByValue on previous line

        else:
            assert 'Name' in self.columns, "LSD is missing 'Name' column."
            # cmap = colors.Colormap('Pastel1')
            names = np.unique(self['Name'])
            cmap = plt.cm.get_cmap('Paired', len(names))
            for j, name in enumerate(names):
                for i, region in enumerate(np.unique(pd.DataFrame.query(self, f'Name == "{name}"').Region)): # OLD: can't use .regions() after using DataFrame.query because it returns a DataFrame
                    lsd_by_region_name = self.query(f'Region == "{region}"')
                    X, S = ECDFByValue(lsd_by_region_name.Area_km2, values_for_sum=lsd_by_region_name.est_g_day * 365.25 / 1e12, reverse=reverse)
                    plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, label=name, color=cmap(j), normalized=normalized, reverse=False)
        ## repeat for all
        if all:
            X, S = ECDFByValue(self.Area_km2, values_for_sum=self.est_g_day * 365.25 / 1e12, reverse=reverse)
            plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, color='black', label='All', normalized=normalized, reverse=False)

        ## Legend
        if plotLegend:
            ax.legend(loc= 'center left', bbox_to_anchor=(1.04, 0.5)) # legend on right (see https://stackoverflow.com/a/43439132/7690975)

        ## Override default axes
        if normalized:
            ylabel = 'Cumulative fraction of emissions'
        else:
            ylabel = 'Cumulative emissions (Tg/yr)'
        ax.set_ylabel(ylabel)

        return ax

    def plot_extrap_lsd(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots LSD using both measured and extrapolated values. 
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False).
        
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        error_bars : boolean
            Whether to include error bars (not recommended, since this plots a CDF)
        returns: ax
        '''       
        ## Prepare values
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)
        
        if self.extrapLSD.hasCI_lsd: # Need to extract means in a different way if there is no CI
            means = self.extrapLSD.binnedAreas.loc[:, 'mean']
        else:
            means = self.extrapLSD.binnedAreas
        X, S = ECDFByValue(self.Area_km2, reverse=False) # plot for observed part of distrib
        geom_means = np.array(list(map(interval_geometric_mean, self.extrapLSD.binnedAreas.loc[:, 'mean'].index.get_level_values(0)))) # take geom mean of each interval
        means=means.values # convert to numpy
        S += self.extrapLSD.sumAreas() # add sum of extrap distrib to original cumsum
        S0 = np.cumsum(means) # pre-pend the cumsum of extrap distrib binned vals
        if not 'color' in kwargs:
            kwargs['color']='black'

        ## Plot
        if normalized: # need to normalize outside of plotECDFByValue function
            denom = self.sumAreas() / 1e6
        else:
            denom = 1

        ## Add error bars
        if error_bars == True:      
            assert self.extrapLSD.hasCI_lsd, 'If plotting error bars, self.extrapLSD.hasCI_lsd needs a CI.'
            
            ## as area plot: extrap section
            S_low0 = np.cumsum(self.extrapLSD.binnedAreas.loc[:, 'lower']) # btm section
            S_up0  = np.cumsum(self.extrapLSD.binnedAreas.loc[:, 'upper'])
            
            ## Observed section
            yerr_top = self.extrapLSD.binnedAreas.loc[:, 'mean'].sum() - self.extrapLSD.binnedAreas.loc[:, 'lower'].sum()
            yerr_btm = self.extrapLSD.binnedAreas.loc[:, 'upper'].sum() - self.extrapLSD.binnedAreas.loc[:, 'mean'].sum()
            S_low = np.maximum(S-yerr_top, 0)
            S_up = S + yerr_btm
            ax.fill_between(np.concatenate((geom_means, X)), np.concatenate((S_low0, S_low))/denom/1e6, np.concatenate((S_up0, S_up))/denom/1e6, alpha=0.1, color=kwargs['color'])
        else:
            pass
        
        ## Plot main curves 
        plotECDFByValue(ax=ax, alpha=1, X=X, S=S/denom/1e6, normalized=False, reverse=reverse, **kwargs)
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0/denom/1e6, normalized=False, reverse=reverse, linestyle='dashed', **kwargs) # second plot in dashed for extrapolated
        if normalized: # need to change label outside of plotECDFByValue function
            ax.set_ylabel('Cumulative fraction of total area')
        if plotLegend:
            ax.legend(loc='best')
        ax.set_ylim(0, ax.get_ylim()[1])
        return ax
    
    def plot_extrap_lev(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots lev from LSD using both measured and extrapolated values. 
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False).
        
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        error_bars : boolean
            Whether to include error bars (not recommended, since this plots a CDF)
        **kwargs : get passed to plt.plot via plotECDFByValue

        returns: ax
        '''       
        assert 'LEV_MEAN' in self.columns, "LSD doesn't have a LEV estimate yet." 
        assert hasattr(self.extrapLSD, 'binnedLEV'), "binnedLSD doesn't have an extrap lev estimate yet." 
        assert reverse==False, "No branch yet written for flux plots in reverse."
        if error_bars:
            assert_vars = ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']
        else:
            assert_vars = ['LEV_MEAN']
        for var in assert_vars:
            assert var in self.columns, f"LSD is missing {var} column, which is required to plot lev cdf."
        
        ## Prepare values
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)
        try:
            geom_means = np.array(list(map(interval_geometric_mean, self.extrapLSD.binnedLEV.index))) # take geom mean of each interval

        except:
            geom_means = np.array(list(map(interval_geometric_mean, self.extrapLSD.binnedLEV.loc[:, 'mean'].index.get_level_values(0)))) # if ci values are present
        means = self.extrapLSD.binnedLEV * self.extrapLSD.binnedAreas.loc[:, 'mean'] # used to be a branch for if self.extrapLSD.hasCI_lsd...

        X, S = ECDFByValue(self.Area_km2, values_for_sum=self.LEV_MEAN*self.Area_km2, reverse=False) # convert from frac to km2
        binned_lev_km2 = (self.extrapLSD.binnedLEV * self.extrapLSD.binnedAreas.loc[:, 'mean']) # TODO: need branch for if no 'mean' col for all occurrences, or ensure it always will have one # HERE 4/5/2023
        S += binned_lev_km2.loc[:, 'mean'].sum() # self.extrapLSD.sumLev0() # add sum of extrap distrib to original cumsum
        S0 = np.cumsum(means.loc[:, 'mean']) # pre-pend the cumsum of extrap distrib binned vals
        if not 'color' in kwargs:
            kwargs['color']='green'

        ## Add error bars
        if error_bars == True:
            assert self.extrapLSD.hasCI_lev, "error_bars is set, but extrapolated lev has no CI (self.extrapLSD.hasCI_lev is False)"
            
            ## Compute error bounds for extrapolated LEV
            S_low0 = np.cumsum(self.extrapLSD.binnedLEV.loc[:, 'lower']*self.extrapLSD.binnedAreas.loc[:, 'mean']) # btm section
            S_up0 =  np.cumsum(self.extrapLSD.binnedLEV.loc[:, 'upper']*self.extrapLSD.binnedAreas.loc[:, 'mean'])
            
            ## Compute error bounds for estimated LEV from obs
            X_low, S_low = ECDFByValue(self.Area_km2, self.LEV_MIN*self.Area_km2, reverse=False)
            S_low += binned_lev_km2.loc[:, 'lower'].sum()
            _, S_up = ECDFByValue(self.Area_km2, self.LEV_MAX*self.Area_km2, reverse=False)
            S_up += binned_lev_km2.loc[:, 'upper'].sum()

            ## Area plot
            ax.fill_between(np.concatenate((geom_means, X_low)), np.concatenate((S_low0, S_low))/1e6, np.concatenate((S_up0, S_up))/1e6, alpha=0.1, color=kwargs['color']) # TODO: remove overlap line

        ## Plot main curves 
        if normalized:
            ylabel = 'Cumulative aquatic vegetation fraction'
            denom = binned_lev_km2.loc[:, 'mean'].sum() # self.sumLev0() # note this won't include extrap lake fluxes if there is no self.extrapBinnedLSD, but the assert checks for this.
        else:
            ylabel = 'Cumulative aquatic vegetation (million $km^2$)'
            denom = 1
        plotECDFByValue(ax=ax, alpha=1, X=X, S=S/denom/1e6, normalized=False, reverse=reverse, **kwargs) # obs values
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0/denom/1e6, normalized=False, reverse=reverse, linestyle='dashed', **kwargs) # second plot is dashed for extrapolation
        ax.set_ylabel(ylabel)
        if plotLegend:
            ax.legend(loc='best')
        ax.set_ylim(0, ax.get_ylim()[1])
        return ax

    def plot_extrap_flux(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots fluxes from LSD using both measured and extrapolated values. 
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False).
        
        TODO: plot with CI
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        error_bars : boolean
            Whether to include error bars (not recommended, since this plots a CDF)
        **kwargs : get passed to plt.plot via plotECDFByValue

        returns: ax
        '''       
        assert 'est_g_day' in self.columns, "LSD doesn't have a flux estimate yet." 
        assert hasattr(self.extrapLSD, 'binnedG_day'), "binnedLSD doesn't have a flux estimate yet." 
        assert reverse==False, "No branch yet written for flux plots in reverse."
        ## Prepare values
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)
        
        # Create plot values for observed part
        means = self.extrapLSD.binnedG_day.loc[:,'mean']
        X, S = ECDFByValue(self.Area_km2, values_for_sum=self.est_g_day * 365.25 / 1e12, reverse=False) # scale to Tg / yr
        geom_means = np.array(list(map(interval_geometric_mean, means.index))) # take geom mean of each interval Try self.extrapLSD.binnedAreas.loc[:, 'mean'].index.get_level_values(0)
        # X = np.concatenate((geom_means, X))
        S += self.extrapLSD.sumFluxes()['mean'] # add to original cumsum
        # S = np.concatenate((np.cumsum(means)* 365.25 / 1e12, S)) # pre-pend the binned vals
        S0 = np.cumsum(means)* 365.25 / 1e12 # pre-pend the binned vals
        if not 'color' in kwargs:
            kwargs['color']='red'
        # TODO: rewrite vectorized to use cumsum over three indexes (mean, upper, lower). Right now, seems to not be equivalent... Try using .sum(level=1)
      
        if normalized:
            ylabel = 'Cumulative fraction of emissions'
            denom = self._total_flux_Tg_yr['mean'] # note this won't include extrap lake fluxes if there is no self.extrapBinnedLSD, but the assert checks for this.
        else:
            ylabel = 'Cumulative emissions (Tg/yr)'
            denom = 1

        ## Add error bars
        # if None:
        if error_bars == True and self.extrapLSD.hasCI_lsd:
            assert self.extrapLSD.hasCI_lsd, "error_bars is set, but extrapolated LSD has no CI (self.extrapLSD.hasCI_lsd is False)"
            
            ## as area plot: extrapolated flux
            S_low0 = np.cumsum(self.extrapLSD.binnedG_day.loc[:, 'lower'])* 365.25 / 1e12 / denom # btm section
            S_up0  = np.cumsum(self.extrapLSD.binnedG_day.loc[:, 'upper'])* 365.25 / 1e12 / denom
            
            ## as area plot: estimated flux from obs
            X_low, S_low = ECDFByValue(self.Area_km2, self.est_g_day * 365.25 / 1e12 / denom, reverse=False)
            S_low += self.extrapLSD.binnedG_day.loc[:, 'lower'].sum() * 365.25 / 1e12 / denom
            _, S_up = ECDFByValue(self.Area_km2, self.est_g_day * 365.25 / 1e12 / denom, reverse=False) # re-compute
            S_up += self.extrapLSD.binnedG_day.loc[:, 'upper'].sum() * 365.25 / 1e12 / denom
            ax.fill_between(np.concatenate((geom_means, X_low)), np.concatenate((S_low0, S_low)), np.concatenate((S_up0, S_up)), alpha=0.1, color=kwargs['color']) # TODO: remove overlap line

        ## Plot main curves

        plotECDFByValue(ax=ax, alpha=1, X=X, S=S/denom, normalized=False, reverse=reverse, **kwargs)
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0/denom, normalized=False, reverse=reverse, linestyle='dashed', **kwargs) # second plot is dashed for extrapolation
        ax.set_ylabel(ylabel)
        if plotLegend:
            ax.legend(loc='best')
        ax.set_ylim(0, ax.get_ylim()[1])
        return ax
    
    def plot_lev_cdf(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots CDF by LEV value 
   
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        error_bars : boolean
            Whether to include error bars (not recommended, since this plots a CDF)
        **kwargs : get passed to plt.plot via plotECDFByValue

        returns: ax

        '''           
        ## colors
        sns.set_palette("colorblind", len(self.regions()))

        ## plot
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)

        for var in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']:
            assert var in self.columns, f"LSD is missing {var} column, which is required to plot lev cdf."

        if normalized:
            plotECDFByValue(self.LEV_MEAN, ax=ax, alpha=0.4, color='black', label='All', reverse=reverse, normalized=normalized, **kwargs)
            ax.set_ylabel('Cumulative fraction of LEV')
        else:
            plotECDFByValue(self.LEV_MEAN * self.Area_km2, ax=ax, alpha=0.4, color='black', label='All', reverse=reverse, normalized=normalized, **kwargs)
            ax.set_ylabel('Cumulative LEV (million $km^2$)')
        ax.set_xlabel('LEV fraction')

        ## Legend
        if plotLegend:
            ax.legend(loc= 'center left', bbox_to_anchor=(1.04, 0.5)) # legend on right (see https://stackoverflow.com/a/43439132/7690975)

        return ax

    def plot_lev_cdf_by_lake_area(self, all=True, plotLegend=True, groupby_name=False, cdf=True, ax=None, normalized=True, reverse=False, error_bars=True, **kwargs):
        '''
        For LEV: Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False). Errors bars plots high and low estimates too.
        
        Parameters
        ----------
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        **kwargs : get passed to plt.plot via plotECDFByValue

        returns: ax
        '''
        if error_bars:
            assert_vars = ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']
        else:
            assert_vars = ['LEV_MEAN']
        for var in assert_vars:
            assert var in self.columns, f"LSD is missing {var} column, which is required to plot lev cdf."
        sns.set_palette("colorblind", len(self.regions()))

        ## plot
        if ax is None:
            _, ax = plt.subplots() # figsize=(5,3)

        ## Override default axes
        def makePlots(var, color='black'):
            '''Quick helper function to avoid re-typing code'''
            if normalized:
                values_for_sum = self[var]
                ylabel = 'Cumulative fraction of total LEV'
            else:
                values_for_sum = self[var] * self.Area_km2
                ylabel = 'Cumulative LEV (million $km^2$)'
            X, S = ECDFByValue(self.Area_km2, values_for_sum=values_for_sum, reverse=reverse)
            plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, color=color, label='All', normalized=normalized, reverse=reverse, **kwargs)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{self.name}: {self.sumLev(asFraction=True)['mean']:.2%} LEV")

        ## Run plots
        makePlots('LEV_MEAN')
        if error_bars:
            makePlots('LEV_MIN', 'grey')
            makePlots('LEV_MAX', 'grey')
        ## Legend
        if plotLegend:
            ax.legend(loc= 'center left', bbox_to_anchor=(1.04, 0.5)) # legend on right (see https://stackoverflow.com/a/43439132/7690975)

        return ax    

class BinnedLSD():
    '''This class represents lakes as area bins with summed areas.'''
    def __init__(self, lsd=None, btm=None, top=None, nbins=None, bins=None, normalize=True, compute_ci_lsd=False, compute_ci_lev=False, compute_ci_flux=False, binned_areas=None, extreme_regions_lsd=None, extreme_regions_lev=None):
        '''
        Bins will have left end closed and right end open.
        When creating this class to extrapolate a LSD, ensure that lsd is top-truncated to ~1km to reduce variability in bin sums. This chosen top limit will be used as the upper limit to the index region (used for normalization) and will be applied to the target LSD used for extrapolation.
        When creating from an lsd, give lsd, btm, top [, nbins, compute_ci_lsd] arguments. When creating from existing binned values (e.g.) from extrapolation, give btm, top, nbins, compute_ci_lsd and binnedAreas args.
        
        Parameters
        ----------
        lsd : LSD
            Lake-size distribution class to bin
        btm : float 
            Leftmost edge of bottom bin. Ignored if bins is given. 
        top : float
            Rightmost edge of top bin. Note: np.inf will be added to it to create one additional top bin for index region. Ignored if bins is given.
        nbins : int
            Number of bins to use for np.geomspace (not counting the top bin that goes to np.inf when 'lsd' arg is given). Can use either nbins or  bins.
        bins : array-like
            Alternative to specifiying number of bins: give actual bin edges. Top edge of infinity will be added.
        normalize : Boolean
            If True (default), normalized each bin by index region
        compute_ci_lsd : Boolean
            Compute confidence interval for lsd by breaking down by region. Function will always bin LSD (at least without CI)
        compute_ci_lev : Boolean
            Compute confidence interval for lev by breaking down by region. Function will always bin LEV (at least without CI)
        compute_ci_flux : Boolean
            Compute confidence interval for flux by multiplying by ci_lsd CI. Function will always bin LEV (at least without CI)
        binnedAreas : pandas.DataFrame
            Used for LSD.extrapolate(). Has structure similar to what is returned by self.binnedAreas if called with lsd argument. Format: multi-index with two columns, first giving bins, and second giving normalized lake area sum statistic (mean, upper, lower).
        extreme_regions_lsd : array-like
            List of region names to use for min/max area
        extreme_regions_lev : array-like
            List of region names to use for min/max LEV
        '''
        ## Assertions
        assert (nbins is None) or (bins is None), "Either nbins or bins must be specified, but not both."
        if (nbins is  None) and (bins is  None):
            nbins = 100 # set default 100 log-spaced bins if not specified
            assert btm is not None and top is not None, "If nbins is set to default, then btm and top must be specified"
        if bins is not None: # if nbins was none, but bins was specified
            nbins = len(bins) -1
            btm = bins[0]
            top = bins[-1]

        ## Common branch
        self.btmEdge = btm
        self.topEdge = top
        self.nbins = nbins
        self.isExtrapolated = False
        self.extreme_regions_lsd = None # init
        self.extreme_regions_lev = None # init
        self.isNormalized = False # init

        if lsd is not None: # sets binnedLSD from existing LSD
            assert (btm is not None and top is not None) or bins is not None, "if 'lsd' argument is given, so must be 'btm' and 'top' or 'bins'"
            assert binned_areas is None, "If 'lsd' is given, 'binned_areas' shouldn't be."

            attrs = lsd.get_public_attrs() # This ensures I can pass through all the attributes of parent LSD
            lsd = LSD(lsd, **attrs) # create copy
            if bins is not None:
                self.bin_edges = np.concatenate((bins, [np.inf])) # bins are user-specified
            else:
                self.bin_edges = np.concatenate((np.geomspace(btm, top, nbins+1), [np.inf])).round(6) # bins computed from nbins and edges
            self.area_bins = pd.IntervalIndex.from_breaks(self.bin_edges, closed='left')
            lsd['size_bin'] = pd.cut(lsd.Area_km2, self.area_bins, right=False)
            self.hasCI_lev = False # init

            ## # Boolean to determine branch for LEV
            hasLEV = 'LEV_MEAN' in lsd.columns
            hasFlux = 'est_g_day' in lsd.columns
            # hasLEV_CI = np.all([attr in lsd.columns for attr in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']])
            if not hasLEV:
                self.binnedLEV = None
            if not hasFlux:
                self.binnedG_day = None
                self.binnedMg_m2_day = None
            
            ## Bin
            group_sums = lsd.groupby(['size_bin']).Area_km2.sum(numeric_only=True) # These don't get lower/upper estimates for now
            group_counts = lsd.groupby(['size_bin']).Area_km2.count() # These don't get lower/upper estimates for now
            if compute_ci_lsd:
                assert extreme_regions_lsd is not None, "If compute_ci_lsd is True, extreme_regions_lsd must be provided"
                assert np.all([region in lsd.Region.unique() for region in extreme_regions_lsd]), "One region name in extreme_regions_lsd is not present in lsd."
                
                ## First, group by area bin and take sum and counts of each bin
                group_low_sums, group_high_sums = [lsd[lsd['Region']==region].groupby(['size_bin']).Area_km2.sum() for region in extreme_regions_lsd]
                self.hasCI_lsd = True
                self.extreme_regions_lsd = extreme_regions_lsd
            else:
                ## Create a series to emulate the structure I used to use to store region confidence intervals                  
                group_low_sums, group_high_sums = None, None
                self.hasCI_lsd = False

            ds = confidence_interval_from_extreme_regions(group_sums, group_low_sums, group_high_sums, name='Area_km2')
            self.binnedAreasNotNormalized = ds.copy()

            ## Normalize areas after binning
            if normalize:
                divisor = ds.loc[ds.index.get_level_values(0)[-1], :] # sum of lake areas in largest bin (Careful: can't be == 0 !!)
                self.isNormalized = True
            else:
                divisor =1
            if np.any(divisor == 0): warn("Careful, normalizing by zero.")
            ds /= divisor

            ## Save values
            self.binnedAreas = ds
            self.binnedCounts = group_counts

            ## bin LEV
            if hasLEV:
                group_means_lev = lsd.groupby(['size_bin']).LEV_MEAN.mean(numeric_only=True)
                if compute_ci_lev:
                    assert extreme_regions_lev is not None, "If compute_ci_lsd is True, and LSD has LEV, extreme_regions_lev must be provided"
                    assert np.all([region in lsd.Region.unique() for region in extreme_regions_lev]), "One region name in extreme_regions_lev is not present in lsd."
                    # group_means_lev_low = lsd.groupby(['size_bin']).LEV_MIN.mean(numeric_only=True)
                    # group_means_lev_high = lsd.groupby(['size_bin']).LEV_MAX.mean(numeric_only=True)
                    group_means_lev_low, group_means_lev_high = [lsd[lsd['Region']==region].groupby(['size_bin']).LEV_MEAN.mean(numeric_only=True) for region in extreme_regions_lev] # Here, my ref distrib has no LEV_MIN/MAX, since it is observed, so I use extreme reegions for uncertainty bounds                    
                    self.hasCI_lev = True
                    self.extreme_regions_lev = extreme_regions_lev
                    pass
                # if not hasLEV_CI and hasLEV: # e.g. if loading from UAVSAR
                #     self.binnedLEV = lsd.groupby(['size_bin']).LEV_MEAN.mean(numeric_only=True) 
                else:
                    group_means_lev_low, group_means_lev_high = None, None        
                    self.hasCI_lev = False

                ## Common branch: put in mean/upper/lower format
                ds_lev = confidence_interval_from_extreme_regions(group_means_lev, group_means_lev_low, group_means_lev_high, name='LEV_frac')
                self.binnedLEV = ds_lev

            ## bin flux
            if hasFlux:
                ds_g_day_from_sum = lsd.groupby(['size_bin']).est_g_day.sum(numeric_only=True) # method #1: sum g/day for each lake   
                self.binnedG_day = confidence_interval_from_extreme_regions(ds_g_day_from_sum, None, None, name='est_g_day') # Save for potential comparison?
                if compute_ci_flux: # CI
                    assert compute_ci_lsd is not None, "If compute_ci_flux is True, compute_ci_lsd must be provided"
                    # group_means_flux0, group_means_flux_low0, group_means_flux_high0 = [lsd.groupby(['size_bin']).est_mg_m2_day.mean(numeric_only=True) * self.binnedAreas.loc[:,stat] * 1e3 for stat in ['mean', 'lower', 'upper']] # method #2: would have units of g / day (don't multiply by area yet). 
                    group_means_flux, group_means_flux_low, group_means_flux_high = [lsd.groupby(['size_bin']).est_mg_m2_day.mean(numeric_only=True) for stat in ['mean', 'lower', 'upper']] # Low/mean/high are all the same on a per-area basis!               
                    self.hasCI_flux = True
                    pass
                else: # This branch is silly, because neither branch computes a meaningful confidence interval. See self.PredictFlux() for actual conversion to CI based on CI of areas and given per-area fluxes
                    group_means_flux = lsd.groupby(['size_bin']).est_mg_m2_day.mean(numeric_only=True) # Low/mean/high are all the same on a per-area basis!
                    group_means_flux_low, group_means_flux_high = None, None
                    self.hasCI_flux = False
                self.binnedMg_m2_day = confidence_interval_from_extreme_regions(group_means_flux, group_means_flux_low, group_means_flux_high, name='est_g_day')

            for attr in ['isTruncated', 'truncationLimits', 'name', 'size_bin']: # copy attribute from parent LSD (note 'size_bin' is added in earlier this method)
                setattr(self, attr, getattr(lsd, attr))
            
            ## Check
            if self.binnedCounts.values[0] == 0:
                warn('The first bin has count zero. Did you set the lowest bin edge < the lower truncation limit of the dataset?')   
                    
        else: # used for area extrapolation
            assert nbins is not None and compute_ci_lsd is not None, "If 'binned_areas' argument is given, so must be 'nbins', and 'compute_ci_lsd'."
            self.bin_edges = 'See self.refBinnedLSD'
            self.area_bins = 'See self.refBinnedLSD'
            self.hasCI_lsd = compute_ci_lsd # retain from arg
            self.hasCI_lev = compute_ci_lev # retain
            self.binnedAreas = binned_areas # retain from arg
            self.binnedCounts = 'See self.refBinnedLSD'
            self.binnedLEV = 'See self.refBinnedLSD'
            self.binnedG_day = None
            self.binnedMg_m2_day = None
            self.hasCI_flux = None
        
        ## More common args at end
        self.isBinned = True

        pass
    
    # def normalize(self):
    #     '''Divide bins by the largest bin.'''
    #     pass
    #     self.isNormalized = True

    def FromExtrap():
        '''Creates a class similar to binnedLSD,'''
        pass
    
    def get_public_attrs(self):
        return public_attrs(self)
    
    def sumAreas(self, ci=False):
        '''
        Sum the areas within the dataframe self.binnedAreas.

        If ci==True, returns the mean, lower and upper estimate. Otherwise, just the mean.
        
        ci : Boolean
            Whether to output the lower and upper confidence intervals.
        '''
        if self.isNormalized:
            warn('Careful: binnedLSD is normalized, and you may be summing the top bin that gives the index region.')

        if self.hasCI_lsd:
            if ci:
                return self.binnedAreas.loc[:,'mean'].sum(), self.binnedAreas.loc[:,'lower'].sum(), self.binnedAreas.loc[:,'upper'].sum()
            else:
                return self.binnedAreas.loc[:,'mean'].sum()
        else: # no .hasCI_lsd
            if ci:
                raise ValueError('BinnedLSD doesnt have a confidence interval, so it cant be included in sum.')
            else:
                return self.binnedAreas.sum()
    
    def plot(self, show_rightmost=False, as_lineplot=False, ax=None, as_cumulative=False):
        '''
        To roughly visualize bins.
        
        show_rightmost : Boolean
            Show remainder bin on right (ignored if self.isExtrapolated==True)
        '''
        if as_cumulative == True:
            assert as_lineplot==True, "If plotting as cumulative plot, be sure to indicate as_lineplot=True"
        binned_areas = self.binnedAreas.copy() # create copy to modify
        diff=0

        ## Remove last bin, if desired
        if self.isExtrapolated==False:
            if show_rightmost == False: # if I need to cut off rightmost bin
                # if self.hasCI_lsd: # sloppy fix
                #     binned_areas = pd.Series({'mean': binned_areas.values[0].iloc[:-1],
                #                                 'lower': binned_areas.values[1].iloc[:-1],
                #                                 'upper': binned_areas.values[2].iloc[:-1]})
                # else:
                binned_areas.drop(index=binned_areas.index.get_level_values(0)[-1], level=0, inplace=True)
            else:
                diff+=1 # subtract from number of bin edges to get plot x axis
        
        ## Plot
        if ax is None:
            _, ax = plt.subplots()
        # plt.bar(self.bin_edges[:-1], binned_areas)
        if as_lineplot:
            assert isinstance(binned_areas, pd.Series), "As written, BinnedLSD.plot requires pd.Series argument is as_lineplot==True."
            xlabel = 'Lake area ($km^2$)'

            ## Normalize by total to make true PDF
            binned_areas /= binned_areas[binned_areas.index.get_level_values(1) == 'mean'].sum()
            geom_means = np.array(list(map(interval_geometric_mean, binned_areas[binned_areas.index.get_level_values(1) == 'mean'].index.get_level_values(0)))) # x-vector
            if as_cumulative:
                data = np.cumsum(binned_areas[binned_areas.index.get_level_values(1) == 'mean'])
            else:
                data = binned_areas[binned_areas.index.get_level_values(1) == 'mean']
            ax.plot(geom_means, data)
        else:
            xlabel = 'Bin number'
            
            ## Convert confidence interval vals to anomalies
            yerr = binnedVals2Error(binned_areas, self.nbins+diff)
            ax.bar(range(self.nbins), binned_areas.loc[:, 'mean'], yerr=yerr, color='orange') 

        
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        if self.isNormalized:
            ax.set_ylabel('Fraction of large lake area')
        else:
            ax.set_ylabel('km2')
        return

    def plotLEV():
        '''Placeholder'''
        pass

    def predictFlux(self, model):
        '''
        Predict methane flux based on area bins and temperature. Assumes temp is constant for all bins.
        
        TODO: 
            * Use temp as a df variable, not common attribute
            * Lazy algorithm- only compute if self._Total_flux_Tg_yr not present
        Parameters
        ----------
        model : statsmodels
        coeff : array-like
            list of model coefficients
        returns: ax
        '''
        assert hasattr(self, 'Temp_C'), "Binned LSD needs a Temp_C attribute in order to predict flux."
        assert self.isNormalized == False, "Binned LSD is normalized so values will be unitless for area."
        means = self.binnedAreas.loc[:, 'mean']
        geom_mean_areas = np.array(list(map(interval_geometric_mean, means.index)))
        
        ## Flux (areal, mgCH4/m2/day)
        est_mg_m2_day = 10**(model.params.Intercept +
        model.params['np.log10(SA)'] * np.log10(geom_mean_areas) 
        + model.params['TEMP'] * self.Temp_C) - 0.01 # jja, ann, son, mam # no uncertainty yet

        ## Flux (flux rate, gCH4/day)
        est_g_day_mean, est_g_day_low, est_g_day_high = [est_mg_m2_day * self.binnedAreas.loc[:,stat] * 1e3 for stat in ['mean', 'lower', 'upper']] # * 1e6 / 1e3 # (convert km2 -> m2 and mg -> g)
                    # group_means_flux, group_means_flux_low, group_means_flux_high = [lsd.groupby(['size_bin']).est_mg_m2_day.mean(numeric_only=True) for stat in ['mean', 'lower', 'upper']] # Low/mean/high are all the same on a per-area basis!
        est_g_day = confidence_interval_from_extreme_regions(est_g_day_mean, est_g_day_low, est_g_day_high, name='est_g_day')
        self._total_flux_Tg_yr = est_g_day.groupby('stat').sum() * 365.25 / 1e12 # see Tg /yr

        ## Add attrs
        self.binnedMg_m2_day = confidence_interval_from_extreme_regions(pd.Series(est_mg_m2_day, index=est_g_day.loc[:,'mean'].index), None, None, name='est_mg_m2_day') # in analogy with binnedAreas and binnedCounts
        self.binnedG_day = est_g_day # in analogy with binnedAreas and binnedCounts # HERE: is this updated value the same as the one computed in BinnedLSD.init?

        # return self._Total_flux_Tg_yr
        return
    
    def sumFluxes(self):
        '''Convenience function for symmetry with sumAreas. Returns total value in Tg/yr.'''
        return self._total_flux_Tg_yr
    
    def sumLev(self, asFraction=False):
        '''
        Weighted mean of LEV fraction by lake area within bin.
        
        Area used for multiplication is the central estimate, not the CI, even when multiplying by LEV CI.
        '''
        summed_lev = (self.binnedLEV * self.binnedAreas.loc[:,'mean']).groupby(level=1).sum() # km2
        if asFraction:
            result = summed_lev / self.sumAreas(ci=False)
        else:
            result = summed_lev
        return result


def runTests():
    '''Practicing loading and functions/methods with/without various arguments.
    Can pause and examine classes to inspect attributes.'''
    # ## Testing from shapefile
    # lsd_from_shp = LSD.from_shapefile('/mnt/f/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/HL_Sweden_md.shp', area_var='Lake_area', idx_var='Hylak_id', name='HL', region_var=None)
    # lsd_from_shp = LSD.from_shapefile('/mnt/f/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/HL_Sweden_md.shp', area_var='Lake_area')
    # lsd_from_shp = LSD.from_shapefile('/mnt/f/PeRL/PeRL_waterbodymaps/waterbodies/arg00120110829_k2_nplaea.shp', area_var='AREA', idx_var=None, name='yuk00120090812', region_var=None, _areaConversionFactor=1e6)
    # print('\tPassed load from shapefile.')
    
    ## Testing from gdf
    # gdf = pyogrio.read_dataframe('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp', read_geometry=True, use_arrow=True)
    # lsd_from_gdf = LSD(gdf, area_var='Area', name='CIR', region_var='Region4')
    # regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    # lsd_from_gdf = LSD(gdf, area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')
    # print('\tPassed load from gdf.')
    
    # ## Loading from dir
    # exclude = ['arg0022009xxxx', 'fir0022009xxxx', 'hbl00119540701','hbl00119740617', 'hbl00120060706', 'ice0032009xxxx', 'rog00219740726', 'rog00220070707', 'tav00119630831', 'tav00119750810', 'tav00120030702', 'yak0012009xxxx', 'bar00120080730_qb_nplaea.shp']
    # lsd_from_dir = LSD.from_paths('/mnt/f/PeRL/PeRL_waterbodymaps/waterbodies/y*.shp', area_var='AREA', name='perl', _areaConversionFactor=1000000, exclude=exclude)

    # ## Test concat
    # lsd_concat = LSD.concat((lsd_from_shp, lsd_from_gdf))
    # lsd_concat = LSD.concat((lsd_from_shp, lsd_from_gdf), broadcast_name=True)
    # print('\tPassed concat.')

    # ## Test truncate
    # lsd_concat.truncate(0.01, 20)
    # print('\tPassed truncate.')
    # pass

    # ## Test compute area from geometry

    # ## Test area fraction
    # lsd_from_gdf.area_fraction(1)
    # print('\tPassed area_fraction.')

    ## Load with proper parameters
    # lsd_hl = LSD.from_shapefile('/mnt/f/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/HL_Sweden_md.shp', area_var='Lake_area', idx_var='Hylak_id', name='HL', region_var=None)
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lsd_cir = LSD.from_shapefile('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp', area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

    ## Test binned PDF
    # plotEPDFByValue(lsd_cir.Area_km2)

    ####################################
    ## LEV Tests
    ####################################
    
    ref_names = ['CSB', 'CSD', 'PAD', 'YF']
    lsd_lev_cat, ref_dfs = loadUAVSAR(ref_names)

    ## Test LEV estimate: Load UAVSAR/GSW overlay stats
    print('Load HL with joined occurrence...')
    # lsd_hl_oc = pyogrio.read_dataframe('/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_full.shp', read_geometry=False, use_arrow=False, max_features=1000) # load shapefile with full histogram of zonal stats occurrence values
    lsd_hl_oc = pd.read_csv('/mnt/g/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full.csv.gz', compression='gzip', nrows=10000) # read smaller csv gzip version of data.

    extreme_regions_lev_for_extrap = ['CSD','PAD']
    lev = computeLEV(lsd_hl_oc, ref_dfs, ref_names, extreme_regions_lev=extreme_regions_lev_for_extrap) # use same regions for extrap as for estimate
    lsd_lev = LSD(lev, area_var='Lake_area', idx_var='Hylak_id')

    ## Test plot LEV  CDF
    lsd_lev.plot_lev_cdf()

    ## Test plot LEV CDF by lake area
    lsd_lev.plot_lev_cdf_by_lake_area()
    print(f"Mean LEV: {lsd_lev.sumLev(asFraction=True)['mean']:.2%}")

    ## Test binned LEV HL LSD (won't actually use this for analysis)
    # extreme_regions_lsd = ['Tuktoyaktuk Peninsula', 'Prairie Potholes North'] # tmp
    # binned = BinnedLSD(lsd_lev.truncate(0.5,1), 0.5, 1000, compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd) # compute_ci_lsd=False will disable plotting CI.

    ## Test 1: Bin the reference UAVSAR LEV LSDs
    # lsd_lev_binneds = []
    # for i, lsd_lev in enumerate(lsd_levs):
    #     lsd_lev_binneds.append(BinnedLSD(lsd_lev, 0.000125, 0.5).binnedLEV)
    # pd.DataFrame(lsd_lev_binneds).T

    ## Test 2: Concat all ref LEV distributions and bin
    # lsd_lev_cat = LSD.concat(lsd_levs, broadcast_name=True)
    # def ci_from_named_regions(LSD, regions):
    binned_lev = BinnedLSD(lsd_lev_cat, 0.0001, 0.5, compute_ci_lev=True, extreme_regions_lev=extreme_regions_lev_for_extrap) # 0.000125 is native

    ####################################
    ## LSD/bin/extrap Tests
    ####################################

    ## Test binnedLSD
    extreme_regions_lsd = ['Tuktoyaktuk Peninsula', 'Prairie Potholes North'] # tmp
    # binned = BinnedLSD(lsd_cir.truncate(0.0001,1), 0.0001, 0.5, compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd) # compute_ci_lsd=False will disable plotting CI.
    binned = BinnedLSD(lsd_cir.truncate(0.0001,1), 0.0001, 0.5, compute_ci_lsd=False) # compute_ci_lsd=False will disable plotting CI.
    binned.plot()
    binned.plot(as_lineplot=True)

    ## Test binnedLSD with edges specified
    binned_manual = BinnedLSD(lsd_cir.truncate(0.0001,1), 0.0001, 0.5, bins=[0.001, 0.1, 0.5], compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd) # compute_ci_lsd=False will disable plotting CI.
    binned_manual.plot()

    ## Test binnedLSD with fluxes
    model = loadBAWLD_CH4()
    lsd_cir['Temp_C'] = 10 # placeholder, required for prediction 
    lsd_cir.predictFlux(model, includeExtrap=False)
    binned = BinnedLSD(lsd_cir.truncate(0.0001,1), 0.0001, 0.5, compute_ci_lsd=True, compute_ci_flux=True, extreme_regions_lsd=extreme_regions_lsd) # compute_ci_lsd=False will disable plotting CI.

    ## Test extrapolate on small data
    # lsd_hl_trunc = lsd_hl.truncate(0.1, np.inf, inplace=False) # Beware chaining unless I return a new variable.
    # lsd_hl_trunc.extrapolate(binned, binned_lev)

    ## Test extrapolate on small data with binned LEV
    lsd_hl_trunc = lsd_lev.truncate(0.5, np.inf, inplace=False)
    lsd_hl_trunc.extrapolate(binned, binned_lev)

    ## Test mean lev with extrap
    print(f"Mean LEV: {lsd_hl_trunc.sumLev(asFraction=True)['mean']:.2%}")

    ## Compare extrapolated sums
    lsd_hl_trunc.extrapLSD.sumAreas()
    lsd_hl_trunc.sumAreas()

    ## Test plot extrap LEV
    # ax = lsd_hl_trunc.plot_extrap_lsd(normalized=False, error_bars=True, color='blue')
    # ax2=ax.twinx()
    # lsd_hl_trunc.plot_extrap_lev(ax=ax2, error_bars=True, color='green')
    # ymin, ymax = ax.get_ylim()
    # ax2.set_ylim([ymin, ymax])
    # plt.tight_layout()

    ## Compare extrapolated area fractions
    # frac = lsd_hl_trunc.extrapolated_area_fraction(lsd_cir, 0.0001, 0.01)
    # print(frac)
    # # lsd_hl_trunc.extrapolated_area_fraction(lsd_cir, 0.00001, 0.01) # test for bin warning
    # # lsd_hl_trunc.extrapolated_area_fraction(lsd_cir, 0.0001, 1)# Test for limit error

    ## Plot
    # lsd_hl_trunc.extrapLSD.plot()
    # ax = lsd_hl_trunc.plot_lsd(reverse=False, normalized=True)
    # lsd_hl_trunc.plot_extrap_lsd(normalized=True, error_bars=False, reverse=False) # ax=ax, 
    # lsd_hl_trunc.plot_extrap_lsd(normalized=False, error_bars=True, reverse=False) # ax=ax, 

    ####################################
    ## Flux Tests
    ####################################

    ## Load climate
    bawld_join_clim_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1___Shapefile_jn_clim.csv'
    gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    df_clim = pd.read_csv(bawld_join_clim_pth)
    gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio' )   
    df_clim = df_clim.merge(gdf_bawld[['Cell_ID', 'Shp_Area']], how='left', on='Cell_ID')
    
    ## Test flux prediction from observed lakes
    model = loadBAWLD_CH4()
    # lsd_hl_trunc['Temp_C'] = 10 # placeholder until I load from file, required for prediction 
    temp = lsd_hl_trunc.merge(df_clim, how='left', left_on='idx_unamed', right_on='Cell_ID') # some rows don't merge, I think bc I think idx_unamed is nto for HL... just use as example
    temperature_metric = 'jja'
    lsd_hl_trunc['Temp_C'] = temp[temperature_metric]
    lsd_hl_trunc.predictFlux(model, includeExtrap=False)

    ## Test flux prediction from extrapolated lakes
    lsd_hl_trunc.extrapLSD.Temp_C = np.average(lsd_hl_trunc.Temp_C, weights=lsd_hl_trunc.Area_km2) # Use lake-area-weighted average temp as temp for extrapolated lakes, since we don't know where they are located
    lsd_hl_trunc.extrapLSD.predictFlux(model)

    ## Test combined prediction
    lsd_hl_trunc.predictFlux(model, includeExtrap=True)

    ## Test plot fluxes
    # lsd_hl_trunc.plot_flux(reverse=False, normalized=True, all=False)
    # lsd_hl_trunc.plot_flux(reverse=False, normalized=False, all=False)

    ## Test plot extrap fluxes
    lsd_hl_trunc.plot_extrap_flux(reverse=False, normalized=False, error_bars=True)
    lsd_hl_trunc.plot_extrap_flux(reverse=False, normalized=True, error_bars=True)

    pass

if __name__=='__main__':

    ## Testing mode or no.
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False,
                        help="Whether to run in test mode or not (default=False)")
    args = parser.parse_args()
    if args.test == 'True':
        print('Test mode.')
        runTests()
        exit()

    ## I/O
    tb_dir = '/mnt/g/Ch4/area_tables'

    ## Common
    temperature_metric = 'jja'

    ## BAWLD domain
    dataset = 'HL'
    roi_region = 'BAWLD'
    gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    gdf_HL_jn_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned.shp' # HL clipped to BAWLD # note V4 is not joined to BAWLD yet
    hl_area_var='Shp_Area'
    hl_join_clim_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v3/joined_climate/run00/HL_clim_full.csv'
    bawld_join_clim_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1___Shapefile_jn_clim.csv'
    hl_nearest_bawld_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned_jnBAWLD.shp' # HL shapefile with ID of nearest BAWLD cell (still uses V3)

    ## BAWLD-NAHL domain
    # dataset = 'HL'
    # roi_region = 'WBD_BAWLD'
    # gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1_clipped_to_WBD.shp'
    # gdf_HL_jn_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_binned_jnBAWLD_roiNAHL.shp' # HL clipped to BAWLD and WBD
    # hl_area_var='Shp_Area'

    ## BAWLD domain (Sheng lakes)
    # dataset = 'Sheng'
    # roi_region = 'BAWLD'
    # gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    # gdf_Sheng_pth = '/mnt/g/Other/Sheng-Arctic-lakes/edk_out/clips/UCLA_ArcticLakes15_BAWLD.shp' # HL clipped to BAWLD
    # sheng_area_var='area'

    ## dynamic vars
    # analysis_dir = os.path.join('/mnt/g/Ch4/Area_extrapolations','v'+str(version))
    # area_extrap_pth = os.path.join(analysis_dir, dataset+'_sub'+roi_region+'_extrap.csv')
    # os.makedirs(analysis_dir, exist_ok=True)

    ## Loading from CIR gdf 
    print('Load HR...')
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta',
               'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River',
               'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North',
               'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lsd_cir = LSD.from_shapefile('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp',
                area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

    ## Loading PeRL LSD
    perl_exclude = ['arg0022009xxxx', 'fir0022009xxxx', 'hbl00119540701','hbl00119740617',
                    'hbl00120060706', 'ice0032009xxxx', 'rog00219740726', 'rog00220070707',
                    'tav00119630831', 'tav00119750810', 'tav00120030702', 'yak0012009xxxx',
                    'bar00120080730_qb_nplaea.shp']
    lsd_perl = LSD.from_paths('/mnt/f/PeRL/PeRL_waterbodymaps/waterbodies/*.shp', area_var='AREA', name='perl', _areaConversionFactor=1000000, exclude=perl_exclude)

    ## Loading from Mullen
    lsd_mullen = LSD.from_paths('/mnt/g/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/*_3Y_lakes-and-ponds.zip', _areaConversionFactor=1000000, name='Mullen', computeArea=True) # '/mnt/g/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/[A-Z][A-Z]_08*.zip'

    ## Combine PeRL and CIR and Mullen
    lsd = LSD.concat((lsd_cir, lsd_perl, lsd_mullen), broadcast_name=True, ignore_index=True)

    # ## plot
    # lsd.truncate(0.0001, 10).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=True, plotLabels=False)
    # lsd.truncate(0.0001, 10).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

    # ## Plot just CIR
    # lsd_cir.truncate(0.0001, 5).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=False)
    # lsd_cir.truncate(0.0001, 5).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

    # ## Plot just PeRL
    # lsd_perl.truncate(0.0001, 5).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=False)
    # lsd_perl.truncate(0.0001, 5).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

    # ## Compute extreme regions and save to spreadsheet
    # df_regions = regionStats(lsd)
    # df_regions.to_csv(os.path.join(tb_dir, 'region_stats.csv'))

    # ## YF compare
    # LSD(lsd.query("Region=='YF_3Y_lakes-and-ponds' or Region=='Yukon Flats Basin'"), name='compare').plot_lsd(all=False, plotLegend=True, reverse=False, groupby_name=False)

    ####################################
    ## LEV Analysis
    ####################################

    ## Load csv and shapefiles
    ref_names = ['CSB', 'CSD', 'PAD', 'YF']
    extreme_regions_lev_for_extrap = ['CSD', 'PAD']
    lsd_lev_cat, ref_dfs = loadUAVSAR(ref_names)
    
    ## Create binned ref LEV distribution from UAVSAR
    binned_lev = BinnedLSD(lsd_lev_cat, 0.0001, 0.5, compute_ci_lev=True, extreme_regions_lev=extreme_regions_lev_for_extrap) # 0.000125 is native

    ## LEV estimate: Load UAVSAR/GSW overlay stats
    print('Load HL with joined occurrence...')
    # lsd_hl_oc = pyogrio.read_dataframe('/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_full.shp', read_geometry=False, use_arrow=True) # load shapefile with full histogram of zonal stats occurrence values # outdated version
    lsd_hl_oc = pd.read_csv('/mnt/g/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full.csv.gz', compression='gzip', low_memory=False) # read smaller csv gzip version of data.
    lev = computeLEV(lsd_hl_oc, ref_dfs, ref_names, extreme_regions_lev=extreme_regions_lev_for_extrap) # use same extreme regions for est as for extrap
    lsd_hl_lev = LSD(lev, area_var='Lake_area', idx_var='Hylak_id', name='HL')

    # ## Plot LEV CDF by lake area (no extrap) and report mean LEV fraction
    # lsd_hl_lev.plot_lev_cdf_by_lake_area()
    # lsd_hl_lev.plot_lev_cdf_by_lake_area(normalized=False)

    ####################################
    ## Climate Analysis: join in temperature
    ####################################
    print('Loading BAWLD and climate data...')
    # df_clim = pd.read_csv(hl_join_clim_pth) # Index(['Unnamed: 0', 'BAWLDCell_', 'Hylak_id', 'Shp_Area', 'geometry','index_right', 'id', 'area', 'perimeter', 'lat', 'lon', 'djf', 'mam', 'jja', 'son', 'ann'],
    df_clim = pd.read_csv(bawld_join_clim_pth)
    gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio')
    df_clim = df_clim.merge(gdf_bawld[['Cell_ID', 'Shp_Area']], how='left', on='Cell_ID')

    ## Next, load HL with nearest BAWLD:
    df_hl_nearest_bawld = pyogrio.read_dataframe(hl_nearest_bawld_pth, read_geometry=False) # Ignore the 0-5 etc. joined Oc columns because they are per grid cell
    df_hl_nearest_bawld = df_hl_nearest_bawld.groupby('Hylak_id').first().reset_index() # take only first lake (for cases where lake is equidistant from multiple cells)
    lsd_hl_lev_m = lsd_hl_lev.merge(df_hl_nearest_bawld[['Hylak_id', 'BAWLD_Cell']], how='left', left_on='idx_HL', right_on='Hylak_id').drop(columns='Hylak_id') # Need to create new var because output of merge is not LSD # '0-5', '5-50', '50-95', '95-100', 'Class_sum', 'BAWLD_Long', 'BAWLD_Lat', 

    ## Use BAWLD Cell to join in temp (e.g. "double-join")
    temperatures = lsd_hl_lev_m[['idx_HL', 'BAWLD_Cell']].merge(df_clim, how='left', left_on='BAWLD_Cell', right_on='Cell_ID').drop(columns=['Cell_ID', 'Shp_Area']) # some rows don't merge, I think bc I think idx_unamed is nto for HL... just use as example
    lsd_hl_lev['Temp_C'] = temperatures[temperature_metric]
    
    ## Compute cell-area-weighted average of climate as FYI
    # print(f'Mean JJA temperature across {roi_region} domain: {np.average(df_clim.jja, weights=df_clim.Shp_Area)}')
    # months = ['ann','djf','mam','jja','son']
    # print(pd.DataFrame(np.average(df_clim[months], weights=df_clim.Shp_Area, axis=0), index=months))

    ####################################
    ## Map Analysis
    ####################################

    # ## Rescale to km2
    # for col in ['LEV_MEAN', 'LEV_MIN','LEV_MAX']:
    #     lsd_hl_lev[col+'_km2'] = lsd_hl_lev[col] * lsd_hl_lev['Area_km2'] # add absolute area units
    # # lsd_hl_lev.to_csv('/mnt/g/Ch4/GSW_zonal_stats/HL/v5/HL_BAWLD_LEV.csv')

    # ## Join df with LEV to df with nearest BAWLD (quicker than in Arcgis)
    # if 'df_hl_nearest_bawld' not in locals():
    #     raise ValueError("Nead to load df_hl_nearest_bawld ")
    # ## Groupby bawld cell and compute sum of LEV and weighted avg of LEV
    # df_bawld_sum_lev = lsd_hl_lev.groupby('BAWLD_Cell').sum(numeric_only=True) # Could add Occ

    # ## Rescale back to LEV percent (lf lake and of grid cell) as well
    # for col in ['LEV_MEAN', 'LEV_MIN','LEV_MAX']:
    #     df_bawld_sum_lev[(col+ '_km2').replace('_km2', '_frac')] = df_bawld_sum_lev[col+ '_km2'] / df_bawld_sum_lev['Area_km2'] # add absolute area units
    #     df_bawld_sum_lev.drop(columns=col, inplace=True) # remove summed means, which are meaningless
    # df_bawld_sum_lev.drop(columns=['idx_HL', 'Hylak_id'], inplace=True) # meaningless

    # ## Join to BAWLD 
    # gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio' )
    # gdf_bawld_sum_lev = df_bawld_sum_lev.merge(gdf_bawld, how='left', right_on='Cell_ID', left_index=True) # [['Cell_ID', 'Shp_Area']]
    # ## Could also join % Occ to lsd_hl_lev

    # for col in ['LEV_MEAN', 'LEV_MIN','LEV_MAX']:
    #     gdf_bawld_sum_lev[(col+ '_km2').replace('_km2', '_grid_frac')] = gdf_bawld_sum_lev[col+ '_km2'] / gdf_bawld_sum_lev['Shp_Area'] # add cell LEV fraction
     
    # ## and write out
    # gpd.GeoDataFrame(gdf_bawld_sum_lev).to_file('/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/joined_lev/BAWLD_V1_LEV.shp')
    
    # ## Stats from BAWLD LEV
    # s=gdf_bawld_sum_lev.sum()
    # print(f"BAWLD domain is {s.LEV_MEAN_km2/s.Shp_Area:0.4%} [{s.LEV_MIN_km2/s.Shp_Area:0.4%}-{s.LEV_MAX_km2/s.Shp_Area:0.4%}] lake vegetation.")
    # print(f"BAWLD domain is {s.WET/s.Shp_Area:0.4%} [{s.WET_L/s.Shp_Area:0.4%}-{s.WET_H/s.Shp_Area:0.4%}] wetlands.")
    # print(f"BAWLD domain: {s.WET/1e6:0.3} [{s.WET_L/1e6:0.3}-{s.WET_H/1e6:0.3}] wetlands.")
    # print(f"BAWLD domain: {s.LEV_MEAN_km2/1e6:0.3} [{s.LEV_MIN_km2/1e6:0.3}-{s.LEV_MAX_km2/1e6:0.3}] lake vegetation.")

    ## What percentage of HL is 0-50 Oc bin?
    # print('Load HL...')
    # # lsd_hl = LSD.from_shapefile(gdf_HL_jn_pth, area_var=hl_area_var, idx_var='Hylak_id', name='HL', region_var=None) # Need to load version with joined in Oc stats per lake
    # use df_hl_nearest_bawld

    ####################################
    ## Holdout Analysis
    ####################################

    ## Load measured holdout LEV dataset
    # a_lev_measured = gpd.read_file('/mnt/g/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/YF_train_holdout/zonal_hist_w_UAVSAR/YFLATS_190914_mosaic_rcls_brn_zHist_UAV_holdout_LEV.shp', engine='pyogrio')
    # a_lev = np.average(a_lev_measured.A_LEV, weights=a_lev_measured.Lake_area)
    # print(f'Measured A_LEV in holdout ds: {a_lev:0.2%}')

    ## Compare to holdout dataset
    # val_lakes_idx = [368946, 365442, 362977,362911,362697,362623,362193,361869,359283] # by Hylak_ID
    # lev_holdout = lsd_hl_lev[np.isin(lsd_hl_lev.idx_HL, val_lakes_idx)]
    # a_lev_pred = np.average(lev_holdout[['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']], axis=0, weights=lev_holdout.Area_km2)
    # print(f'Predicted A_LEV in holdout ds: {a_lev_pred[0]:0.2%} ({a_lev_pred[1]:0.2%}, {a_lev_pred[2]:0.2%})')
    # print(f'Correlation: {np.corrcoef(lev_holdout.LEV_MEAN, a_lev_measured.A_LEV)[0,1]:0.2%}')
    # print(f'RMSE: {mean_squared_error(lev_holdout.LEV_MEAN, a_lev_measured.A_LEV, squared=False)}')
    # plt.scatter(lev_holdout.LEV_MEAN, a_lev_measured.A_LEV)
    # plt.xlabel('Predicted LEV (%)')
    # plt.ylabel('Measured LEV (%)')

    ## Load WBD
    if roi_region == 'WBD_BAWLD':
        print('Load WBD...')
        lsd_wbd = LSD.from_shapefile('/mnt/g/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', area_var='Area', name='WBD', idx_var='OBJECTID')
        lsd_wbd.truncate(0.001, inplace=True)

    ## Plot WBD
    # lsd_wbd.plot_lsd(reverse=False, all=False)

    ## Combine WBD with HR dataset over small lakes for plotting comparison
    # setattr(lsd, 'name', 'HR datasets')
    # lsd['Region'] = 'NaN' # Hot fix to tell it not to plot a curve for each region # ***This is the buggy line!!!! Uncomment to get good curves, but no error bars if I haven't set compute_ci = False.
    # lsd_compare = LSD.concat((lsd.truncate(0.001, 50), lsd_wbd.truncate(0.001, 50)), broadcast_name=True, ignore_index=True)
    # lsd_compare.plot_lsd(all=False, plotLegend=True, reverse=False, groupby_name=True)

    ## Estimate area fraction
    # lsd_wbd.area_fraction(0.1)
    # lsd_wbd.area_fraction(0.01)
    # lsd_wbd.area_fraction(0.001)

    # lsd.area_fraction(0.1)
    # lsd.area_fraction(0.01)
    # lsd.area_fraction(0.001)  

    ## Load hydrolakes. if not using lsd_hl_lev
    # print('Load HL...')
    # lsd_hl = LSD.from_shapefile(gdf_HL_jn_pth, area_var=hl_area_var, idx_var='Hylak_id', name='HL', region_var=None)

    ## Load sheng
    # print('Load Sheng...')
    # lsd_hl = LSD.from_shapefile(gdf_Sheng_pth, area_var=sheng_area_var, idx_var=None, name='Sheng', region_var=None)

    ## Extrapolate
    extreme_regions_lsd = ['Tuktoyaktuk Peninsula', 'sur00120130802_tsx_nplaea'] #['Tuktoyaktuk Peninsula', 'Peace-Athabasca Delta']
    tmin, tmax = (0.0001,5) # Truncation limits for ref LSD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
    emax = 0.5 # Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
    binned_ref = BinnedLSD(lsd.truncate(tmin, tmax), tmin, emax, compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd) # reference distrib (try 5, 0.5 as second args)
    lsd_hl_trunc = lsd_hl_lev.truncate(emax, np.inf) # Beware chaining unless I return a new variable. # Try 0.1
    lsd_hl_trunc.extrapolate(binned_ref, binned_lev)
    meas=lsd_hl_lev.sumAreas(includeExtrap=False)
    extrap=lsd_hl_trunc.sumAreas()

    limit=0.01
    frac = lsd_hl_trunc.extrapolated_area_fraction(lsd, 0.0001, limit)
    print(f'Total measured lake area in {roi_region} domain: {meas:,.0f} km2')
    print(f'Total extrapolated lake area in {roi_region} domain: {extrap:,.0f} km2')
    print(f'{1-(meas / extrap):.1%} of lake area is < 0.1 km2.')
    print(f'{frac:.1%} of lake area is < {limit} km2.')
    print(f'{lsd_hl_trunc.extrapolated_area_fraction(lsd, 0.0001, 0.001):.1%} of lake area is < 0.001 km2.')

    ## Report extrapolated area fractions (need method for area fractions on extrapolatedLSD)
    # print(f'Area fraction < 0.01 km2: {lsd.area_fraction(0.01):,.2%}')
    # print(f'Area fraction < 0.1 km2: {lsd.area_fraction(0.1):,.2%}')

    # ## Plot HL extrapolation
    # # ax = lsd_hl.plot_lsd(all=False, reverse=False, normalized=False)
    # ax = lsd_hl_trunc.plot_extrap_lsd(label='HL-extrapolated', error_bars=False, normalized=False)
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')

    ## Flux prediction from observed and extrap lakes
    model = loadBAWLD_CH4()
    lsd_hl_trunc.predictFlux(model, includeExtrap=True)

    ## Plot combined extrap LSD/LEV
    fig, ax = plt.subplots(2,1, sharex=True)
    lsd_hl_trunc.plot_extrap_lsd(ax=ax[0], label='Lake area', error_bars=True, normalized=False, color='blue', plotLegend=False)
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    ax2=ax[0].twinx()
    lsd_hl_trunc.plot_extrap_lev(ax=ax[0], error_bars=True, color='green', label='Lake vegetation area', plotLegend=False, )
    ymin, ymax = ax[0].get_ylim()
    ax2.set_ylim([ymin, ymax/lsd_hl_trunc.sumAreas()])
    ax[0].set_ylabel('Cumulative area (million $km^2$)')
    ax[0].set_xlabel('')
    ax2.set_ylabel('Cumulative area fraction')
    # plt.tight_layout()

    ## Plot extrapolated fluxes
    lsd_hl_trunc.plot_extrap_flux(ax=ax[1], reverse=False, normalized=False, error_bars=True, plotLegend=False, label='Emissions')
    ax2=ax[1].twinx()
    ymin, ymax = ax[1].get_ylim()
    ax2.set_ylim([ymin, ymax/lsd_hl_trunc._total_flux_Tg_yr['mean']])
    ax2.set_ylabel('Cumulative emissions fraction')
    plt.tight_layout()
    [ax2.get_figure().savefig('/mnt/d/pic/BAWLD_areas_v4'+ext, transparent=True, dpi=300) for ext in ['.png','.pdf']]

    # ## Plot combined extrap LSD/Flux
    # norm = True # False
    # ax = lsd_hl_trunc.plot_extrap_lsd(label='HL-extrapolated', error_bars=True, normalized=False, color='blue')
    # # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    # ax2=ax.twinx()
    # lsd_hl_trunc.plot_extrap_flux(ax=ax2, reverse=False, normalized=norm, error_bars=True)
    # plt.tight_layout()

    ## Plot inset with just LEV, with normalized second axis
    sns.set_theme('poster', font='Ariel')
    sns.set_style('ticks')
    ax = lsd_hl_trunc.plot_extrap_lev(error_bars=True, color='green', plotLegend=False)
    ax2=ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim([ymin, ymax/lsd_hl_trunc.sumLev()['mean']])
    ax.set_ylabel('') #'Cumulative aquatic vegetation area (million $km^2$)')
    ax2.set_ylabel('') #'Cumulative aquatic vegetation area fraction')
    plt.tight_layout()
    [ax.get_figure().savefig('/mnt/d/pic/BAWLD_areas_inset_v4', transparent=True, dpi=300) for ext in ['.png','.pdf']]
    sns.set_theme('notebook', font='Ariel')
    sns.set_style('ticks')

    # ## Retrieve data from plot
    # ax.get_lines()[0].get_ydata() # gives right part of LSD plot # [1] is left part 
    # ax2.get_lines()[0].get_ydata() # gives right part of LEV plot
    # X_lev = np.concatenate((ax.get_lines()[1].get_xdata(), ax.get_lines()[0].get_xdata()))
    # S_lev = np.concatenate((ax.get_lines()[1].get_ydata(), ax.get_lines()[0].get_ydata()))

    ## LEV fraction stats, without and with extrap
    lev_est = lsd_hl_trunc.sumLev(includeExtrap=False, asFraction=True)
    print(f"Mean inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
    lev_est = lsd_hl_trunc.extrapLSD.sumLev(asFraction=True)
    print(f"Mean non-inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
    lev_est = lsd_hl_trunc.sumLev(includeExtrap=True, asFraction=True)
    print(f"Mean total LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")

    ## Area vs LEV plots (TODO: add extrap points)
    fig, ax = plt.subplots()
    # ax.scatter(lsd_hl_trunc.Area_km2, lsd_hl_trunc.LEV_MEAN)
    sns.scatterplot(lsd_hl_trunc, x='Area_km2', y='LEV_MEAN', ax=ax, alpha=0.1)
    ax.set_xscale('log')
    ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax})')

    ####################################
    ## Downing Analysis
    ####################################

    ## Remake plot for LSD
    fig, ax = plt.subplots()
    lsd_hl_trunc.plot_extrap_lsd(ax=ax, label='Lake area', error_bars=True, normalized=True, color='grey', plotLegend=False)
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    # ax2=ax.twinx()
    ax.set_ylabel('Cumulative area (normalized)')
    # ax.set_xlabel('')
    # ax2.set_ylabel('Cumulative area (normalized)')

    ## Compare to Downing, re-using code snippet from BinnedLSD and plot over first plot
    btm = 0.001
    top = 100000
    nbins = 8
    bin_edges = np.concatenate((np.geomspace(btm, top, nbins+1), [np.inf])).round(6) # bins computed from nbins and edges
    area_bins = pd.IntervalIndex.from_breaks(bin_edges, closed='left')
    # X = np.array(list(map(interval_geometric_mean, area_bins))) # take geom mean of each interval to get X-val
    X = bin_edges[1:] # plot against right bin edge
    d06 = [692600, 602100, 523400, 455100, 392362, 329816, 257856, 607650, 378119]
    group_sums = pd.Series(d06, index=area_bins, name='Area_km2') # from Downing 2006 paper
    binnedAreas = confidence_interval_from_extreme_regions(group_sums, None, None, name='Area_km2') # # Why are lower/upper non NaN?? Ignore.

    ## Put Downing number into my BinnedLSD data structure, just to verify plot
    lsd_d06 = BinnedLSD(btm=btm, top=top, nbins=nbins, binned_areas=binnedAreas, compute_ci_lsd=False) # give btm, top, nbins, compute_ci_lsd and binnedAreas args
    # lsd_d06_canon = BinnedLSD(btm=bin_edges[4], top=bin_edges[-2], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[4:-1], None, None, name='Area_km2'), compute_ci_lsd=False) # give btm, top, nbins, compute_ci_lsd and binnedAreas args
    # lsd_d06_extrap = BinnedLSD(btm=bin_edges[0], top=bin_edges[4], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[:4], None, None, name='Area_km2'), compute_ci_lsd=False) # give btm, top, nbins, compute_ci_lsd and binnedAreas args
    lsd_d06.plot(ax=ax, show_rightmost=False, as_lineplot=True, as_cumulative=True) # plot as binnedLSD, skipping top bin with Caspian Sea
    
    # fig, ax = plt.subplots()
    # ax.plot(X, np.cumsum(d06)/np.sum(d06)) # units Mkm2 /1e6
    d06_canonical = d06[4:]
    d06_extrap = d06[:4]
    ax.plot(X[:-1], np.cumsum(d06[:-1])/np.sum(d06[:-1]), color='orange', marker='x',linestyle='dashed') # This time, exclude top bin to better compare with BAWLD domain
    ax.plot(X[4:-1], (np.cumsum(d06_canonical[:-1])+np.sum(d06_extrap))/(np.sum(d06_canonical[:-1]) + np.sum(d06_extrap)), color='orange') # Plot canonical
    # ax.plot(X[:4], np.cumsum(d06_extrap)/np.sum(d06[:-1]), color='orange', linestyle='dashed') # Plot extrap
    fig.tight_layout()
    ax.set_yscale('linear')
    # ax.set_xscale('log')
    # ax.set_xticks(X)

    # print(f'Area in two smallest bins: {np.sum(d06[:2])/1e6}\nArea in three largest: {np.sum(d06[-3:])/1e6}')

    ###########################
    ## Create Table
    ## Create extrap LSD with all bin edge lining up with powers of 10, for Table
    ###########################
    
    ## Extrapolate with log10 bins
    tmin, tmax = (0.0001,5) # Truncation limits for ref LSD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
    log_bins_lower = [tmin, 0.001, 0.01, 0.1, emax]
    emax = 0.5 # Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
    binned_ref_log10bins = BinnedLSD(lsd.truncate(tmin, tmax), bins=log_bins_lower, compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd) # reference distrib (try 5, 0.5 as second args)
    binned_lev_log10bins = BinnedLSD(lsd_lev_cat, bins=log_bins_lower, compute_ci_lev=True, extreme_regions_lev=extreme_regions_lev_for_extrap) # 0.000125 is native
    lsd_hl_trunc_log10bins = lsd_hl_lev.truncate(emax, np.inf) # Beware chaining unless I return a new variable. # Try 0.1
    lsd_hl_trunc_log10bins.extrapolate(binned_ref_log10bins, binned_lev_log10bins)  

    ## Predict flux on extrapolated part
    lsd_hl_trunc_log10bins.predictFlux(model, includeExtrap=True)

    ## Predict flux on upper part
    lsd_hl_lev.predictFlux(model, includeExtrap=False)

    ## bin upper with log10 bins
    log_bins_upper = [0.5, 1, 10, 100, 1000, 10000, 100000]
    lsd_hl_lev_log10bins = BinnedLSD(lsd_hl_lev, bins=log_bins_upper, compute_ci_lsd=True, compute_ci_lev=True, normalize=False, extreme_regions_lsd=['HL', 'HL'], extreme_regions_lev=['HL', 'HL']) # now, bin upper values for Table estimate, use regions as placeholder to get dummy CI

    ## Make table: can ignore CI if nan or same as mean
    tb_area = pd.concat((lsd_hl_trunc_log10bins.extrapLSD.binnedAreas, lsd_hl_lev_log10bins.binnedAreas))
    tb_lev = pd.concat((lsd_hl_trunc_log10bins.extrapLSD.binnedLEV, lsd_hl_lev_log10bins.binnedLEV)) * tb_area #.loc[:, 'mean']
    tb_flux = pd.concat((lsd_hl_trunc_log10bins.extrapLSD.binnedG_day, lsd_hl_lev_log10bins.binnedG_day))
    tb_comb = pd.concat((tb_area, tb_lev, tb_flux), axis=1)
    tb_comb.columns = ['Area_km2', 'LEV_km2', 'G_day']

    ## Normalize
    tb_comb_norm = tb_comb / tb_comb.loc[:,'mean',:].sum(axis=0)

    ## Save as csv
    tb_comb.to_csv(os.path.join(tb_dir, 'Size_bin_table.csv'))
    tb_comb_norm.to_csv(os.path.join(tb_dir, 'Size_bin_table_norm.csv'))

    ## Print totals
    sums = tb_comb.loc[:,'mean',:].sum(axis=0)
    sums
    print(f"Total flus: {sums['G_day'] * 365.25 / 1e12}")
    ## print number of ref lakes:
    # len(lsd_hl_trunc)
    # lsd_hl_trunc.refBinnedLSD.binnedCounts.sum()
    # lsd_hl_trunc.extrapLSD.sumAreas()

    ###########################
    ## Compare emissions from small lakes if they are considered lakes vs wl
    # 33 marsh studies.
    # Marshes: mean: 171.6
    # Marshes: median: 106.0
    # Median flux from lakes < 0.001 km2: 90.6
    # Mean flux from lakes < 0.001 km2: 185.8
    # Flux ratio (Marsh:OW, method 1): 3.1
    ###########################

    ## filter out small size bins and compute area-weighted emissions # TODO: run with the actual veg methane estimate I'll be using # TODO: double check why my values are so much lower than BAWLD means, even when I compare my non-inv mean flux to 0.1-1 km2 BAWLD size bin
    non_inv_lks = tb_comb.loc[(tb_comb.index.get_level_values('size_bin')[0:9], 'mean'), :] # non-inventoried lakes
    non_inv_lks_mean_flux =  non_inv_lks.G_day.sum() / non_inv_lks.Area_km2.sum() / 1e6 * 1000 # area-weighted mean flux in mg/m2/day
    all_lks_mean_flux =  tb_comb.G_day.sum() / tb_comb.Area_km2.sum() / 1e6 * 1000
    emissions_factor_ni_lks = non_inv_lks_mean_flux / all_lks_mean_flux # emissions factor (ratio) for non-inventoried lakes (compare to 3.1 for wl, from bald marsh:lake emissions ratio)
    
    ## load bawld veg
    df_terr = pd.read_csv('/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD-CH4/data/ek_out/BAWLD_CH4_Terrestrial.csv', 
    encoding = "ISO-8859-1", dtype={'CH4.E.FLUX ':'float'}, na_values='-')
    df_terr.query('Class == "Marshes"')['CH4Av'].median()

    ####################################
    ## WBD comparison Analysis
    ####################################

    ## Compare HL extrapolation to WBD:
    # lsd_hl_trunc.truncate(0, 1000).plot_lsd(all=False, reverse=False, normalized=False)
    assert roi_region == 'WBD_BAWLD', f"Carefull, you are comparing to WBD, but roi_region is {roi_region}."
    ax = lsd_wbd.truncate(0.001, 10000).plot_lsd(all=False, reverse=False, normalized=False, color='r')
    lsd_hl_trunc.truncate(0, 10000).plot_extrap_lsd(label='HL-extrapolated', normalized=False, ax=ax, error_bars=False)
    ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax})')

    ## Report vals (for WBD)
    wbd_sum = lsd_wbd.truncate(0.001, 10000).Area_km2.sum()
    hl_extrap_sum = lsd_hl_trunc.truncate(0, 10000).sumAreas()
    print(f'{wbd_sum:,.0f} vs {hl_extrap_sum:,.0f} km2 ({((hl_extrap_sum - wbd_sum) / hl_extrap_sum):.1%}) difference between observed datasets WBD and HL in {roi_region}.')
    print(f'Area fraction < 0.01 km2: {lsd_wbd.truncate(0.001, np.inf).area_fraction(0.01):,.2%}')
    print(f'Area fraction < 0.1 km2: {lsd_wbd.truncate(0.001, np.inf).area_fraction(0.1):,.2%}')

    # ## Compare HL to WBD measured lakes in same domain:
    # # lsd_hl.truncate(0, 1000).plot_lsd(all=False, reverse=False, normalized=False) 
    # # lsd_hl = LSD.from_shapefile(gdf_HL_jn_pth, area_var='Shp_Area', idx_var='Hylak_id', name='HL', region_var=None) # reload, if needed # don't truncate this time
    # ax = lsd_wbd.truncate(0.1, 1000).plot_lsd(all=False, reverse=False, normalized=False, color='r')
    # lsd_hl.truncate(0.1, 1000).plot_lsd(normalized=False, reverse=False, ax=ax, all=False) # need to have loaded proper lsd hl bawld
    # ax.set_title(f'[{roi_region}]')
    # ax.get_figure().tight_layout()

    ## Compare WBD [self-]extrapolation to WBD (control tests):
    # lsd_hl.truncate(0, 1000).plot_lsd(all=False, reverse=False, normalized=False)
    tmin, tmax = (0.001, 30) # Truncation limits for ref LSD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
    emax = 0.5 # Extrapolation limit emax defines the left bound of the index region (and right bound of the extrapolation region).
    # binned_ref = BinnedLSD(lsd_wbd.truncate(tmin, tmax), tmin, emax) # uncomment to use self-extrap
    # txt='self-'
    binned_ref = BinnedLSD(lsd.truncate(tmin, tmax), tmin, emax, compute_ci_lsd=True, extreme_regions_lsd=extreme_regions_lsd)
    txt=''
    lsd_wbd_trunc = lsd_wbd.truncate(emax, np.inf)
    lsd_wbd_trunc.extrapolate(binned_ref)
    ax = lsd_wbd.truncate(0.001, 1000).plot_lsd(all=False, reverse=False, normalized=False, color='blue', plotLegend=False)
    lsd_wbd_trunc.truncate(0.001, 1000).plot_extrap_lsd(label=f'WBD-{txt}extrapolated', normalized=False, ax=ax, error_bars=True, plotLegend=False)

    ax2=ax.twinx() # Add normalized axis
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim([ymin, ymax/lsd_wbd_trunc.sumAreas()*1e6*1.28]) # hot fix
    ax2.set_ylabel('Cumulative area fraction')
    ax.get_figure().tight_layout()
    [ax.get_figure().savefig('/mnt/d/pic/WBD_compare_v2'+ext, transparent=False, dpi=300) for ext in ['.png','.pdf']]
 
    pass
################

# TODO: 
'''
* make equivalence to hl_pond_frac_cir x
* [try using numba to accelerate?]
* save 1 vs. 0.3 cutoff as var 
* add std or CI x
* write out x
* find a way to relate to flux estimates
* Re-define LSD so if called with no args but proper column names it returns a LSD correctly.
* Use fid_as_index argument when loading with pyarrow
* Preserve og index when concatenating so I can look up lakes from raw file (combine with above re: fid)
* Branches for if there is no CI/error bars in binned distrib. Make sure there is still a second index called 'stat' with constant val 'mean'
* Rewrite sumLev() to output a series x
* Go back and add branches for no CI to the various methods. Make sure it still has a second index for 'stat' with constant val 'mean'
* Make predictFlux() calls consistent bw LSD and BinnedLSD, whether they return a value or add an attribute.
* Thorough testing of all function options

NOTES:
* Every time a create an LSD() object in a function from an existing LSD (e.g. making a copy), I should pass it the public attributes of its parent, or they will be lost.
'''
