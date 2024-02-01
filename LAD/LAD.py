## Imports
from warnings import warn
import os
import math
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from glob import glob
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from labellines import labelLines
import seaborn as sns
import pandas as pd
import geopandas as gpd
import pyogrio

## Plotting params
sns.set_theme('notebook', font='Ariel')
sns.set_style('ticks')
# mpl.rc('text', usetex=True)
mpl.rcParams['pdf.fonttype'] = 42

## Functions and classes
## Plotting functions

## Common
temperature_metric = 'ERA5_stl1'
use_low_oc = True  # If false, use to compare to mutually-exclusive double-counted areas with Oc < 50%
# No effect, because computed in prep_data.ipynb. Factor to multiply D flux and divide E flux to fill in missing pathway e.g. (1.2)
eb_scaling = 0.580


def findNearest(arr, val):
    ''' Function to find index of value nearest to target value'''
    # calculate the difference array
    difference_array = np.absolute(arr - val)

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
    if values_for_sum is None:
        nanmask = ~np.isnan(values)
    else:
        # Values to keep, based on nan in values and values_for_sum
        nanmask = ~np.logical_or(np.isnan(values), np.isnan(values_for_sum))
    values = values[nanmask]
    X = np.sort(values)
    if reverse:
        X = X[-1::-1]  # highest comes first bc I reversed order

    if values_for_sum is None:
        # cumulative sum, starting with highest [lowest, if reverse=False] values
        S = np.cumsum(X)
    else:
        # convert to pandas, since X is now pandas DF
        if isinstance(values_for_sum, pd.Series):
            values_for_sum = values_for_sum.values
        values_for_sum = values_for_sum[nanmask]
        assert len(values_for_sum) == len(
            values), f"Values ({len(values)}) must be same length as values_for_sum ({len(values_for_sum)})"
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
    if X is None and S is None:  # values is given
        # compute, returns np arrays
        X, S = ECDFByValue(values, reverse=reverse)
    else:    # X and S given
        if isinstance(S, pd.Series):  # convert to pandas, since X is now pandas DF
            S = S.values
        if isinstance(X, pd.Series):  # sloppy repeat
            X = X.values
    if normalized:
        # S/np.sum(X) has unintended result when using for extrapolated, when S is not entirely derived from X
        S = S / S[-1]
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
    S = np.cumsum(X)  # cumulative sum, starting with lowest values
    if not ax:
        _, ax = plt.subplots()
    ax.plot(X, X / np.sum(X), **kwargs)

    ## Viz params
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylabel('Fraction of total area')
    ax.set_xlabel('Lake area')
    return  # S


def plotEmaxSens(lad, extreme_regions_lad, lad_binned_ref, lad_ref, tmin=0.0001, tmax=5, emax=0.1, y_scaler=1e6 * 1.11, ax=None):
    '''
    Shortcut to plot WBD comparison against HL LAD in WBD-NAHL domain. Prepares binned_ref based on tmin, tmax, emax and allows user to run in a loop.

    Parameters
    ----------
    extreme_regions_lad
    binned_ref_lad (LAD)
        Airborne HR data to use for extrapolation
    lad_ref (LAD)
        LAD to use for ground truth (e.g. WBD)
    tmin, tmax (floats) : (0.0001,30)
        Truncation limits for ref LAD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
    emax (float) : 0.5
        Upper extrapolation limit, defining the left bound of the index region (and right bound of the extrapolation region).
    y_scaler (float) : 1e6*1.11
        Ability to manually shift the normalized Y axis.
    ax (<class 'matplotlib.axes._axes.Axes'>) : None
        Optional axes to plot in
    Returns
    -------
    ax (<class 'matplotlib.axes._axes.Axes'>)

    '''
    if not ax:
        _, ax = plt.subplots()
    # Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
    binned_ref = BinnedLAD(lad_binned_ref.truncate(tmin, tmax), tmin, emax, compute_ci_lad=True,
                           extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
    # Beware chaining unless I return a new variable. # Try 0.1
    lad_trunc = lad.truncate(emax, np.inf)
    lad_trunc.extrapolate(binned_ref)  # no binned_lev included for now

    ax = lad_ref.truncate(0.001, 10000).plot_lad(
        all=False, reverse=False, normalized=False, color='r', ax=ax)
    lad_trunc.truncate(0, 10000).plot_extrap_lad(
        label='HL-extrapolated', normalized=False, ax=ax, error_bars=True)
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax})') # 'roi_region' must be a global var for this to work
    ax.set_title(
        f'Extrapolation limit: {emax} $km^2$ | Total area: {lad_trunc.truncate(0, 10000).sumAreas()/1e6:0.2} $Mkm^2$')

    ax2 = ax.twinx()  # Add normalized axis
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim([ymin, ymax / lad_trunc.sumAreas() * y_scaler])  # hot fix
    ax2.set_ylabel('Cumulative area fraction')
    ax.get_figure().tight_layout()
    ax.legend(loc='upper left')

    return ax


def weightedStd(x, w):
    '''Computes standard deviation of values given as group means x, with weights w'''
    return np.sqrt((np.average((x - np.average(x, weights=w, axis=0))**2, weights=w, axis=0)).astype('float'))


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
    idx = pd.MultiIndex.from_tuples([(label, stat) for label in group_sums.index for stat in [
                                    'mean', 'lower', 'upper']], names=['size_bin', 'stat'])
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
    idx = pd.MultiIndex.from_tuples([(label, stat) for label in group_means.index for stat in [
                                    'mean', 'lower', 'upper']], names=['size_bin', 'stat'])
    ds = pd.Series(index=idx, name=name, dtype=float)

    ## fille with nan if no CI
    if group_low is None:
        group_low = pd.Series(np.full_like(
            group_means, np.nan), index=group_means.index)
    if group_high is None:
        group_high = pd.Series(np.full_like(
            group_means, np.nan), index=group_means.index)

    ## set values
    ds.loc[ds.index.get_level_values(1) == 'lower'] = group_low.values
    ds.loc[ds.index.get_level_values(1) == 'upper'] = group_high.values
    ds.loc[ds.index.get_level_values(1) == 'mean'] = group_means.values
    return ds


def binnedVals2Error(binned_areas, n):
    '''Convert BinnedLAD.binnedAreas to error bars for plotting. May need to subtract 1 from n.'''
    ci = binned_areas.loc[:, ['lower', 'upper']].to_numpy().reshape((n, 2)).T
    mean = binned_areas.loc[:, ['mean']].to_numpy()
    yerr = np.abs(ci - mean)
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
    df = pd.read_csv('/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD-CH4/data/ek_out/BAWLD_CH4_Aquatic_ERA5.csv',
                     encoding="ISO-8859-1", dtype={'CH4.E.FLUX ': 'float'}, na_values='-')
    len0 = len(df)

    ## Add total open water flux column
    df['CH4.E.FLUX'].fillna(df['CH4.D.FLUX'] * eb_scaling, inplace=True)
    df['CH4.D.FLUX'].fillna(df['CH4.E.FLUX'] / eb_scaling, inplace=True)
    df['CH4.DE.FLUX'] = df['CH4.D.FLUX'] + df['CH4.E.FLUX']

    ## Filter and pre-process
    # df.query("SEASON == 'Icefree' ", inplace=True)  # and `D.METHOD` == 'CH'
    df.dropna(subset=['SA', 'CH4.DE.FLUX', temperature_metric],
              inplace=True)  # 'TEMP'

    ## if I want transformed y as its own var
    # df['CH4.DE.FLUX.LOG'] = np.log10(df['CH4.DE.FLUX']+1)

    ## print filtering
    len1 = len(df)
    print(f'Filtered out {len0-len1} BAWLD-CH4 values ({len1} remaining).')
    # print(f'Variables: {df.columns}')

    ## Linear models (regression)
    # 'Seasonal.Diff.Flux' 'CH4.D.FLUX'
    formula = f"np.log10(Q('CH4.DE.FLUX')+0.01) ~ np.log10(SA) + {temperature_metric}"
    model = ols(formula=formula, data=df).fit()

    return model


def computeLAV(df: pd.DataFrame, ref_dfs: list, names: list, extreme_regions_lev=None, use_zero_oc=False, use_low_oc=True) -> True:
    """
    Uses Bayes' law and reference Lake Emergent Vegetation (LEV) distribution to estimate the Lake Aquatic Vegetation (LAV) in a given df, based on water Occurrence.

    Parameters
    ----------
    df (pd.DataFrame) : A dataframe, where each row refers to one lake, with 101 water occurrence (Pekel 2016) classes ranging from 0-100%, named as 'Class_0',... 'Class_100'.

    ref_dfs (list) : where each item is a dataframe with format: Index: (LEV, dry land, invalid, water, SUM), Columns: ('HISTO_0', ... 'HISTO_100')

    names (list) : list of strings with dataset/region names in same order as ref_dfs

    extreme_regions_lev : array-like
        List of two region names to use for min/max LEV. Note, here region names must match entries in 'names,' so must be acronyms (e.g. CSD), unlike in BinnedLAD.

    use_zero_oc : Boolean (False)
        Whether to include zero occurrence bin in estimator. Setting to False (default) decreases sensitivity to misalignment between vector lakes and occurrence values because a complete shift will produce NaN due to only the zero bin overlaping with a lake 

    use_low_oc : Boolean (True)
        Whether to include <50 % occurrence bin in estimator, which might be desirable to compute LEV only over lake zones that aren't double-counted with wetlands. Setting to True (default) uses the bins.

    Returns
    -------
    lev : pd.DataFrame with same index as df and a column for each reference LEV distribution with name from names. Units are unitless (fraction)

    """
    ## Rename columns in ref_df to match df
    def func(x): return x.replace('HISTO_', 'Class_')
    ref_dfs = [ref_df.rename(columns=func) for ref_df in ref_dfs]

    # Multiply the dataframes element-wise based on common columns
    cols = ['LEV_' + name for name in names]
    # will have same length as df, which can become a LAD
    df_lev = pd.DataFrame(columns=cols)

    ## Branch
    nbins = 101  # init
    if use_zero_oc is False:
        nbins -= 1
    if use_low_oc is False:
        nbins -= 50

    ## Loop
    for i, ref_df in enumerate(ref_dfs):
        ''' df is in units of km2, ref_df is in units of px'''
        common_cols = df.columns.intersection(
            ref_df.columns.drop('Class_sum'))  # init
        if use_zero_oc is False:
            common_cols = common_cols.drop(['Class_0'])
        if use_low_oc is False:
            common_cols = common_cols.drop(
                [f'Class_{n}' for n in range(1, 51)])
        if i == 0:
            # update to exclude the 0 bin and only perform for first loop iteration
            df['Class_sum'] = df[common_cols].sum(axis=1)

        assert len(
            common_cols) == nbins, f"{len(common_cols)} common columns found bw datasets. {nbins} desired."
        # change order and only keep common (hist class) columns [repeats unnecessesarily ev time...]
        df_tmp = df[common_cols].reindex(columns=common_cols)
        # change order permanently (note: these orders are actually not numerical- no matter, as long as consistent bw two dfs)
        ref_df = ref_df[common_cols].reindex(columns=common_cols)
        # Mult Oc fraction of each oc bin by LEV fraction of each Oc bin.Broadcast ref_df over number of lakes
        result = df_tmp / \
            df.Class_sum.values[:, None] * \
            ref_df.loc['LEV', :] / ref_df.loc['CLASS_sum', :]
        df_lev['LEV_' + names[i]] = np.nansum(result, axis=1)

        ## If all bins are nan, set LEV to nan, rather than 0 as output by np.nansum
        df_lev.loc[np.isnan(result).sum(axis=1) == nbins,
                   'LEV_' + names[i]] = np.nan

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
        assert np.all(['LEV_' + region_col in df_lev.columns for region_col in extreme_regions_lev]
                      ), "One region name in extreme_regions_lev is not present in lad."
        df_lev['LEV_MEAN'] = df_lev[cols].mean(axis=1)
        df_lev['LEV_MIN'] = df_lev['LEV_' + extreme_regions_lev[0]]
        df_lev['LEV_MAX'] = df_lev['LEV_' + extreme_regions_lev[1]]
    else:
        df_lev['LEV_MEAN'] = df_lev[cols].mean(
            axis=1)  # df_lev[cols].mean(axis=1)
        df_lev['LEV_MIN'] = df_lev[cols].min(axis=1)  # df_lev[] #
        df_lev['LEV_MAX'] = df_lev[cols].max(axis=1)  # df_lev[] #

    ## Join and return
    df_lev = pd.concat((df.drop(columns=np.concatenate(
        (common_cols.values, ['Class_sum']))), df_lev), axis=1)

    return df_lev


def convertOccurrenceFormat(df):
    """
    Converts columns from format HISTO_# to Class_# and adds in any missing columns to ensure Class_0 through Class_100 are all present.
    """
    # Create a dictionary to map old column names to new column names
    column_mapping = {f'HISTO_{i}': f'Class_{i}' for i in range(101)}

    # Rename the columns
    df = df.rename(columns=column_mapping)

    # Add missing columns if they don't exist
    for i in range(101):
        col_name = f'Class_{i}'
        if col_name not in df.columns:
            df[col_name] = 0  # Add a new column with default value 0

    return df


def produceRefDs(ref_df_pth: str) -> True:
    """
    Pre-process raw dataframe in prep for computeLAV function.

    Parameters
    ----------
    ref_df (str) : Path to a csv file where each name is a region name and the values are dataframes with format: Index: (LEV, dry land, invalid, water, SUM), Columns: ('HISTO_0', ... 'HISTO_100')

    Returns: 
    -------
    df_out (pd.DataFrame): df with re-normalized and re-named columns

    """
    df = pd.read_csv(ref_df_pth, index_col='Broad_class').drop(
        index=['invalid', 'dry land', 'SUM'])

    ## Add missing HISTO_100 column if needed
    for i in range(101):
        if not f'HISTO_{i}' in df.columns:
            df[f'HISTO_{i}'] = 0

    df['HISTO_sum'] = df.sum(axis=1)
    df.loc['CLASS_sum', :] = df.sum(axis=0)

    return df


def loadUAVSAR(ref_names=['CSB', 'CSD', 'PAD', 'YF'], pths_shp: list = None, pths_csv: list = None):
    """
    Loads UAVSAR (LEV) data from shapefiles and pre-processed overlay CSV.

    Uses the regions in ref_names for averaging and regions in extreme_regions_lev_for_extrap for confidence interval. If no paths are given, defaults to pre-defined paths. Run after produceRefDs.

    Parameters
    ----------
    ref_names: list
        Short names for regions to use
    pths_shp : list
        Path to shapefiles.
    pths_csv : list
        Path to CSV files.

    Returns
    -------
    lad_lev_cat : LAD
        LAD for reference regions concatenated together. Used as LEV for non-inventoried lakes.
    ref_dfs : list
        Giving Occurrence values by land cover class (e.g. LEV, Water) for reference UAVSAR regions.
        Used to estimate LEV for inventoried lakes. List of pandas.DataFrames.
    """

    ## Load LEV/LAD from UAVSAR
    print('Loading UAVSAR and Pekel overlay...')
    if pths_shp and pths_csv:
        assert len(pths_shp) == len(
            pths_csv), "pths_shp must be same length as pths_csv"
    if (pths_shp is None) ^ (pths_csv is None):
        raise ValueError("pths_shp must be same length as pths_csv.")

    if pths_shp is None:  # default values
        pths_shp = ['/Volumes/thebe/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/bakerc_16008_19059_012_190904_L090_CX_01_Freeman-inc_rcls_lakes.shp',
                    '/Volumes/thebe/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/daring_21405_17094_010_170909_L090_CX_01_LUT-Freeman_rcls_lakes.shp',
                    '/Volumes/thebe/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/padelE_36000_19059_003_190904_L090_CX_01_Freeman-inc_rcls_lakes.shp',
                    '/Volumes/thebe/PAD2019/classification_training/PixelClassifier/Final-ORNL-DAAC/shp_no_rivers_subroi_no_smoothing/YFLATS_190914_mosaic_rcls_lakes.shp']

    if pths_csv is None:  # default values
        pths_csv = [  # CSV
            '/Volumes/thebe/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay_v2/bakerc_16008_19059_012_190904_L090_CX_01_Freeman-inc_rcls_brn_zHist_Oc_train_LEV_s.csv',
            '/Volumes/thebe/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay_v2/daring_21405_17094_010_170909_L090_CX_01_LUT-Freeman_rcls_brn_zHist_Oc_train_LEV_s.csv',
            '/Volumes/thebe/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay_v2/padelE_36000_19059_003_190904_L090_CX_01_Freeman-inc_rcls_brn_zHist_Oc_train_LEV_s.csv',
            '/Volumes/thebe/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/LEV_GSW_overlay_v2/YFLATS_190914_mosaic_rcls_brn_zHist_Oc_train_LEV_s.csv'
        ]
    values = [{'pths_shp': pths_shp[i], 'pths_csv': pths_csv[i]}
              for i in range(len(pths_shp))]
    pths_dict = {k: v for k, v in zip(ref_names, values)}

    lad_levs = []
    for i, pth in enumerate(pths_dict.values()):
        lad_lev_tmp = LAD.from_shapefile(pth['pths_shp'], name=ref_names[i], area_var='area_px_m2',
                                         lev_var='em_fractio', idx_var='label', _areaConversionFactor=1e6, other_vars=['edge', 'cir_observ'])
        lad_lev_tmp.query('edge==0 and cir_observ==1', inplace=True)
        lad_levs.append(lad_lev_tmp)

    ## Plots (only works for four regions)
    # fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
    # for i, ax in enumerate(axes.flatten()):
    #     lad_levs[i].plot_lev_cdf_by_lake_area(error_bars=False, ax=ax, plotLegend=False)

    ref_dfs = list(map(produceRefDs, [value['pths_csv']
                   for value in pths_dict.values()]))  # load ref dfs
    lad_lev_cat = LAD.concat(lad_levs, broadcast_name=True)
    return lad_lev_cat, ref_dfs


def regionStats(lad) -> True:
    """
    Analyze regions to choose min and max for confidence interval. Creates a df where each row is a region, with attributes for stats.

    Parameters
    ----------
    lad : LAD

    Returns
    -------
    df_regions : pd.DataFrame

    """
    cols = ['region', 'maxA', 'A_0.1', 'nGT0.5', 'nGT1', 'nGT5']
    df_regions = pd.DataFrame(columns=cols)
    for i, region in enumerate(np.unique(lad.Region)):
        lad_tmp = lad.query(f"Region == '{region}'")
        df_dict = {
            'region': region,
            'maxA': lad_tmp.Area_km2.max(),  # make sure region has data beyond index range
            'A_0.1': lad_tmp.area_fraction(0.1),
            'nGT0.5': lad_tmp.query("Area_km2 > 0.5").shape[0],
            'nGT1': lad_tmp.query("Area_km2 > 1").shape[0],
            'nGT5': lad_tmp.query("Area_km2 > 5").shape[0],
        }
        df_tmp = pd.DataFrame(df_dict, index=pd.Index([i]))
        df_regions = pd.concat((df_regions, df_tmp), ignore_index=True)
    return df_regions

## Classes (using inheritance)


class LAD(pd.core.frame.DataFrame):  # inherit from df? pd.DataFrame #
    '''Lake size distribution'''

    def __init__(self, df, name=None, area_var=None, region_var=None, idx_var=None, name_var=None, lev_var=None, t_var=None, _areaConversionFactor=1, regions=None, computeArea=False, other_vars=None, **kwargs):
        '''
        Loads df or gdf and creates copy only with relevant var names.
        If called with additional arguments (e.g. attributes from LAD class that is having a pd operation applied to it), they will be added as attributes

        Parameters
        ----------
        df : pandas.DataFrame or geopandas.GeoDataframe
            Dataframe to convert to LAD
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
            If provided, will compute Area from geometry. Doesn't need a crs (but make sure it is equal-area projection), but needs user input for 'areaConversionFactor.'
        other_vars : list, optional
            If provided, LAD will retain these columns.
        '''

        ## Add default name
        if name == None:
            name = 'unamed'

        ## Check if proper column headings exist (this might occur if I am initiating from a merged group of existing LAD objects)
        if 'idx_' + name in df.columns:
            idx_var = 'idx_' + name
        if 'Area_km2' in df.columns:
            area_var = 'Area_km2'
        if 'em_fractio' in df.columns:
            lev_var = 'em_fractio'
        if 'Region' in df.columns:
            region_var = 'Region'
        if 'geometry' in df.columns:  # if I am computing geometry, etc.
            geometry_var = 'geometry'
        else:
            geometry_var = None
        if 'est_mg_m2_day' in df.columns:
            mg_var = 'est_mg_m2_day'
        else:
            mg_var = None
        if 'est_g_day' in df.columns:
            g_var = 'est_g_day'
        else:
            g_var = None
        if 'Temp_K' in df.columns:
            t_var = 'Temp_K'
        else:
            t_var = None

        ## Choose which columns to keep (based on function arguments, or existing vars with default names that have become arguments)
        columns = [col for col in [idx_var, area_var, region_var, name_var, geometry_var,
                                   mg_var, g_var, lev_var, t_var] if col is not None]  # allows 'region_var' to be None

        ## Retain LEV variables if they exist
        # 'LEV_CSB', 'LEV_CSD', 'LEV_PAD', 'LEV_YF',       'LEV_MEAN', 'LEV_MIN', 'LEV_MAX'
        columns += [col for col in ['LEV_MEAN',
                                    'LEV_MIN', 'LEV_MAX'] if col in df.columns]

        ##  Retain other vars, if provided
        if other_vars is not None:
            columns += [col for col in other_vars]
        # remove duplicate columns (does it change order?)
        columns = np.unique(columns)

        # This inititates the class as a DataFrame and sets self to be the output. By importing a slice, we avoid mutating the original var for 'df'. Problem here is that subsequent functions might not recognize the class as an LAD. CAn I re-write without using super()?
        super().__init__(df[columns])

        ## Compute areas if they don't exist
        if computeArea == True:
            if area_var is None:
                gdf = gpd.GeoDataFrame(geometry=self.geometry)
                self['Area_km2'] = gdf.area
                self.drop(columns='geometry', inplace=True)
            else:
                raise ValueError(
                    'area_var is provided, but computeArea is set to True.')

        ## rename vars
        if region_var is not None:
            self.rename(columns={idx_var: 'idx_' + name, area_var: 'Area_km2',
                        lev_var: 'LEV_MEAN', region_var: 'Region'}, inplace=True)
        else:
            self.rename(columns={
                        idx_var: 'idx_' + name, area_var: 'Area_km2', lev_var: 'LEV_MEAN'}, inplace=True)

        ## Assert
        assert np.all(self.Area_km2 > 0), "Not all lakes have area > 0."

        ## Add attributes from input variables that get used in this class def
        self.name = name
        self.orig_area_var = area_var
        self.orig_region_var = region_var
        self.orig_idx_var = idx_var
        # important, because otherwise re-initiating won't know to retain this column
        self.name_var = name_var

        ## Add default passthrough attributes that get used (or re-used) by methods. Put all remaining attributes here.
        self.regions_ = None  # np.unique(self.Region) # retain, if present
        self.isTruncated = False
        self.truncationLimits = None
        self.isBinned = False
        self.bins = None
        self.refBinnedLAD = None  # Not essential to pre-set this

        ## Add passthrough attributes if they are given as kwargs (e.g. after calling a Pandas function). This overwrites defaults defined above.
        ## Any new attributes I create in future methods: ensure they start with '_' if I don't want them passed out.
        ## Examples: is_binned, orig_area_var, orig_region_var, orig_idx_var
        for attr, val in kwargs.items():
            setattr(self, attr, val)

        ## Set new attributes (Ensure only executed first time upon definition...)
        if _areaConversionFactor != 1:
            self.Area_km2 = self.Area_km2 / _areaConversionFactor
        if regions is not None:
            self.reindexregions_(regions)
        if idx_var is None:  # if loaded from shapefile that didn't have an explicit index column
            self.reset_index(inplace=True, drop=True)
        if region_var is None:  # auto-name region from name if it's not given
            self['Region'] = name

    def get_public_attrs(self):
        return public_attrs(self)

    @classmethod
    def from_shapefile(cls, path, name=None, area_var=None, lev_var=None, region_var=None, idx_var=None, **kwargs):
        ''' 
        Load from shapefile on disk.
        Accepts all arguments to LAD.
        '''
        columns = [col for col in [idx_var, area_var, lev_var, region_var]
                   if col is not None]  # allows 'region_var' to be None

        ##  Retain other vars, if provided
        if 'other_vars' in kwargs:
            columns += [col for col in kwargs['other_vars']]

        read_geometry = False
        if 'computeArea' in kwargs:
            if kwargs['computeArea'] == True:
                read_geometry = True
        df = pyogrio.read_dataframe(
            path, read_geometry=read_geometry, use_arrow=True, columns=columns)
        if name is None:
            name = os.path.basename(path).replace(
                '.shp', '').replace('.zip', '')
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
        dfs = []  # create an empty list to store the loaded shapefiles

        ## Filter out raw regions
        if exclude is not None:
            shapefiles = [file for file in shapefiles if not any(
                fname in file for fname in exclude)]

        # loop through the shapefiles and load each one using Geopandas
        for shpfile in shapefiles:
            lad = cls.from_shapefile(
                shpfile, area_var=area_var, lev_var=lev_var, region_var=region_var, idx_var=idx_var, **kwargs)
            dfs.append(lad)

        # merge all the loaded shapefiles into a single GeoDataFrame
        lad = LAD.concat(dfs, ignore_index=True)  # , crs=gdfs[0].crs)
        lad.name = name
        return lad

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

    @classmethod  # not sure if necessary
    def concat(cls, lads, broadcast_name=False, **kwargs):
        '''
        Concatenates multiple LAD instances.
        Don't call on an instance, but call from the base class. Takes the name of the first LAD
        lads : arry_like
            Array of LADS to concatenate.
        broadcast_name : boolean
            If true, add a column to each of the input LADS with their name (name of dataset)

        Example: LAD.concat((lad1, lad2))
        '''
        if broadcast_name:  # if combining multiple lads from different sources
            for lad in lads:
                lad['Name'] = lad.name
                name_var = 'Name'
                name = 'multi'
        else:  # if loading in lads from same source, but different files
            name_var = None
            name = lads[0].name

        cols = None
        # Need to re-init before returning because pd.DataFrame.concat is a function, not method and can't return in-place. Therefore, it returns a pd.DataFrame object that needs to be converted back to a LAD.
        return cls(pd.concat(lads, **kwargs), name=name, name_var=name_var, other_vars=cols)

    def truncate(self, min: float, max: float = np.inf, inplace=False, **kwargs):
        '''
        Truncates LAD by keeping only lakes >= min threshold [and < max threshold].
        Always performed inplace.
        TODO: rewrite to call query() instead of repeating lines.
        '''
        if inplace == True:
            pd.DataFrame.query(self, "(Area_km2 >= @min) and (Area_km2 < @max)",
                               inplace=inplace, **kwargs)  # true inplace
            self.isTruncated = True
            self.truncationLimits = (min, max)
        else:
            attrs = self.get_public_attrs()
            cols = self.columns.to_list()
            lad = LAD(pd.DataFrame.query(self, "(Area_km2 >= @min) and (Area_km2 < @max)",
                      inplace=inplace, **kwargs), other_vars=cols, **attrs)  # false
            lad.isTruncated = True
            lad.truncationLimits = (min, max)
            return lad

    def query(self, expr, inplace=False):
        '''
        Runs np.DataFrame.query and returns an LAD of results. 
        When inplace=False, output is re-generated as an LAD. When inplace=True, output LAD class is unchanged.
        '''
        if inplace == True:
            pd.DataFrame.query(self, expr, inplace=inplace)  # true
        else:
            cols = self.columns.to_list()
            attrs = self.get_public_attrs()
            # false
            return LAD(pd.DataFrame.query(self, expr, inplace=inplace), other_vars=cols, **attrs)

    def area_fraction(self, limit):
        '''
        Compute the fraction of areas from lakes < area given by lim, only if not computed already.
        Creates attribute A_[n]_ where n=0.001, 0.01, etc.
        No need to call A_0.001_ directly, because it may not exist.
        Will not include extrapolated areas in estimate, because bin edges may not align. Use extrapolated_area_fraction() instead.

        TODO: add option to include extrapolation in estimate, or to give units of km2.
        '''
        if self.isTruncated:
            if self.truncationLimits[1] is not np.inf:
                warn("Careful, you are computing an area fraction based on a top-truncated LAD, so the fraction may be too high.")
        # dynamically-named attribute (_ prefix means it won't get copied over after a truncation or concat)
        attr = f'_A_{limit}'
        if attr in self.get_public_attrs():
            return getattr(self, attr)
        else:  # do computation
            area_fraction = self.truncate(
                0, limit).Area_km2.sum() / self.Area_km2.sum()
            setattr(self, attr, area_fraction)
            return area_fraction

    def extrapolated_area_fraction(self, ref_LAD, bottomLim, limit, emax=0.5, tmax=5):
        """
        Computes area fraction from lakes < limit in area, for limit < minimum observed lake area.

        Does this by calling extrapolate() twice, using different bottomLims to ensure the area thresholds in question exactly match the desired limit. Creates attribute A_[n]_ where n=0.001, 0.01, etc. Self should be truncated, ref_LAD shouldn't.

        TODO: add confidence interval

        Parameters
        ----------
        self : LAD (truncated)
        limit  : float
            threshold. Will be used for extrapolation (for numerator).
        ref_LAD : LAD (not truncated)
            Reference LAD to bin and then use for extrapolation.
        bottomLim : float
            Bottom size limit to which to extrapolate (for denominator).    
        emax : float, default 0.5
            Extrapolation limit max. emax defines the right bound of the extrapolation region (and the left bound of the index region). Should be set a little higher than the resolution of the dataset to be extrapolated to account for biases near its detection limit.
        tmax : float, default 5
            tmax defines the right bound of the index region (so it truncates the ref LAD). don't set it too high, which introduces variability between datasets in the index region fractional areas.
        Returns
        -------
        area_fraction (float): sum all areas < limit and divide

        """
        ## Checks
        if self is None:
            raise ValueError("self cannot be None")
        assert isinstance(ref_LAD, LAD), "ref_LAD must be type LAD"
        assert not ref_LAD.isTruncated, "ref_LAD shouldn't already be truncated, because it needs to happen inside this function."
        assert self.isTruncated, "To be explicit, truncate self, which indicates which region to extrapolate to."
        # auto run area_fraction
        assert limit < self.truncationLimits[0], f"Limit ({limit}) is >= the lower truncation limit of LAD ({self.truncationLimits[0]}), so use area_fraction() method instead."
        assert limit > bottomLim, f"'limit' ({limit}) must be > bottomLim ({bottomLim})."
        assert emax >= self.truncationLimits[
            0], f"emax ({emax}) should be >= the top truncation limit of self ({self.truncationLimits[1]})"
        if self.truncationLimits[1] is not np.inf:
            warn("Careful, you are computing an extrapolated area fraction based on a top-truncated LAD, so the fraction may be too high.")

        ## Make copy for modification via extrapolate function
        attrs = self.get_public_attrs()
        lad = LAD(self, **attrs)

        ## Computation
        binned_ref = BinnedLAD(ref_LAD.truncate(
            0, tmax), btm=limit, top=emax, compute_ci_lad=False)
        lad.truncate(emax, np.inf, inplace=True)
        lad.extrapolate(binned_ref)
        num = lad.sumAreas(includeExtrap=True)

        lad = LAD(self, **attrs)  # re-init
        binned_ref = BinnedLAD(ref_LAD.truncate(
            0, tmax), bottomLim, emax, compute_ci_lad=False)
        lad.truncate(emax, np.inf, inplace=True)
        lad.extrapolate(binned_ref)
        denom = lad.sumAreas(includeExtrap=True)
        area_fraction = 1 - num / denom

        ## update area fract on original lad and return
        attr = f'_A_{limit}'
        setattr(self, attr, area_fraction)  # Save value within LAD structure
        return area_fraction

    def extrapolate(self, ref_BinnedLAD, ref_BinnedLEV=None):
        '''
        Extrapolate by filling empty bins below the dataset's resolution.

        Note: The limits of the extrapolation are defined by the btmEdge/topEdge of ref_BinnedLAD. The top limit of rererence distribution is defined by the top truncation of the reference binnedLAD. Function checks to make sure it is truncated <= 5 km2.
        TODO: give error if trying to extrapolate to a smaller area than is present in ref distrib; truncate last bin?

        Parameters
        ----------
        ref_BinnedLAD : binnedLAD
            Reference binnedLAD used for extrapolation.
        ref_BinnedLEV : binnedLAD that has LEV 
            Reference binnedLAD used for LEV extrapolation.This binned LAD probably comes from different data, hence the additional argument.

        Returns: binnedLAD
        '''

        ## Check validity
        assert ref_BinnedLAD.isTruncated, "Reference binnedLAD must be top truncated or its bin estimates will be highly variable."
        assert self.isTruncated, "LAD should be bottom-truncated when used for extrapolation to be explicit."
        assert self.truncationLimits[
            0] == ref_BinnedLAD.topEdge, f"Mismatch between LAD bottom truncation limit ({self.truncationLimits[0]}) and ref binned LAD top edge ({ref_BinnedLAD.topEdge})"
        assert ref_BinnedLAD.isNormalized
        if ref_BinnedLAD.truncationLimits[1] > 5:
            warn("Reference binned LAD should be top truncated to <= ~ 5 km2 to avoid too large of an index region (highly variable and HR lad might be biased low in a large index region).")
        if ref_BinnedLEV is not None:
            assert hasattr(
                ref_BinnedLEV, 'binnedLEV'), "If ref_BinnedLEV is provided, it must have a a binnedLEV attribute."
            assert ref_BinnedLAD.nbins == ref_BinnedLEV.nbins, "nbins differs between ref_BinnedLAD and ref_BinnedLEV"
            assert ref_BinnedLAD.btmEdge == ref_BinnedLEV.btmEdge, f"Ref binned LAD btm edge ({ref_BinnedLAD.btmEdge}) doesn't equal ref binned LEV btm edge ({ref_BinnedLEV.btmEdge})"
            assert ref_BinnedLAD.topEdge == ref_BinnedLEV.topEdge, f"Ref binned LAD top edge ({ref_BinnedLAD.topEdge}) doesn't equal ref binned LEV top edge ({ref_BinnedLEV.topEdge})"
        ## Perform the extrapolation (bin self by its bottom truncation limit to the indexTopLim, and simply multiply it by the normalized refBinnedLAD to do the extrapolation)!

        # Sum all area in LAD in the index region, defined by the topEdge and top-truncation limit of the ref LAD.
        index_region_sum = self.truncate(
            ref_BinnedLAD.topEdge, ref_BinnedLAD.truncationLimits[1]).Area_km2.sum()
        last = ref_BinnedLAD.binnedAreas.index.get_level_values(
            0).max()  # last index with infinity to drop (use as mask)
        # remove the last entries, which are mean, lower, upper for the bin that goes to np.inf
        binned_areas = ref_BinnedLAD.binnedAreas.drop(
            index=last) * index_region_sum

        ## Return in place a new attribute called extrapLAD which is an instance of a binnedLAD
        self.extrapLAD = BinnedLAD(btm=ref_BinnedLAD.btmEdge, top=ref_BinnedLAD.topEdge,
                                   nbins=ref_BinnedLAD.nbins, compute_ci_lad=ref_BinnedLAD.hasCI_lad, binned_areas=binned_areas)

        ## Add binnedLEV, if specified. Technically, not even extrapolating, just providing the reference distribution, but using same syntax as for LAD for compatibility.
        if ref_BinnedLEV is not None:
            last_lev = ref_BinnedLEV.binnedAreas.index.get_level_values(
                0).max()
            self.extrapLAD.binnedLEV = ref_BinnedLEV.binnedLEV.drop(
                index=last_lev)  # will this work if no LEV_MIN/MAX?

        ## Manually update its attributes
        # since bottom of index region matches, ensure top does as well in definition.
        self.extrapLAD.indexTopLim = ref_BinnedLAD.truncationLimits[1]
        self.extrapLAD.isExtrapolated = True
        self.extrapLAD.isNormalized = False  # units of km2, not fractions
        self.extrapLAD.bottomLim = ref_BinnedLAD.btmEdge
        self.extrapLAD.topLim = ref_BinnedLAD.topEdge
        self.extrapLAD.hasCI_lad = ref_BinnedLAD.hasCI_lad
        self.extrapLAD.binnedDC = None
        if hasattr(ref_BinnedLEV, 'hasCI_lev'):
            self.extrapLAD.hasCI_lev = ref_BinnedLEV.hasCI_lev
        self.extrapLAD.extreme_regions_lad = ref_BinnedLAD.extreme_regions_lad
        if hasattr(ref_BinnedLEV, 'extreme_regions_lev'):
            self.extrapLAD.extreme_regions_lev = ref_BinnedLEV.extreme_regions_lev

        ## Save reference binned LAD (which has any remaining attrs)
        self.refBinnedLAD = ref_BinnedLAD
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
        self._extrap_area = None  # init
        self._total_area = None
        if includeExtrap:
            assert hasattr(
                self, 'extrapLAD'), "LAD doesn't include an extrapLAD attribute."
            if ci == False:
                self._extrap_area = self.extrapLAD.sumAreas(ci=ci)
                out = self._extrap_area + self._observed_area
            else:
                self._extrap_area = np.array(self.extrapLAD.sumAreas(ci=ci))
                # convert to tuple, as is common for python fxns to return
                out = tuple(self._extrap_area + self._observed_area)
            self._total_area = self._observed_area + self._extrap_area
        else:
            out = self._observed_area

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
        lev_sum = confidence_interval_from_extreme_regions(
            *[(pd.Series((self[param] * self.Area_km2).sum())) for param in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']]).loc[0, :]
        result = lev_sum
        total_area = self.sumAreas(includeExtrap=False, ci=False)
        if includeExtrap:
            if hasattr(self, 'extrapLAD'):
                if hasattr(self.extrapLAD, 'binnedLEV'):
                    lev_sum_extrap = self.extrapLAD.sumLev(asFraction=False)
                    result = lev_sum + lev_sum_extrap
                    total_area = self.sumAreas(includeExtrap=True)
                    pass
                else:
                    warn(
                        'No LEV in extrapolated LAD. Returning calculation based on inventoried lake LEV only.')
            else:
                warn(
                    'No extrapolated LAD. Returning calculation based on inventoried lake LEV only.')
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
        assert 'Temp_K' in self.columns, "LAD needs a Temp_K attribute in order to predict flux."
        if includeExtrap == True:
            assert hasattr(
                self, 'extrapLAD'), "includeExtrap was set to true, but no self.extrapLAD found."
            # Use lake-area-weighted average temp as temp for extrapolated lakes, since we don't know where they are located
            self.extrapLAD.Temp_K = np.average(
                self.Temp_K, weights=self.Area_km2)
            self.extrapLAD.predictFlux(model)
            binned_total_flux_Tg_yr = self.extrapLAD._total_flux_Tg_yr
        else:
            binned_total_flux_Tg_yr = 0  # Not NaN, because needs to be added

        ## Flux (areal, mgCH4/m2/day)
        self['est_mg_m2_day'] = 10**(model.params.Intercept +
                                     model.params['np.log10(SA)'] *
                                     np.log10(self.Area_km2)
                                     + model.params[temperature_metric] * self.Temp_K) - 0.01  # jja, ann, son, mam

        ## Flux (flux rate, gCH4/day)
        self['est_g_day'] = self.est_mg_m2_day * self.Area_km2 * \
            1e3  # * 1e6 / 1e3 # (convert km2 -> m2 and mg -> g)

        self._total_flux_Tg_yr = self['est_g_day'].sum(
        ) * 365.25 / 1e12 + binned_total_flux_Tg_yr  # see Tg /yr
        # return self._Total_flux_Tg_yr
        return

    ## Plotting
    def plot_lad(self, all=True, plotLegend=True, groupby_name=False, cdf=True, ax=None, plotLabels=False, **kwargs):
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
        sns.set_palette("colorblind", len(self.regions()))

        ## plot
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)

        if groupby_name == False:  # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                X, S = ECDFByValue(pd.DataFrame.query(
                    self, 'Region == @region').Area_km2, reverse=False)
                plotECDFByValue(X=X, S=S / 1e6, ax=ax,
                                alpha=0.4, label=region, **kwargs)

        else:
            assert 'Name' in self.columns, "LAD is missing 'Name' column."
            # cmap = colors.Colormap('Pastel1')
            names = np.unique(self['Name'])
            cmap = plt.cm.get_cmap('Paired', len(names))
            for j, name in enumerate(names):
                # can't use .regions() after using DataFrame.query because it returns a DataFrame
                for i, region in enumerate(np.unique(pd.DataFrame.query(self, 'Name == @name').Region)):
                    X, S = ECDFByValue(pd.DataFrame.query(
                        self, 'Region == @region').Area_km2, reverse=False)
                    plotECDFByValue(X=X, S=S / 1e6, ax=ax, alpha=0.6,
                                    label=name, color=cmap(j), **kwargs)

        ## repeat for all
        if all:
            X, S = ECDFByValue(self.Area_km2, reverse=False)
            plotECDFByValue(X=X, S=S / 1e6, ax=ax, alpha=0.4,
                            color='black', label='All', **kwargs)

        ## Legend and labels
        if plotLegend:
            # legend on right (see https://stackoverflow.com/a/43439132/7690975)
            ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
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
        assert 'est_g_day' in self.columns, "LAD doesn't have a flux estimate yet."
        # colors from https://stackoverflow.com/a/46152327/7690975 Other option is: `from cycler import cycler; `# ax.set_prop_cycle(rainbow_cycler), plt(... prop_cycle=rainbow_cycler, )
        sns.set_palette("colorblind", len(self.regions()))

        ## plot
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)

        if groupby_name == False:  # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                lad_by_region = self.query(f'Region == "{region}"')
                X, S = ECDFByValue(lad_by_region.Area_km2, values_for_sum=lad_by_region.est_g_day *
                                   365.25 / 1e12, reverse=reverse)  # convert from g/day to Tg/yr
                # if reverse, set this in ECDFByValue on previous line
                plotECDFByValue(X=X, S=S, ax=ax, alpha=0.4,
                                label=region, normalized=normalized, reverse=False)

        else:
            assert 'Name' in self.columns, "LAD is missing 'Name' column."
            # cmap = colors.Colormap('Pastel1')
            names = np.unique(self['Name'])
            cmap = plt.cm.get_cmap('Paired', len(names))
            for j, name in enumerate(names):
                # OLD: can't use .regions() after using DataFrame.query because it returns a DataFrame
                for i, region in enumerate(np.unique(pd.DataFrame.query(self, f'Name == "{name}"').Region)):
                    lad_by_region_name = self.query(f'Region == "{region}"')
                    X, S = ECDFByValue(
                        lad_by_region_name.Area_km2, values_for_sum=lad_by_region_name.est_g_day * 365.25 / 1e12, reverse=reverse)
                    plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, label=name, color=cmap(
                        j), normalized=normalized, reverse=False)
        ## repeat for all
        if all:
            X, S = ECDFByValue(
                self.Area_km2, values_for_sum=self.est_g_day * 365.25 / 1e12, reverse=reverse)
            plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, color='black',
                            label='All', normalized=normalized, reverse=False)

        ## Legend
        if plotLegend:
            # legend on right (see https://stackoverflow.com/a/43439132/7690975)
            ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))

        ## Override default axes
        if normalized:
            ylabel = 'Cumulative fraction of emissions'
        else:
            ylabel = 'Cumulative emissions (Tg/yr)'
        ax.set_ylabel(ylabel)

        return ax

    def plot_extrap_lad(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots LAD using both measured and extrapolated values. 
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
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)

        if self.extrapLAD.hasCI_lad:  # Need to extract means in a different way if there is no CI
            means = self.extrapLAD.binnedAreas.loc[:, 'mean']
        else:
            means = self.extrapLAD.binnedAreas
        # plot for observed part of distrib
        X, S = ECDFByValue(self.Area_km2, reverse=False)
        # take geom mean of each interval
        geom_means = np.array(list(map(
            interval_geometric_mean, self.extrapLAD.binnedAreas.loc[:, 'mean'].index.get_level_values(0))))
        means = means.values  # convert to numpy
        S += self.extrapLAD.sumAreas()  # add sum of extrap distrib to original cumsum
        # pre-pend the cumsum of extrap distrib binned vals
        S0 = np.cumsum(means)
        if not 'color' in kwargs:
            kwargs['color'] = 'black'

        ## Plot
        if normalized:  # need to normalize outside of plotECDFByValue function
            denom = self.sumAreas() / 1e6
        else:
            denom = 1

        ## Add error bars
        if error_bars == True:
            assert self.extrapLAD.hasCI_lad, 'If plotting error bars, self.extrapLAD.hasCI_lad needs a CI.'

            ## as area plot: extrap section
            S_low0 = np.cumsum(
                self.extrapLAD.binnedAreas.loc[:, 'lower'])  # btm section
            S_up0 = np.cumsum(self.extrapLAD.binnedAreas.loc[:, 'upper'])

            ## Observed section
            yerr_top = self.extrapLAD.binnedAreas.loc[:, 'mean'].sum(
            ) - self.extrapLAD.binnedAreas.loc[:, 'lower'].sum()
            yerr_btm = self.extrapLAD.binnedAreas.loc[:, 'upper'].sum(
            ) - self.extrapLAD.binnedAreas.loc[:, 'mean'].sum()
            S_low = np.maximum(S - yerr_top, 0)
            S_up = S + yerr_btm
            ax.fill_between(np.concatenate((geom_means, X)), np.concatenate((S_low0, S_low)) / denom /
                            1e6, np.concatenate((S_up0, S_up)) / denom / 1e6, alpha=0.1, color=kwargs['color'])
        else:
            pass

        ## Plot main curves
        plotECDFByValue(ax=ax, alpha=1, X=X, S=S / denom / 1e6,
                        normalized=False, reverse=reverse, **kwargs)
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0 / denom / 1e6, normalized=False,
                        reverse=reverse, linestyle='dashed', **kwargs)  # second plot in dashed for extrapolated
        if normalized:  # need to change label outside of plotECDFByValue function
            ax.set_ylabel('Cumulative fraction of total area')
        if plotLegend:
            ax.legend(loc='best')
        ax.set_ylim(0, ax.get_ylim()[1])
        return ax

    def plot_extrap_lev(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots lev from LAD using both measured and extrapolated values. 
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
        assert 'LEV_MEAN' in self.columns, "LAD doesn't have a LEV estimate yet."
        assert hasattr(
            self.extrapLAD, 'binnedLEV'), "binnedLAD doesn't have an extrap lev estimate yet."
        assert reverse == False, "No branch yet written for flux plots in reverse."
        if error_bars:
            assert_vars = ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']
        else:
            assert_vars = ['LEV_MEAN']
        for var in assert_vars:
            assert var in self.columns, f"LAD is missing {var} column, which is required to plot lev cdf."

        ## Prepare values
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)
        try:
            # take geom mean of each interval
            geom_means = np.array(
                list(map(interval_geometric_mean, self.extrapLAD.binnedLEV.index)))

        except:
            # if ci values are present
            geom_means = np.array(list(map(
                interval_geometric_mean, self.extrapLAD.binnedLEV.loc[:, 'mean'].index.get_level_values(0))))
        # used to be a branch for if self.extrapLAD.hasCI_lad...
        means = self.extrapLAD.binnedLEV * \
            self.extrapLAD.binnedAreas.loc[:, 'mean']

        X, S = ECDFByValue(self.Area_km2, values_for_sum=self.LEV_MEAN *
                           self.Area_km2, reverse=False)  # convert from frac to km2
        # TODO: need branch for if no 'mean' col for all occurrences, or ensure it always will have one # HERE 4/5/2023
        binned_lev_km2 = (self.extrapLAD.binnedLEV *
                          self.extrapLAD.binnedAreas.loc[:, 'mean'])
        # self.extrapLAD.sumLev0() # add sum of extrap distrib to original cumsum
        S += binned_lev_km2.loc[:, 'mean'].sum()
        # pre-pend the cumsum of extrap distrib binned vals
        S0 = np.cumsum(means.loc[:, 'mean'])
        if not 'color' in kwargs:
            kwargs['color'] = 'green'

        ## Add error bars
        if error_bars == True:
            assert self.extrapLAD.hasCI_lev, "error_bars is set, but extrapolated lev has no CI (self.extrapLAD.hasCI_lev is False)"

            ## Compute error bounds for extrapolated LEV
            S_low0 = np.cumsum(
                self.extrapLAD.binnedLEV.loc[:, 'lower'] * self.extrapLAD.binnedAreas.loc[:, 'mean'])  # btm section
            S_up0 = np.cumsum(
                self.extrapLAD.binnedLEV.loc[:, 'upper'] * self.extrapLAD.binnedAreas.loc[:, 'mean'])

            ## Compute error bounds for estimated LEV from obs
            X_low, S_low = ECDFByValue(
                self.Area_km2, self.LEV_MIN * self.Area_km2, reverse=False)
            S_low += binned_lev_km2.loc[:, 'lower'].sum()
            _, S_up = ECDFByValue(
                self.Area_km2, self.LEV_MAX * self.Area_km2, reverse=False)
            S_up += binned_lev_km2.loc[:, 'upper'].sum()

            ## Area plot
            ax.fill_between(np.concatenate((geom_means, X_low)), np.concatenate((S_low0, S_low)) / 1e6,
                            np.concatenate((S_up0, S_up)) / 1e6, alpha=0.1, color=kwargs['color'])  # TODO: remove overlap line

        ## Plot main curves
        if normalized:
            ylabel = 'Cumulative aquatic vegetation fraction'
            # self.sumLev0() # note this won't include extrap lake fluxes if there is no self.extrapBinnedLAD, but the assert checks for this.
            denom = binned_lev_km2.loc[:, 'mean'].sum()
        else:
            ylabel = 'Cumulative aquatic vegetation (million $km^2$)'
            denom = 1
        plotECDFByValue(ax=ax, alpha=1, X=X, S=S / denom / 1e6,
                        normalized=False, reverse=reverse, **kwargs)  # obs values
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0 / denom / 1e6, normalized=False,
                        reverse=reverse, linestyle='dashed', **kwargs)  # second plot is dashed for extrapolation
        ax.set_ylabel(ylabel)
        if plotLegend:
            ax.legend(loc='best')
        ax.set_ylim(0, ax.get_ylim()[1])
        return ax

    def plot_extrap_flux(self, plotLegend=True, ax=None, normalized=False, reverse=False, error_bars=False, **kwargs):
        '''
        Plots fluxes from LAD using both measured and extrapolated values. 
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
        assert 'est_g_day' in self.columns, "LAD doesn't have a flux estimate yet."
        assert hasattr(
            self.extrapLAD, 'binnedG_day'), "binnedLAD doesn't have a flux estimate yet."
        assert reverse == False, "No branch yet written for flux plots in reverse."
        ## Prepare values
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)

        # Create plot values for observed part
        means = self.extrapLAD.binnedG_day.loc[:, 'mean']
        X, S = ECDFByValue(self.Area_km2, values_for_sum=self.est_g_day *
                           365.25 / 1e12, reverse=False)  # scale to Tg / yr
        # take geom mean of each interval Try self.extrapLAD.binnedAreas.loc[:, 'mean'].index.get_level_values(0)
        geom_means = np.array(list(map(interval_geometric_mean, means.index)))
        S += self.extrapLAD.sumFluxes()['mean']  # add to original cumsum
        S0 = np.cumsum(means) * 365.25 / 1e12  # pre-pend the binned vals
        if not 'color' in kwargs:
            kwargs['color'] = 'red'
        # TODO: rewrite vectorized to use cumsum over three indexes (mean, upper, lower). Right now, seems to not be equivalent... Try using .sum(level=1)

        if normalized:
            ylabel = 'Cumulative fraction of emissions'
            # note this won't include extrap lake fluxes if there is no self.extrapBinnedLAD, but the assert checks for this.
            denom = self._total_flux_Tg_yr['mean']
        else:
            ylabel = 'Cumulative emissions (Tg/yr)'
            denom = 1

        ## Add error bars
        if error_bars == True:
            assert self.extrapLAD.hasCI_lad, "error_bars is set, but extrapolated LAD has no CI (self.extrapLAD.hasCI_lad is False)"

            ## as area plot: extrapolated flux
            S_low0 = np.cumsum(
                self.extrapLAD.binnedG_day.loc[:, 'lower']) * 365.25 / 1e12 / denom  # btm section
            S_up0 = np.cumsum(
                self.extrapLAD.binnedG_day.loc[:, 'upper']) * 365.25 / 1e12 / denom

            ## as area plot: estimated flux from obs
            X_low, S_low = ECDFByValue(
                self.Area_km2, self.est_g_day * 365.25 / 1e12 / denom, reverse=False)
            S_low += self.extrapLAD.binnedG_day.loc[:,
                                                    'lower'].sum() * 365.25 / 1e12 / denom
            _, S_up = ECDFByValue(self.Area_km2, self.est_g_day *
                                  365.25 / 1e12 / denom, reverse=False)  # re-compute
            S_up += self.extrapLAD.binnedG_day.loc[:,
                                                   'upper'].sum() * 365.25 / 1e12 / denom
            ax.fill_between(np.concatenate((geom_means, X_low)), np.concatenate((S_low0, S_low)), np.concatenate(
                (S_up0, S_up)), alpha=0.1, color=kwargs['color'])  # TODO: remove overlap line

        ## Plot main curves

        plotECDFByValue(ax=ax, alpha=1, X=X, S=S / denom,
                        normalized=False, reverse=reverse, **kwargs)
        plotECDFByValue(ax=ax, alpha=1, X=geom_means, S=S0 / denom, normalized=False, reverse=reverse,
                        linestyle='dashed', **kwargs)  # second plot is dashed for extrapolation
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
        if ax == None:
            _, ax = plt.subplots()  # figsize=(5,3)

        for var in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']:
            assert var in self.columns, f"LAD is missing {var} column, which is required to plot lev cdf."

        if normalized:
            plotECDFByValue(self.LEV_MEAN, ax=ax, alpha=0.4, color='black',
                            label='All', reverse=reverse, normalized=normalized, **kwargs)
            ax.set_ylabel('Cumulative fraction of LEV')
        else:
            plotECDFByValue(self.LEV_MEAN * self.Area_km2, ax=ax, alpha=0.4, color='black',
                            label='All', reverse=reverse, normalized=normalized, **kwargs)
            ax.set_ylabel('Cumulative LEV (million $km^2$)')
        ax.set_xlabel('LEV fraction')

        ## Legend
        if plotLegend:
            ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))

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
            assert var in self.columns, f"LAD is missing {var} column, which is required to plot lev cdf."
        sns.set_palette("colorblind", len(self.regions()))

        ## plot
        if ax is None:
            _, ax = plt.subplots()  # figsize=(5,3)

        ## Override default axes
        def makePlots(var, color='black'):
            '''Quick helper function to avoid re-typing code'''
            if normalized:
                values_for_sum = self[var]
                ylabel = 'Cumulative fraction of total LEV'
            else:
                values_for_sum = self[var] * self.Area_km2
                ylabel = 'Cumulative LEV (million $km^2$)'
            X, S = ECDFByValue(
                self.Area_km2, values_for_sum=values_for_sum, reverse=reverse)
            plotECDFByValue(X=X, S=S, ax=ax, alpha=0.6, color=color,
                            label='All', normalized=normalized, reverse=reverse, **kwargs)
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{self.name}: {self.sumLev(asFraction=True)['mean']:.2%} LEV")

        ## Run plots
        makePlots('LEV_MEAN')
        if error_bars:
            makePlots('LEV_MIN', 'grey')
            makePlots('LEV_MAX', 'grey')
        ## Legend
        if plotLegend:
            ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))

        return ax


class BinnedLAD():
    '''This class represents lakes as area bins with summed areas.'''

    def __init__(self, lad=None, btm=None, top=None, nbins=None, bins=None, normalize=True, compute_ci_lad=False, compute_ci_lev=False, compute_ci_lev_existing=False, compute_ci_flux=False, binned_areas=None, extreme_regions_lad=None, extreme_regions_lev=None):
        '''
        Bins will have left end closed and right end open.
        When creating this class to extrapolate a LAD, ensure that lad is top-truncated to ~1km to reduce variability in bin sums. This chosen top limit will be used as the upper limit to the index region (used for normalization) and will be applied to the target LAD used for extrapolation.
        When creating from an lad, give lad, btm, top [, nbins, compute_ci_lad] arguments. When creating from existing binned values (e.g.) from extrapolation, give btm, top, nbins, compute_ci_lad and binnedAreas args.

        Parameters
        ----------
        lad : LAD
            Lake-size distribution class to bin
        btm : float 
            Leftmost edge of bottom bin. Ignored if bins is given. 
        top : float
            Rightmost edge of top bin. Note: np.inf will be added to it to create one additional top bin for index region. Ignored if bins is given.
        nbins : int
            Number of bins to use for np.geomspace (not counting the top bin that goes to np.inf when 'lad' arg is given). Can use either nbins or  bins.
        bins : array-like
            Alternative to specifiying number of bins: give actual bin edges. Top edge of infinity will be added.
        normalize : Boolean (True)
            If True (default), normalized each bin by index region
        compute_ci_lad : Boolean (False)
            Compute confidence interval for lad by breaking down by region. Function will always bin LAD (at least without CI)
        compute_ci_lev : Boolean (False)
            Compute confidence interval for lev by breaking down by region. Function will always bin LEV (at least without CI)
        compute_ci_lev_existing : Boolean (False)
            Compute confidence interval for lev by using existing columns LEV_MIN and LEV_MAX. Useful for binning observed (not extrapolated) lakes data.
        compute_ci_flux : Boolean (False)
            Compute confidence interval for flux by multiplying by ci_lad CI. Function will always bin LEV (at least without CI)
        binnedAreas : pandas.DataFrame (None)
            Used for LAD.extrapolate(). Has structure similar to what is returned by self.binnedAreas if called with lad argument. Format: multi-index with two columns, first giving bins, and second giving normalized lake area sum statistic (mean, upper, lower).
        extreme_regions_lad : array-like (None)
            List of region names to use for min/max area
        extreme_regions_lev : array-like (None)
            List of region names to use for min/max LEV
        '''
        ## Assertions
        assert (nbins is None) or (
            bins is None), "Either nbins or bins must be specified, but not both."
        if (nbins is None) and (bins is None):
            nbins = 100  # set default 100 log-spaced bins if not specified
            assert btm is not None and top is not None, "If nbins is set to default, then btm and top must be specified"
        if bins is not None:  # if nbins was none, but bins was specified
            nbins = len(bins) - 1
            btm = bins[0]
            top = bins[-1]

        ## Common branch
        self.btmEdge = btm
        self.topEdge = top
        self.nbins = nbins
        self.isExtrapolated = False
        self.extreme_regions_lad = None  # init
        self.extreme_regions_lev = None  # init
        self.isNormalized = False  # init

        if lad is not None:  # sets binnedLAD from existing LAD
            assert (btm is not None and top is not None) or bins is not None, "if 'lad' argument is given, so must be 'btm' and 'top' or 'bins'"
            assert binned_areas is None, "If 'lad' is given, 'binned_areas' shouldn't be."

            # This ensures I can pass through all the attributes of parent LAD
            attrs = lad.get_public_attrs()
            cols = lad.columns.to_list()
            # create copy since deepcopy returns a pd.DataFrame
            lad = LAD(lad, **attrs, other_vars=cols)
            if bins is not None:
                self.bin_edges = np.concatenate(
                    (bins, [np.inf]))  # bins are user-specified
            else:
                self.bin_edges = np.concatenate((np.geomspace(
                    btm, top, nbins + 1), [np.inf])).round(6)  # bins computed from nbins and edges
            self.area_bins = pd.IntervalIndex.from_breaks(
                self.bin_edges, closed='left')
            lad['size_bin'] = pd.cut(lad.Area_km2, self.area_bins, right=False)
            self.hasCI_lev = False  # init

            ## # Boolean to determine branch for LEV
            hasLEV = 'LEV_MEAN' in lad.columns
            hasFlux = 'est_g_day' in lad.columns
            hasDC = 'd_counting_frac' in lad.columns

            if not hasLEV:
                self.binnedLEV = None
            if not hasFlux:
                self.binnedG_day = None
                self.binnedMg_m2_day = None

            ## Bin
            group_sums = lad.groupby(['size_bin'], observed=False).Area_km2.sum(
                numeric_only=True)  # These don't get lower/upper estimates for now
            # These don't get lower/upper estimates for now
            group_counts = lad.groupby(
                ['size_bin'], observed=False).Area_km2.count()
            if compute_ci_lad:
                assert extreme_regions_lad is not None, "If compute_ci_lad is True, extreme_regions_lad must be provided"
                assert np.all([region in lad.Region.unique() for region in extreme_regions_lad]
                              ), "One region name in extreme_regions_lad is not present in lad."

                ## First, group by area bin and take sum and counts of each bin
                group_low_sums, group_high_sums = [lad[lad['Region'] == region].groupby(
                    ['size_bin'], observed=False).Area_km2.sum() for region in extreme_regions_lad]
                self.hasCI_lad = True
                self.extreme_regions_lad = extreme_regions_lad
            else:
                ## Create a series to emulate the structure I used to use to store region confidence intervals
                group_low_sums, group_high_sums = None, None
                self.hasCI_lad = False

            ds = confidence_interval_from_extreme_regions(
                group_sums, group_low_sums, group_high_sums, name='Area_km2')
            self.binnedAreasNotNormalized = ds.copy()

            ## Normalize areas after binning
            if normalize:
                # sum of lake areas in largest bin (Careful: can't be == 0 !!)
                divisor = ds.loc[ds.index.get_level_values(0)[-1], :]
                self.isNormalized = True
            else:
                divisor = 1
            if np.any(divisor == 0):
                warn("Careful, normalizing by zero.")
            ds /= divisor

            ## Save values
            self.binnedAreas = ds
            self.binnedCounts = confidence_interval_from_extreme_regions(
                group_counts, None, None, name='Count')

            ## bin LEVbinnedDCz
            if hasLEV:
                group_means_lev = lad.groupby(
                    ['size_bin'], observed=False).LEV_MEAN.mean(numeric_only=True)
                if compute_ci_lev:
                    assert extreme_regions_lev is not None, "If compute_ci_lad is True, and LAD has LEV, extreme_regions_lev must be provided"
                    assert np.all([region in lad.Region.unique() for region in extreme_regions_lev]
                                  ), "One region name in extreme_regions_lev is not present in lad."
                    # Here, my ref distrib has no LEV_MIN/MAX, since it is observed, so I use extreme reegions for uncertainty bounds
                    group_means_lev_low, group_means_lev_high = [lad[lad['Region'] == region].groupby(
                        ['size_bin'], observed=False).LEV_MEAN.mean(numeric_only=True) for region in extreme_regions_lev]
                    self.hasCI_lev = True
                    self.extreme_regions_lev = extreme_regions_lev
                    pass
                elif compute_ci_lev_existing:
                    assert compute_ci_lev is False, "If selecting 'compute_ci_lev_existing', make sure 'compute_ci_lev' is False."
                    assert np.all(np.isin(['LEV_MIN', 'LEV_MAX'], lad.columns)
                                  ), "If selecting 'compute_ci_lev_existing', both 'LEV_MIN' and 'LEV_MAX' need to be present in lad columns."
                    group_means_lev_low, group_means_lev_high = [lad.groupby(
                        ['size_bin'], observed=False)[stat].mean(numeric_only=True) for stat in ['LEV_MIN', 'LEV_MAX']]
                else:
                    group_means_lev_low, group_means_lev_high = None, None
                    self.hasCI_lev = False

                ## Common branch: put in mean/upper/lower format
                ds_lev = confidence_interval_from_extreme_regions(
                    group_means_lev, group_means_lev_low, group_means_lev_high, name='LEV_frac')
                self.binnedLEV = ds_lev

            ## bin flux
            if hasFlux:
                ds_g_day_from_sum = lad.groupby(['size_bin'], observed=False).est_g_day.sum(
                    numeric_only=True)  # method #1: sum g/day for each lake
                self.binnedG_day = confidence_interval_from_extreme_regions(
                    ds_g_day_from_sum, None, None, name='est_g_day')  # Save for potential comparison?
                if compute_ci_flux:  # CI
                    assert compute_ci_lad is not None, "If compute_ci_flux is True, compute_ci_lad must be provided"
                    # group_means_flux0, group_means_flux_low0, group_means_flux_high0 = [lad.groupby(['size_bin']).est_mg_m2_day.mean(numeric_only=True) * self.binnedAreas.loc[:,stat] * 1e3 for stat in ['mean', 'lower', 'upper']] # method #2: would have units of g / day (don't multiply by area yet).
                    group_means_flux, group_means_flux_low, group_means_flux_high = [lad.groupby(['size_bin'], observed=False).est_mg_m2_day.mean(
                        numeric_only=True) for stat in ['mean', 'lower', 'upper']]  # Low/mean/high are all the same on a per-area basis!
                    self.hasCI_flux = True
                    pass
                else:  # This branch is silly, because neither branch computes a meaningful confidence interval. See self.PredictFlux() for actual conversion to CI based on CI of areas and given per-area fluxes
                    group_means_flux = lad.groupby(['size_bin'], observed=False).est_mg_m2_day.mean(
                        numeric_only=True)  # Low/mean/high are all the same on a per-area basis!
                    group_means_flux_low, group_means_flux_high = None, None
                    self.hasCI_flux = False
                self.binnedMg_m2_day = confidence_interval_from_extreme_regions(
                    group_means_flux, group_means_flux_low, group_means_flux_high, name='est_mg_m2_day')

            # copy attribute from parent LAD (note 'size_bin' is added in earlier this method)
            for attr in ['isTruncated', 'truncationLimits', 'name', 'size_bin']:
                setattr(self, attr, getattr(lad, attr))

            ## bin double counting
            if hasDC:
                dc = lad.groupby(['size_bin'], observed=False).d_counting_frac.mean(
                    numeric_only=True)
                binnedDC = confidence_interval_from_extreme_regions(
                    dc, None, None, name='dc')
                self.binnedDC = binnedDC
            else:
                self.binnedDC = None

            ## Check
            if group_counts.values[0] == 0:
                warn('The first bin has count zero. Did you set the lowest bin edge < the lower truncation limit of the dataset?')

        else:  # used for area extrapolation
            assert nbins is not None and compute_ci_lad is not None, "If 'binned_areas' argument is given, so must be 'nbins', and 'compute_ci_lad'."
            self.bin_edges = 'See self.refBinnedLAD'
            self.area_bins = 'See self.refBinnedLAD'
            self.hasCI_lad = compute_ci_lad  # retain from arg
            self.hasCI_lev = compute_ci_lev  # retain
            self.binnedAreas = binned_areas  # retain from arg
            self.binnedCounts = 'See self.refBinnedLAD'
            self.binnedLEV = 'See self.refBinnedLAD'
            self.binnedG_day = None
            self.binnedMg_m2_day = None
            self.hasCI_flux = None

        ## More common args at end
        self.isBinned = True
        pass

    def FromExtrap():
        '''Creates a class similar to binnedLAD,'''
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
            warn('Careful: binnedLAD is normalized, and you may be summing the top bin that gives the index region.')

        if self.hasCI_lad:
            if ci:
                return self.binnedAreas.loc[:, 'mean'].sum(), self.binnedAreas.loc[:, 'lower'].sum(), self.binnedAreas.loc[:, 'upper'].sum()
            else:
                return self.binnedAreas.loc[:, 'mean'].sum()
        else:  # no .hasCI_lad
            if ci:
                raise ValueError(
                    'BinnedLAD doesnt have a confidence interval, so it cant be included in sum.')
            else:
                return self.binnedAreas.sum()

    def plot(self, show_rightmost=False, as_lineplot=False, ax=None, as_cumulative=False):
        '''
        To roughly visualize bins.

        show_rightmost : Boolean
            Show remainder bin on right (ignored if self.isExtrapolated==True)
        '''
        if as_cumulative == True:
            assert as_lineplot == True, "If plotting as cumulative plot, be sure to indicate as_lineplot=True"
        binned_areas = self.binnedAreas.copy()  # create copy to modify
        diff = 0

        ## Remove last bin, if desired
        if self.isExtrapolated == False:
            if show_rightmost == False:  # if I need to cut off rightmost bin
                binned_areas.drop(index=binned_areas.index.get_level_values(
                    0)[-1], level=0, inplace=True)
            else:
                diff += 1  # subtract from number of bin edges to get plot x axis

        ## Plot
        if ax is None:
            _, ax = plt.subplots()
        if as_lineplot:
            assert isinstance(
                binned_areas, pd.Series), "As written, BinnedLAD.plot requires pd.Series argument is as_lineplot==True."
            xlabel = 'Lake area ($km^2$)'

            ## Normalize by total to make true PDF
            binned_areas /= binned_areas[binned_areas.index.get_level_values(
                1) == 'mean'].sum()
            geom_means = np.array(list(map(interval_geometric_mean, binned_areas[binned_areas.index.get_level_values(
                1) == 'mean'].index.get_level_values(0))))  # x-vector
            if as_cumulative:
                data = np.cumsum(
                    binned_areas[binned_areas.index.get_level_values(1) == 'mean'])
            else:
                data = binned_areas[binned_areas.index.get_level_values(
                    1) == 'mean']
            ax.plot(geom_means, data)
        else:
            xlabel = 'Bin number'

            ## Convert confidence interval vals to anomalies
            yerr = binnedVals2Error(binned_areas, self.nbins + diff)
            ax.bar(range(self.nbins),
                   binned_areas.loc[:, 'mean'], yerr=yerr, color='orange')

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
        assert hasattr(
            self, 'Temp_K'), "Binned LAD needs a Temp_K attribute in order to predict flux."
        assert self.isNormalized == False, "Binned LAD is normalized so values will be unitless for area."
        means = self.binnedAreas.loc[:, 'mean']
        geom_mean_areas = np.array(
            list(map(interval_geometric_mean, means.index)))

        ## Flux (areal, mgCH4/m2/day)
        est_mg_m2_day = 10**(model.params.Intercept +
                             model.params['np.log10(SA)'] *
                             np.log10(geom_mean_areas)
                             + model.params[temperature_metric] * self.Temp_K) - 0.01  # jja, ann, son, mam # no uncertainty yet

        ## Flux (flux rate, gCH4/day)
        est_g_day_mean, est_g_day_low, est_g_day_high = [est_mg_m2_day * self.binnedAreas.loc[:, stat] * 1e3 for stat in [
            'mean', 'lower', 'upper']]  # * 1e6 / 1e3 # (convert km2 -> m2 and mg -> g)
        est_g_day = confidence_interval_from_extreme_regions(
            est_g_day_mean, est_g_day_low, est_g_day_high, name='est_g_day')
        self._total_flux_Tg_yr = est_g_day.groupby(
            'stat', observed=False).sum() * 365.25 / 1e12  # see Tg /yr

        ## Add attrs
        self.binnedMg_m2_day = confidence_interval_from_extreme_regions(pd.Series(
            est_mg_m2_day, index=est_g_day.loc[:, 'mean'].index), None, None, name='est_mg_m2_day')  # in analogy with binnedAreas and binnedCounts
        # in analogy with binnedAreas and binnedCounts # HERE: is this updated value the same as the one computed in BinnedLAD.init?
        self.binnedG_day = est_g_day

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
        summed_lev = (
            self.binnedLEV * self.binnedAreas.loc[:, 'mean']).groupby(level=1, observed=False).sum()  # km2
        if asFraction:
            result = summed_lev / self.sumAreas(ci=False)
        else:
            result = summed_lev
        return result

    def to_df(self):
        '''
        Generates a pandas DataFrame by concatennating the binnedAreas, binnedLEV, binnedG_day, binnedMg_m2_day fields into a table.
        Includes central and upper/lower estimates.
        '''
        table = pd.concat((self.binnedAreas, self.binnedLEV,
                          self.binnedG_day, self.binnedMg_m2_day), axis=1)
        table.rename(columns={'LEV_frac': 'LAV_frac'}, inplace=True)
        return table


def combineBinnedLADs(lads):
    '''Combines two or more BinnedLADs in the tuple lads and returns a new BinnedLAD.'''

    ## Concatenate binnedAreas and init BinnedLAD class
    combined_binned_areas = pd.concat(
        [lad.binnedAreas.reset_index() for lad in lads]).set_index(['size_bin', 'stat'])
    nbins = np.sum([lad.nbins for lad in lads])
    cmb_binned_lad = BinnedLAD(binned_areas=combined_binned_areas, nbins=nbins, compute_ci_lad=False,
                               compute_ci_lev=False)

    ## Add in binned LEV, flux attributes
    cmb_binned_lad.binnedLEV = pd.concat([lad.binnedLEV.reset_index(
    ) for lad in lads], axis=0).set_index(['size_bin', 'stat'])
    cmb_binned_lad.binnedG_day = pd.concat([lad.binnedG_day.reset_index(
    ) for lad in lads], axis=0).set_index(['size_bin', 'stat'])
    cmb_binned_lad.binnedMg_m2_day = pd.concat([lad.binnedMg_m2_day.reset_index(
    ) for lad in lads], axis=0).set_index(['size_bin', 'stat'])
    cmb_binned_lad.binnedDC = pd.concat([lad.binnedDC.reset_index(
    ) for lad in lads], axis=0).set_index(['size_bin', 'stat'])
    cmb_binned_lad.binnedCounts = pd.concat([lad.binnedCounts.reset_index(
    ) for lad in lads], axis=0).set_index(['size_bin', 'stat'])

    return cmb_binned_lad

## Custom grouping function (Thanks ChatGPT!)


def interval_group(interval, edges=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, np.inf]):
    ''' Custom grouping function that aggregates pd.Interval indexes based on edges given by edges. Combines bins in histograms.'''
    n = len(edges)
    assert interval.left != interval.right
    for i, edge in enumerate(edges[:-1]):
        if (interval.left >= edge) and (interval.right <= edges[i + 1]):
            return pd.Interval(left=edge, right=edges[i + 1])
    return None  # if interval doesn't fit between any two adjacent edges


def norm_table(table, mean_table):
    ''' Normalizes a table by the sum of another table.'''
    return table / mean_table.sum(axis=0)


def runTests():
    '''Legacy tests that still need to be added to tests/test_LAD.py pytest framework. Can run with `python LAD.py --test True` '''

    lad_hr_pth = '/Users/ekyzivat/Library/CloudStorage/Dropbox/Python/Ch4/sample_data/CIR_Canadian_Shield.shp'
    lad_lr_pth = '/Users/ekyzivat/Library/CloudStorage/Dropbox/Python/Ch4/sample_data/HydroLAKESv10_Sweden.shp'

    # ## Loading from dir
    # exclude = ['arg0022009xxxx', 'fir0022009xxxx', 'hbl00119540701','hbl00119740617', 'hbl00120060706', 'ice0032009xxxx', 'rog00219740726', 'rog00220070707', 'tav00119630831', 'tav00119750810', 'tav00120030702', 'yak0012009xxxx', 'bar00120080730_qb_nplaea.shp']
    # lad_from_dir = LAD.from_paths('/Volumes/thebe/PeRL/PeRL_waterbodymaps/waterbodies/y*.shp', area_var='AREA', name='perl', _areaConversionFactor=1000000, exclude=exclude)

    ## Load with proper parameters
    # lad_hl = LAD.from_shapefile('/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/HL_Sweden_md.shp', area_var='Lake_area', idx_var='Hylak_id', name='HL', region_var=None)
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin',
               'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lad_cir = LAD.from_shapefile('/Volumes/thebe/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp',
                                 area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

    ####################################
    ## LEV Tests
    ####################################

    ########## Workflow for creating sample data with occurrence values
    ## Save to sample data
    # gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lad_hl_oc.Pour_long, lad_hl_oc.Pour_lat))
    # gdf.plot()

    # # Extract the list of 'hylak_id' from df_lad_from_shp['idx_HL']
    # idx_HL_list = lad_from_shp['idx_HL'].tolist()

    # # Filter df_lad_hl_oc based on the condition
    # filtered_df = lad_hl_oc[lad_hl_oc['Hylak_id'].isin(idx_HL_list)]

    # gdf_filt = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df.Pour_long, filtered_df.Pour_lat), crs='EPSG:4326')

    # gdf_filt_og = gpd.GeoDataFrame(lad_from_shp, geometry=gpd.points_from_xy(lad_from_shp.Pour_long, lad_from_shp.Pour_lat), crs='EPSG:4326')

    # gdf_filt.to_file('/Users/ekyzivat/Library/CloudStorage/Dropbox/Python/Ch4/sample_data/HydroLAKESv10_Sweden_Occurrence.shp')

    # gdf_filt.to_csv('/Users/ekyzivat/Library/CloudStorage/Dropbox/Python/Ch4/sample_data/HydroLAKESv10_Sweden_Occurrence.csv.gz', compression='gzip')

    ###########

    ## Test 1: Bin the reference UAVSAR LEV LADs
    # lad_lev_binneds = []
    # for i, lad_lev in enumerate(lad_levs):
    #     lad_lev_binneds.append(BinnedLAD(lad_lev, 0.000125, 0.5).binnedLEV)
    # pd.DataFrame(lad_lev_binneds).T

    ## Test 2: Concat all ref LEV distributions and bin
    # lad_lev_cat = LAD.concat(lad_levs, broadcast_name=True)
    # def ci_from_named_regions(LAD, regions):
    binned_lev = BinnedLAD(lad_lev_cat, 0.0001, 0.5, compute_ci_lev=True,
                           extreme_regions_lev=extreme_regions_lev_for_extrap)  # 0.000125 is native

    ####################################
    ## LAD/bin/extrap Tests
    ####################################

    ## Test binnedLAD with fluxes
    model = loadBAWLD_CH4()
    lad_cir['Temp_K'] = 10  # placeholder, required for prediction
    lad_cir.predictFlux(model, includeExtrap=False)
    binned = BinnedLAD(lad_cir.truncate(0.0001, 1), 0.0001, 0.5, compute_ci_lad=True, compute_ci_flux=True,
                       extreme_regions_lad=extreme_regions_lad)  # compute_ci_lad=False will disable plotting CI.

    ## Compare extrapolated sums
    lad_hl_trunc.extrapLAD.sumAreas()
    lad_hl_trunc.sumAreas()

    ## Compare extrapolated area fractions
    # frac = lad_hl_trunc.extrapolated_area_fraction(lad_cir, 0.0001, 0.01)
    # print(frac)
    # # lad_hl_trunc.extrapolated_area_fraction(lad_cir, 0.00001, 0.01) # test for bin warning
    # # lad_hl_trunc.extrapolated_area_fraction(lad_cir, 0.0001, 1)# Test for limit error

    ## Plot
    # lad_hl_trunc.extrapLAD.plot()
    # ax = lad_hl_trunc.plot_lad(reverse=False, normalized=True)
    # lad_hl_trunc.plot_extrap_lad(normalized=True, error_bars=False, reverse=False) # ax=ax,
    # lad_hl_trunc.plot_extrap_lad(normalized=False, error_bars=True, reverse=False) # ax=ax,

    ####################################
    ## Flux Tests
    ####################################

    ## Load climate
    bawld_join_clim_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1___Shapefile_jn_clim.csv'
    gdf_bawld_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
    df_clim = pd.read_csv(bawld_join_clim_pth)
    gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio')
    df_clim = df_clim.merge(
        gdf_bawld[['Cell_ID', 'Shp_Area']], how='left', on='Cell_ID')

    ## Test flux prediction from observed lakes
    model = loadBAWLD_CH4()
    # lad_hl_trunc['Temp_K'] = 10 # placeholder until I load from file, required for prediction
    # some rows don't merge, I think bc I think idx_unamed is nto for HL... just use as example
    temp = lad_hl_trunc.merge(
        df_clim, how='left', left_on='idx_unamed', right_on='Cell_ID')
    temperature_metric = 'jja'
    lad_hl_trunc['Temp_K'] = temp[temperature_metric]
    lad_hl_trunc.predictFlux(model, includeExtrap=False)

    ## Test flux prediction from extrapolated lakes
    # Placeholder # np.average(lad_hl_trunc.Temp_K, weights=lad_hl_trunc.Area_km2) # Use lake-area-weighted average temp as temp for extrapolated lakes, since we don't know where they are located
    lad_hl_trunc.extrapLAD.Temp_K = 9.8 + 273.15
    lad_hl_trunc.extrapLAD.predictFlux(model)

    ## Test combined prediction
    lad_hl_trunc.predictFlux(model, includeExtrap=True)

    ## Test plot fluxes
    # lad_hl_trunc.plot_flux(reverse=False, normalized=True, all=False)
    # lad_hl_trunc.plot_flux(reverse=False, normalized=False, all=False)

    ## Test plot extrap fluxes
    lad_hl_trunc.plot_extrap_flux(
        reverse=False, normalized=False, error_bars=True)
    lad_hl_trunc.plot_extrap_flux(
        reverse=False, normalized=True, error_bars=True)

    pass
