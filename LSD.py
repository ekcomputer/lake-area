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
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.colors as colors
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndi 
import pandas as pd
import geopandas as gpd
import pyogrio
import powerlaw
from tqdm import tqdm

## Plotting style
# plt.style.use('/mnt/c/Users/ekyzivat/Dropbox/Python/Matplotlib-rcParams/presentation.mplstyle')
# %matplotlib inline

## Plotting params
sns.set_theme('notebook', font='Ariel')
sns.set_style('ticks')

## I/O

## BAWLD domain
dataset = 'HL'
roi_region = 'BAWLD'
gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
gdf_HL_jn_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_binned_jnBAWLD.shp' # HL clipped to BAWLD
version=3

## BAWLD-NAHL domain
# dataset = 'HL'
# roi_region = 'WBD_BAWLD'
# gdf_bawld_pth = '/mnt/g/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1_clipped_to_WBD.shp'
# gdf_HL_jn_pth = '/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_binned_jnBAWLD_roiNAHL.shp' # HL clipped to BAWLD and WBD
# version=1

## dynamic vars
analysis_dir = os.path.join('/mnt/g/Ch4/Area_extrapolations','v'+str(version))
area_extrap_pth = os.path.join(analysis_dir, dataset+'_sub'+roi_region+'_extrap.csv')
os.makedirs(analysis_dir, exist_ok=True)

# ## Functions and classes
# ### Plotting functions

def findNearest(arr, val):
    ''' Function to find index of value nearest to target value'''
    # calculate the difference array
    difference_array = np.absolute(arr-val)
    
    # find the index of minimum element from the array
    index = difference_array.argmin()
    return index

def ECDFByValue(values, reverse=True):
    """
    Returns sorted values and their cumsum.
    Called by plotECDFByValue.

    Parameters
    ----------
    values (array-like) : Values for histogram

    Returns
    -------
    X : sorted array
    X : cumsum of sorted array

    """
    if reverse:
        X = np.sort(values)[-1::-1] # highest comes first bc I reversed order
    else:
        X = np.sort(values)
    S = np.cumsum(X) # cumulative sum, starting with highest [lowest, if reverse=False] values
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
    if X is None and S is None:
        X, S = ECDFByValue(values, reverse=reverse) # compute

    if normalized:
        S = S/S[-1] # S/np.sum(X) has unintended result when using for extrapolated, when S is not entirely derived from X
        ylabel = 'Cumulative fraction of total area'
    else:
        ylabel = 'Cumulative area (km2)'
    if not ax:
        _, ax = plt.subplots()
    ax.plot(X, S, **kwargs) 

    ## Viz params
    ax.set_xscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Lake area')
    return

def plotEPDFByValue(values, ax=None, **kwargs):
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
    # return S

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

def binnedVals2Error(binned_values, n):
    '''Convert BinnedLSD.binnedValues to error bars for plotting. May need to subtract 1 from n.'''
    ci = binned_values.loc[:, ['lower', 'upper']].to_numpy().reshape((n,2)).T
    mean = binned_values.loc[:, ['mean']].to_numpy()
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
   
## Class (using inheritance)
class LSD(pd.core.frame.DataFrame): # inherit from df? pd.DataFrame # 
    '''Lake size distribution'''
    def __init__(self, df, name=None, area_var=None, region_var=None, idx_var=None, name_var=None, _areaConversionFactor=1, regions=None, computeArea=False, **kwargs):
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
        areaConversionFactor : float, optional, defaults to 1.
            Denominator for unit conversion. 1 for km2, 1e6 for m2
        regions : list, optional
            If provided, will transform numeric regions to text
        computeArea : Boolean, default:False
            If provided, will compute Area from geometry. Doesn't need a crs, but needs user input for 'areaConversionFactor.'
        ''' 

        ## Add default name
        if name == None:
            name='unamed'

        ## Check if proper column headings exist (this might occur if I am initiating from a merged group of existing LSD objects)
        if 'idx_'+name in df.columns:
            idx_var = 'idx_'+name
        if 'Area_km2' in df.columns:
            area_var = 'Area_km2'
        if 'Region' in df.columns:
            region_var = 'Region'
        if 'geometry' in df.columns: # if I am computing geometry
            geometry_var = 'geometry'
        else: geometry_var = None

        ## Choose which columns to keep (based on function arguments, or existing vars with default names that have become arguments)
        columns = [col for col in [idx_var, area_var, region_var, name_var, geometry_var] if col is not None] # allows 'region_var' to be None
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
            self.rename(columns={idx_var:'idx_'+name, area_var:'Area_km2', region_var:'Region'}, inplace=True)
        else:
            self.rename(columns={idx_var:'idx_'+name, area_var:'Area_km2'}, inplace=True)

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
    def from_shapefile(cls, path, name=None, area_var=None, region_var=None, idx_var=None, **kwargs): #**kwargs): #name='unamed', area_var='Area', region_var='NaN', idx_var='OID_'): # **kwargs
        ''' 
        Load from shapefile on disk.
        Accepts all arguments to LSD.
        '''
        columns = [col for col in [idx_var, area_var, region_var] if col is not None] # allows 'region_var' to be None
        read_geometry = False
        if 'computeArea' in kwargs:
            if kwargs['computeArea']==True:
                read_geometry = True
        df = pyogrio.read_dataframe(path, read_geometry=read_geometry, use_arrow=True, columns=columns)
        if name is None:
            name = os.path.basename(path).replace('.shp','').replace('.zip','')
        return cls(df, name=name, area_var=area_var, region_var=region_var, idx_var=idx_var, **kwargs)
    
    @classmethod
    def from_paths(cls, file_pattern, name='unamed', area_var=None, region_var=None, idx_var=None, exclude=None, **kwargs):
        '''Load in serial with my custom class
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
            lsd = cls.from_shapefile(shpfile, area_var=area_var, region_var=region_var, idx_var=idx_var, **kwargs)
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
            lsd = LSD(pd.DataFrame.query(self, "(Area_km2 >= @min) and (Area_km2 < @max)", inplace=inplace, **kwargs), **attrs) # false    
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

        TODO: add option to include extrapolation in estimate.
        '''
        attr = f'_A_{limit}' # dynamically-named attribute (_ prefix means it won't get copied over after a truncation or concat)
        if attr in self.get_public_attrs():
            return getattr(self, attr)
        else:
            area_fraction = self.truncate(0, limit).Area_km2.sum()/self.Area_km2.sum()
            setattr(self, attr, area_fraction)
            return area_fraction
    
    def extrapolate(self, ref_BinnedLSD, bottomLim):
        '''
        Extrapolate by filling empty bins below the dataset's resolution.

        Note: The limits of the extrapolation are defined by the btmEdge/topEdge of ref_BinnedLSD. The top limit of rererence distribution is defined by the top truncation of the reference binnedLSD. Function checks to make sure it is truncated <= 1 km2.
        
        Parameters
        ----------
        refBinnedLSD : binnedLSD
            Reference binnedLSD used for extrapolation.
        bottomLim : float
            Bottom size limit to which to extrapolate.

        ''' 
        # returns a binnedLSD   
        # give error if trying to extrapolate to a smaller area than is present in ref distrib

        # HERE truncate last bin

        ## Check validity
        # assert self.isTruncated # can correct for this
        assert ref_BinnedLSD.isTruncated, "Reference binnedLSD must be top truncated or its bin estimates will be highly variable."
        assert self.isTruncated, "LSD should be bottom-truncated when used for extrapolation to be explicit."
        assert self.truncationLimits[0] == ref_BinnedLSD.topEdge, f"Mismatch between LSD bottom truncation limit ({self.truncationLimits[0]} and ref binned LSD top edge ({ref_BinnedLSD.topEdge}))"
        assert bottomLim >= ref_BinnedLSD.btmEdge, f"You are trying to extrapolate to {bottomLim}, but ref binned LSD only goes to {ref_BinnedLSD.topEdge}." 
        assert ref_BinnedLSD.isNormalized
        assert ref_BinnedLSD.hasCI, "This method requires ref_BinnedLSD to have a confidence interval (as currently written)."
        if ref_BinnedLSD.truncationLimits[1] > 1:
            warn("Reference binned LSD should be top truncated to ~1 km2.")
        
        ## Perform the extrapolation (bin self by its bottom truncation limit to the indexTopLim, and simply multiply it by the normalized refBinnedLSD to do the extrapolation)!
        index_region_sum = self.truncate(ref_BinnedLSD.topEdge, ref_BinnedLSD.truncationLimits[1]).Area_km2.sum() # Sum all area in LSD in the index region, defined by the topEdge and top-truncation limit of the ref LSD.
        last = ref_BinnedLSD.binnedValues.index.get_level_values(0).max() # last index with infinity to drop (use as mask)
        binned_values = ref_BinnedLSD.binnedValues.drop(index=last) * index_region_sum # remove the last entries, which are mean, lower, upper for the bin that goes to np.inf
        
        ## Return in place a new attribute called extrapLSD which is an instance of a binnedLSD
        self.extrapLSD = BinnedLSD(btm=ref_BinnedLSD.btmEdge, top=ref_BinnedLSD.topEdge, nbins=ref_BinnedLSD.nbins, has_ci=ref_BinnedLSD.hasCI, binned_values=binned_values)
       
        ## Update its attributes
        self.extrapLSD.indexTopLim = ref_BinnedLSD.truncationLimits[1] # since bottom of index region matches, ensure top does as well in definition.
        self.extrapLSD.isExtrapolated=True
        self.extrapLSD.isNormalized = False # units of km2, not fractions
        self.extrapLSD.bottomLim = bottomLim
        self.extrapLSD.topLim = ref_BinnedLSD.topEdge

        ## Save reference binned LSD
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
        measured_sum = self.Area_km2.sum()
        if includeExtrap:
            assert hasattr(self, 'extrapLSD'), "LSD doesn't include an extrapLSD attribute."
            if ci==False:
                return self.extrapLSD.sumAreas(ci=ci) + measured_sum
            else:
                return tuple((np.array(self.extrapLSD.sumAreas(ci=ci)) + measured_sum)) # convert to tuple, as is common for python fxns to return
        else:
            return measured_sum
    def predictFlux(self, temp):
        '''
        Predict methane flux based on area bins and temperature
        '''
        pass

    ## Plotting
    def plot_lsd(self, all=True, plotLegend=True, groupby_name=False, cdf=True, ax=None, **kwargs):
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
        sns.set_palette("colorblind", len(self.regions())) # colors from https://stackoverflow.com/a/46152327/7690975 Other option is: `from cycler import cycler; `# ax.set_prop_cycle(rainbow_cycler), plt(... prop_cycle=rainbow_cycler, )

        ## plot
        if ax==None:
            _, ax = plt.subplots() # figsize=(5,3)

        if groupby_name==False: # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                plotECDFByValue(pd.DataFrame.query(self, 'Region == @region').Area_km2, ax=ax, alpha=0.4, label=region, **kwargs)

        else:
            assert 'Name' in self.columns, "LSD is missing 'Name' column."
            # cmap = colors.Colormap('Pastel1')
            names = np.unique(self['Name'])
            cmap = plt.cm.get_cmap('Paired', len(names))
            for j, name in enumerate(names):
                for i, region in enumerate(np.unique(pd.DataFrame.query(self, 'Name == @name').Region)): # can't use .regions() after using DataFrame.query because it returns a DataFrame
                    plotECDFByValue(pd.DataFrame.query(self, 'Region == @region').Area_km2, ax=ax, alpha=0.6, label=name, color=cmap(j), **kwargs)

        ## repeat for all
        if all:
            plotECDFByValue(self.Area_km2, ax=ax, alpha=0.4, color='black', label='All', **kwargs)

        ## Legend
        if plotLegend:
            ax.legend(loc= 'center left', bbox_to_anchor=(1.04, 0.5)) # legend on right (see https://stackoverflow.com/a/43439132/7690975)

        return ax
    
    def plot_extrap_lsd(self, plotLegend=True, ax=None, normalized=False, error_bars=False, **kwargs):
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
        X, S = ECDFByValue(self.Area_km2, reverse=False)
        geom_means = np.array(list(map(interval_geometric_mean, self.extrapLSD.binnedValues.loc[:, 'mean'].index))) # take geom mean of each interval
        X = np.concatenate((geom_means, X))
        S += self.extrapLSD.sumAreas() # add to original cumsum
        S = np.concatenate((np.cumsum(self.extrapLSD.binnedValues[:, 'mean']), S)) # pre-pend the binned vals

        ## Add error bars
        if error_bars == True and self.extrapLSD.hasCI:

            ## as error bars (miniscule)
            # yerr = binnedVals2Error(self.extrapLSD.binnedValues, self.extrapLSD.nbins-1)
            # # yerr = np.concatenate((np.cumsum(yerr[0])[np.newaxis, :], np.cumsum(yerr[1])[np.newaxis, :])) # don't need to cumsum errors
            # ax.errorbar(geom_means, np.cumsum(self.extrapLSD.binnedValues.loc[:, 'mean']), xerr=None, yerr=yerr, fmt='none', )

            ## as area plot
            ax.stackplot(geom_means, self.extrapLSD.binnedValues.loc[:, 'upper'] - self.extrapLSD.binnedValues.loc[:, 'lower'], )
        
        ## Plot
        plotECDFByValue(ax=ax, alpha=0.4, color='black', X=X, S=S, normalized=normalized, reverse=False, **kwargs)
        ax.legend()

        return ax

class BinnedLSD():
    '''This class represents lakes as area bins with summed areas.'''
    def __init__(self, lsd=None, btm=None, top=None, nbins=100, has_ci=True, binned_values=None):
        '''
        Bins will have left end closed and right end open.
        When creating this class to extrapolate a LSD, ensure that lsd is top-truncated to ~1km to reduce variability in bin sums. This chosen top limit will be used as the upper limit to the index region (used for normalization) and will be applied to the target LSD used for extrapolation.
        When creating from an lsd, give lsd, btm, top [, nbins, has_ci] arguments. When creating from existing binned values (e.g.) from extrapolation, give btm, top, nbins, has_ci and binnedValues args.
        
        Parameters
        ----------
        lsd : LSD
            Lake-size distribution class to bin
        btm : float 
            Leftmost edge of bottom bin 
        top : float
            Rightmost edge of top bin. Note: np.inf will be added to it to create one additional top bin.
        nbins : int
            Number of bins to use for np.geomspace (not counting the top bin that goes to np.inf when 'lsd' arg is given).
        has_ci : Boolean
            Compute confidence interval by breaking down by region.
        binnedValues : pandas.DataFrame
            Used for LSD.extrapolate(). Has structure similar to what is returned by self.binnedValues if called with lsd argument. Format: multi-index with two columns, first giving bins, and second giving normalized lake area sum statistic (mean, upper, lower).

        '''
        ## Common branch
        self.btmEdge = btm
        self.topEdge = top
        self.nbins = nbins
        self.isExtrapolated = False

        if lsd is not None: # sets binnedLSD from existing LSD
            assert btm is not None and top is not None, "if 'lsd' argument is given, so must be 'btm' and 'top'"
            assert binned_values is None, "If 'lsd' is given, 'binned_values' shouldn't be."

            attrs = lsd.get_public_attrs() # This ensures I can pass through all the attributes of parent LSD
            lsd = LSD(lsd, **attrs) # create copy
            self.bin_edges = np.concatenate((np.geomspace(btm, top, nbins+1), [np.inf])).round(6)
            self.area_bins = pd.IntervalIndex.from_breaks(self.bin_edges, closed='left')
            lsd['labels'] = pd.cut(lsd.Area_km2, self.area_bins, right=False)
            self.isNormalized = False # init

            ## Bin
            if has_ci:
                ## First, group by region and area bin and take sum and counts of each bin
                group_sums = lsd.groupby(['Region', 'labels']).Area_km2.sum(numeric_only=True)
                group_counts = lsd.groupby(['Region', 'labels']).Area_km2.count()

                ## Normalize before binning
                # self.normalize()
                divisor = group_sums.loc[:, group_sums.index.get_level_values(1)[-1]] # sum of lake areas in largest bin for each region (Careful: can't be == 0 !!) ## HERE: remove regions that create infs (or do it in extrapolate() )!
                group_sums /= divisor

                ## Remove any regions with w/o lakes in the index size bin that cause dividing by zero
                group_sums = group_sums.loc[(divisor.loc[divisor!=0]).index] # or could simply drop all infs, since every bin in a region will be inf if one bin is
                group_counts = group_counts.loc[(divisor.loc[divisor!=0]).index]
                
                self.isNormalized = True

                # Compute the mean and confidence interval for each group along the second index
                self.binnedValues = group_sums.groupby(level=1).apply(confidence_interval)
                self.binnedCounts = group_counts.groupby(level=1).sum()
                self.hasCI = True

            else: # not maintaining this branch, could use it to avoid throwing out data from regions w/o lakes in the index size bin
                self.binnedValues = lsd.groupby('labels').Area_km2.sum(numeric_only=True)
                self.hasCI = False
                self.binnedCounts = np.nan
            for attr in ['isTruncated', 'truncationLimits', 'name', 'labels']: # copy attribute from parent LSD (note 'labels' is added in earlier this method)
                setattr(self, attr, getattr(lsd, attr))
            
            ## Check
            if self.binnedCounts.values[0] == 0:
                warn('The first bin has count zero. Did you set the lowest bin edge < the lower truncation limit of the dataset?')   
                    
        else: # used for extrapolation
            assert btm is not None and top is not None and nbins is not None and has_ci is not None, "If 'binned_values' argument is given, so must be 'btm', 'top', 'nbins', and 'has_ci'."
            self.bin_edges = 'See self.refBinnedLSD'
            self.area_bins = 'See self.refBinnedLSD'
            self.isNormalized = False
            self.hasCI = has_ci # retain from arg
            self.binnedValues = binned_values # retain from arg
            self.binnedCounts = 'See self.refBinnedLSD'
        
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
        Sum the areas within the dataframe self.binnedValues.

        If ci==True, returns the mean, lower and upper estimate. Otherwise, just the mean.
        
        ci : Boolean
            Whether to output the lower and upper confidence intervals.
        '''
        if ci:
            return self.binnedValues.loc[:,'mean'].sum(), self.binnedValues.loc[:,'lower'].sum(), self.binnedValues.loc[:,'upper'].sum()
        else:
            return self.binnedValues.loc[:,'mean'].sum()
    
    def plot(self, show_rightmost=False):
        '''
        To roughly visualize bins.
        
        show_rightmost : Boolean
            Show remainder bin on right (ignored if self.isExtrapolated==True)
        '''
        binned_values = self.binnedValues.copy() # create copy to modify
        diff=0
        ## Remove last bin, if desired
        if self.isExtrapolated==False:
            if show_rightmost == False: # if I need to cut off rightmost bin
                binned_values.drop(index=binned_values.index.get_level_values(0)[-1], inplace=True)
                diff+=1 # subtract from number of bin edges to get plot x axis
        else: # if extrapolated
            diff+=1
        
        ## Plot
        fig, ax = plt.subplots()
        # plt.bar(self.bin_edges[:-1], binned_values)

        if self.hasCI:
            ## Convert confidence interval vals to anomalies
            yerr = binnedVals2Error(binned_values, self.nbins)
            ax.bar(range(self.nbins), binned_values.loc[:, 'mean'], yerr=yerr, color='orange') 
        else:  # Needs testing
            ax.bar(range(self.nbins), binned_values)
        
        ax.set_yscale('log')
        ax.set_xlabel('Bin number')
        if self.isNormalized:
            ax.set_ylabel('Fraction of large lake area')
        else:
            ax.set_ylabel('km2')
        return

    def predictFlux(self, temp):
        '''
        Predict methane flux based on area bins and temperature
        '''
        pass
    

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
    lsd_hl = LSD.from_shapefile('/mnt/f/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/HL_Sweden_md.shp', area_var='Lake_area', idx_var='Hylak_id', name='HL', region_var=None)
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lsd_cir = LSD.from_shapefile('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp', area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

    ## Test binnedLSD
    binned = BinnedLSD(lsd_cir.truncate(0.0001,1), 0.0001, 0.1)
    binned.plot()

    ## Test extrapolate on small data
    lsd_hl.truncate(0.1, np.inf, inplace=True) # Beware chaining unless I return a new variable.
    lsd_hl.extrapolate(binned, 0.0001)
    
    ## Compare extrapolated sums
    lsd_hl.extrapLSD.sumAreas()
    lsd_hl.sumAreas()

    ## Plot
    lsd_hl.extrapLSD.plot()
    ax = lsd_hl.plot_lsd(reverse=False, normalized=False)
    lsd_hl.plot_extrap_lsd(ax=ax, normalized=False)
    pass

## Testing mode or no.
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False,
                    help="Whether to run in test mode or not (default=False)")
args = parser.parse_args()
if args.test == 'True':
    print('Test mode.')
    __name__ ='__test__'

if __name__=='__test__':
    runTests()
    
if __name__=='__main__':

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

    ## plot
    # lsd.truncate(0.0001, 10).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=True)

    # ## YF compare
    # LSD(lsd.query("Region=='YF_3Y_lakes-and-ponds' or Region=='Yukon Flats Basin'"), name='compare').plot_lsd(all=False, plotLegend=True, reverse=False, groupby_name=False)

    ## Load WBD
    print('Load WBD...')
    lsd_wbd = LSD.from_shapefile('/mnt/g/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', area_var='Area', name='WBD', idx_var='OBJECTID')
    lsd_wbd.truncate(0.001, inplace=True)

    ## Plot WBD
    # lsd_wbd.plot_lsd(reverse=False, all=False)

    ## Combine WBD with HR dataset for plotting comparison
    # setattr(lsd, 'name', 'HR datasets')
    # lsd['Region'] = 'NaN' # Hot fix to tell it not to plot a curve for each region
    # lsd_compare = LSD.concat((lsd.truncate(0.001, 50), lsd_wbd.truncate(0.001, 50)), broadcast_name=True, ignore_index=True)
    # lsd_compare.plot_lsd(all=False, plotLegend=True, reverse=False, groupby_name=True)

    ## Estimate area fraction
    # lsd_wbd.area_fraction(0.1)
    # lsd_wbd.area_fraction(0.01)
    # lsd_wbd.area_fraction(0.001)

    # lsd.area_fraction(0.1)
    # lsd.area_fraction(0.01)
    # lsd.area_fraction(0.001)  

    ## Load hydrolakes
    print('Load HL...')
    lsd_hl = LSD.from_shapefile('/mnt/g/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_binned_jnBAWLD.shp', area_var='Shp_Area', idx_var='Hylak_id', name='HL', region_var=None)

    ## Extrapolate
    tmin, tmax = (0.001,30)
    emin, emax = (tmin, 0.5)
    binned_ref = BinnedLSD(lsd.truncate(tmin, tmax), emin, emax) # reference distrib (try 5, 0.5 as second args)
    lsd_hl.truncate(emax, np.inf, inplace=True) # Beware chaining unless I return a new variable. # Try 0.1
    lsd_hl.extrapolate(binned_ref, tmin)

    ## Plot to verify HL extrapolation
    ax = lsd_hl.plot_lsd(all=False, reverse=False, normalized=False)
    lsd_hl.plot_extrap_lsd(ax=ax, label='HL-extrapolated')
    ax.set_title(f'truncate: ({tmin}, {tmax}), extrap: ({emin}, {emax})')

    ## print number of ref lakes:
    len(lsd_hl)
    lsd_hl.refBinnedLSD.binnedCounts.sum()
    lsd_hl.extrapLSD.sumAreas()

    ## Compare extrapolation to WBD:
    # lsd_hl.truncate(0, 1000).plot_lsd(all=False, reverse=False, normalized=False)
    ax = lsd_wbd.truncate(0.001, 1000).plot_lsd(all=False, reverse=False, normalized=True, color='r')
    lsd_hl.truncate(0.001, 1000).plot_extrap_lsd(label='HL-extrapolated', normalized=True, ax=ax)
    ax.set_title(f'truncate: ({tmin}, {tmax}), extrap: ({emin}, {emax})')

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

NOTES:
* Every time a create an LSD() object in a function from an existing LSD (e.g. making a copy), I should pass it the public attributes of its parent, or they will be lost.
'''
