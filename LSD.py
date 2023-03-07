#!/home/ekyzivat/mambaforge/envs/geospatial/bin/python
# # Lake-size distribution (LSD) scale comparison.
# 
# Goal: to load lake maps from the same region at two scale (HR and LR) and predict the small water body coverage (defined as area < 0.001 or 0.01 km2) from the LR dataset and physiographic region (with uncertainty).
# 
# Steps:
# 1. plot both LSD as survivor functions in log-log space (see functions from TGRS paper)

## Imports
import argparse
import os
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

def plotECDFByValue(values, reverse=True, ax=None, **kwargs):
    '''Cumulative histogram by value (lake area), not count. Creates, but doesn't return fig, ax if they are not provided. By default, CDF order is reversed to emphasize addition of small lakes (flips over ahorizontal axis).'''
    if reverse:
        X = np.sort(values)[-1::-1] # highest comes first bc I reversed order
    else:
        X = np.sort(values)
    S = np.cumsum(X) # cumulative sum, starting with highest [lowest, if reverse=False] values
    if not ax:
        _, ax = plt.subplots()
    ax.plot(X, S/np.sum(X), **kwargs) 

    ## Viz params
    ax.set_xscale('log')
    ax.set_ylabel('Cumulative fraction of total area')
    ax.set_xlabel('Lake area')
    # return S

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
        public_attrs = {}
        for attr, value in self.__dict__.items():
            if not attr.startswith('_'):
                public_attrs[attr] = value
        return public_attrs
    
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
        '''
        attr = f'_A_{limit}' # dynamically-named attribute (_ prefix means it won't get copied over after a truncation or concat)
        if attr in self.get_public_attrs():
            return getattr(self, attr)
        else:
            area_fraction = self.truncate(0, limit).Area_km2.sum()/self.Area_km2.sum()
            setattr(self, attr, area_fraction)
            return area_fraction
    
    def extrapolate(self, refBinnedLSD, bottomLim, topLim):
        '''
        Extrapolate by filling empty bins below the dataset's resolution
        ''' 
        # returns a binnedLSD   
        # give error if trying to extrapolate to a smaller area than is present in ref distrib
        pass
    
    def predictFlux(self, temp):
        '''
        Predict methane flux based on area bins and temperature
        '''
        pass

    ## Plotting
    def plot_lsd(self, all=True, plotLegend=True, groupby_name=False, cdf=True, **kwargs):
        '''
        Calls plotECDFByValue and sends it any remaining argumentns (e.g. reverse=False)
        groupby_name : boolean
            Group plots by dataset name, given by variable 'Name'
        '''
        ## Cumulative histogram by value (lake area), not count
        
        ## colors
        # rainbow_cycler = cycler
        sns.set_palette("colorblind", len(self.regions())) # colors from https://stackoverflow.com/a/46152327/7690975 Other option is: `from cycler import cycler; `# ax.set_prop_cycle(rainbow_cycler), plt(... prop_cycle=rainbow_cycler, )

        ## plot
        fig, ax = plt.subplots() # figsize=(5,3)

        if groupby_name==False: # wish there was a way to do this without both plots in the if/then statement
            for region in self.regions():
                plotECDFByValue(pd.DataFrame.query(self, 'Region == @region').Area_km2, ax=ax, alpha=0.4, label=region, **kwargs)

        else:
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

class BinnedLSD():
    '''This class represents lakes as area bins with summed areas.'''
    def __init__(self, lsd, btm, top, nbins=100, ci=True):
        '''
        Bins will have left end closed and right end open
        Parameters
        ----------
        lsd : LSD
            Lake-size distribution class to bin
        btm : float 
            Leftmost edge of bottom bin 
        top : float
            Rightmost edge of top bin 
        nbins : int
            Number of bins to use for np.geomspace.
        ci : Boolean
            Compute confidence interval by breaking down by region.
        '''
        lsd=LSD(lsd) # create copy
        self.nbins = nbins
        self.bin_edges = np.concatenate((np.geomspace(btm, top, nbins), [np.inf])).round(6)
        self.area_bins = pd.IntervalIndex.from_breaks(self.bin_edges, closed='left')
        lsd['labels'] = pd.cut(lsd.Area_km2, self.area_bins, right=False)
        self.isNormalized = False # init

        ## Bin
        if ci:
            ## First, group by region and area bin and take sum of each bin
            group_sums = lsd.groupby(['Region', 'labels']).Area_km2.sum(numeric_only=True)

            ## Normalize before binning
            # self.normalize()
            divisor = group_sums.loc[:, group_sums.index.get_level_values(1)[-1]] # sum of lake areas in largest bin for each region
            group_sums /= divisor
            self.isNormalized = True

            # Compute the confidence interval for each group along the second index
            self.binnedValues = group_sums.groupby(level=1).apply(confidence_interval)
            self.hasCI = True
        else:
            self.binnedValues = lsd.groupby('labels').Area_km2.sum(numeric_only=True)
            self.hasCI = False
        self.isBinned = True
        self.isExtrap = False
        self.labels = lsd.labels # copy to the new class
        pass
    
    def normalize(self):
        '''Divide bins by the largest bin.'''
        pass
        self.isNormalized = True

    def plot(self, show_rightmost=False):
        '''To roughly visualize bins.'''
        binned_values = self.binnedValues.copy() # create copy to modify
        diff=0
        ## Remove last bin, if desired
        if show_rightmost == False:
            binned_values.drop(index=binned_values.index.get_level_values(0)[-1], inplace=True)
            diff=1 # subtract from number of bin edges to get plot x axis

        # plt.bar(self.bin_edges[:-1], binned_values)
        if self.hasCI:
            ## Convert confidence interval vals to anomalies
            ci = binned_values.loc[:, ['lower', 'upper']].to_numpy().reshape((self.nbins-diff,2)).T
            # upper = binned_values.loc[:, ['upper']].to_numpy()
            # lower = binned_values.loc[:, ['lower']].to_numpy()
            mean = binned_values.loc[:, ['mean']].to_numpy()
            # yerr = np.array([mean-lower, upper-mean])
            yerr = np.abs(ci-mean)
            plt.bar(range(self.nbins-diff), binned_values.loc[:, 'mean'], yerr=yerr, color='orange') 
        else:  # Needs testing
            plt.bar(range(self.nbins-diff), binned_values)
        plt.yscale('log')
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
    gdf = pyogrio.read_dataframe('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp', read_geometry=True, use_arrow=True)
    lsd_from_gdf = LSD(gdf, area_var='Area', name='CIR', region_var='Region4')
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lsd_from_gdf = LSD(gdf, area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')
    print('\tPassed load from gdf.')
    
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
    
    ## Test binnedLSD
    binned = BinnedLSD(lsd_from_gdf.truncate(0,np.inf), 0.0001, 0.1)
    binned.plot()
    
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
    regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
    lsd_cir = LSD.from_shapefile('/mnt/g/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp', area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

    ## Loading PeRL LSD
    perl_exclude = ['arg0022009xxxx', 'fir0022009xxxx', 'hbl00119540701','hbl00119740617', 'hbl00120060706', 'ice0032009xxxx', 'rog00219740726', 'rog00220070707', 'tav00119630831', 'tav00119750810', 'tav00120030702', 'yak0012009xxxx', 'bar00120080730_qb_nplaea.shp']
    lsd_perl = LSD.from_paths('/mnt/f/PeRL/PeRL_waterbodymaps/waterbodies/*.shp', area_var='AREA', name='perl', _areaConversionFactor=1000000, exclude=perl_exclude)

    ## Loading from Mullen
    lsd_mullen = LSD.from_paths('/mnt/g/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/*_3Y_lakes-and-ponds.zip', _areaConversionFactor=1000000, name='Mullen', computeArea=True) # '/mnt/g/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/[A-Z][A-Z]_08*.zip'

    ## Combine PeRL and CIR and Mullen
    lsd = LSD.concat((lsd_cir, lsd_perl, lsd_mullen), broadcast_name=True, ignore_index=True)

    ## plot
    lsd.truncate(0.0001, 10).plot_lsd(all=True, plotLegend=False, reverse=False, groupby_name=True)

    # ## YF compare
    # LSD(lsd.query("Region=='YF_3Y_lakes-and-ponds' or Region=='Yukon Flats Basin'"), name='compare').plot_lsd(all=False, plotLegend=True, reverse=False, groupby_name=False)

    ## Load WBD
    lsd_wbd = LSD.from_shapefile('/mnt/g/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', area_var='Area', name='WBD', idx_var='OBJECTID')
    lsd_wbd.truncate(0.001, inplace=True)

    ## Plot WBD
    lsd_wbd.plot_lsd(reverse=False, all=False)

    ## Estimate area fraction
    lsd_wbd.area_fraction(0.1)
    lsd_wbd.area_fraction(0.01)
    lsd_wbd.area_fraction(0.001)

    lsd.area_fraction(0.1)
    lsd.area_fraction(0.01)
    lsd.area_fraction(0.001)  

    ## Legacy stuff
    ## Cycle through 14 regions and compute additional stats
    df_cir_rec = pd.DataFrame(columns = ['Region_ID', 'Region', 'Lake_area', 'stat', 'A_0.001', 'A_0.001_cor', 'A_0.01_cor', 'A_0.01', 'A_g100m', 'A_g1', 'PeRL_pnd_f']) # cir recomputed, for double checking A_0.001 from paper; A_0.001_cor excludes the largest lakes like in PeRL, but uses native CIR min size of 40 m2; computing A_0.01 analogous to PeRL pond fraction, PeRL_pnd for equivalent to perl analysis (min is 100 m2, max is 1 km2), A_g100m, for lakes > 100 m2, analogous to PeRL, F_g1 for area of lakes larger than PeRL maximum size, 'PeRL_all' to simulate perl size domain, and 'PeRL_pnd_f' which is perl pond divided perl all # 'PeRL_pnd', 'PeRL_all'
    for i, region in enumerate(range(2, 16)): # no region 1
        if region!=15: 
            gdf_tmp = gdf_cir_lsd.query('Region4 == @region')
        else:
            gdf_tmp = gdf_cir_lsd # compute over all
        df_cir_rec.loc[i, 'Region_ID'] = region
        df_cir_rec.loc[i, 'Region'] = regions[i]
        df_cir_rec.loc[i, 'Lake_area'] = gdf_tmp.Area.sum()
        df_cir_rec.loc[i, 'A_0.001'] = gdf_tmp.query('(Area < 0.001)').Area.sum() / df_cir_rec.loc[i, 'Lake_area'] * 100
        df_cir_rec.loc[i, 'A_0.001_cor'] = gdf_tmp.query('(Area < 0.001)').Area.sum() / gdf_tmp.query('(Area < 1)').Area.sum() * 100
        df_cir_rec.loc[i, 'A_0.01_cor'] = gdf_tmp.query('(Area < 0.01)').Area.sum() / gdf_tmp.query('(Area < 1)').Area.sum() * 100
        df_cir_rec.loc[i, 'A_0.01'] = gdf_tmp.query('(Area < 0.01)').Area.sum() / df_cir_rec.loc[i, 'Lake_area'] * 100
        df_cir_rec.loc[i, 'A_g1'] = gdf_tmp.query('(Area >= 1)').Area.sum() / df_cir_rec.loc[i, 'Lake_area'] * 100    
        df_cir_rec.loc[i, 'A_g100m'] = gdf_tmp.query('(Area >= 0.0001)').Area.sum() / df_cir_rec.loc[i, 'Lake_area'] * 100
        df_cir_rec.loc[i, 'PeRL_pnd_f'] = gdf_tmp.query('(Area >= 0.0001) and  (Area < 0.01)').Area.sum() / \
            gdf_tmp.query('(Area >= 0.0001) and  (Area < 1)').Area.sum() * 100
        df_cir_rec.loc[i, 'HL_pnd_f_4'] = gdf_tmp.query('(Area >= 0.0001) and  (Area < 0.1)').Area.sum() / \
            gdf_tmp.query('(Area >= 0.0001) and  (Area < 1)').Area.sum() * 100 # Pond fraction, as defined by HydroLakes lower limit and upper limit defined by CIR # 4 means that lower limit is 10^-4
        df_cir_rec.loc[i, 'HL_pnd_r_4'] = gdf_tmp.query('(Area >= 0.0001) and  (Area < 0.1)').Area.sum() / \
            gdf_tmp.query('(Area >= 0.1) and  (Area < 1)').Area.sum() * 100 # Pond fraction, as defined by HydroLakes lower limit and upper limit defined by CIR (fraction, not ratio, so it can be used as divident for extrapolation from mid lakes)
        df_cir_rec.loc[i, 'HL_pnd_r_3'] = gdf_tmp.query('(Area >= 0.001) and  (Area < 0.1)').Area.sum() / \
            gdf_tmp.query('(Area >= 0.1) and  (Area < 1)').Area.sum() * 100 # to compare to WBD
        df_cir_rec.loc[i, 'HL_pnd_r_2'] = gdf_tmp.query('(Area >= 0.01) and  (Area < 0.1)').Area.sum() / \
            gdf_tmp.query('(Area >= 0.1) and  (Area < 1)').Area.sum() * 100 # to compare to Sheng

    ## Re-index
    df_cir_rec.set_index('Region', inplace=True)

    ## Calculate means and uncertainties for pond percentage from BAWLD

    ## Lookup table with ratios for extrapolation
    ratio_cols = ['HL_pnd_r_4', 'HL_pnd_r_3', 'HL_pnd_r_2']
    ratio_table = pd.DataFrame({
        'All': df_cir_rec.loc['All', 'HL_pnd_r_4':'HL_pnd_r_2'],
        'Std': weightedStd(df_cir_rec.drop(index='All')[ratio_cols], df_cir_rec.drop(index='All')['Lake_area']), # [:, 'HL_pnd_r_4':'HL_pnd_r_2']
        'Std_unwt': df_cir_rec.drop(index='All')[ratio_cols].std(), # unweighted std dev (dont use)
        'Quant5': df_cir_rec.drop(index='All')[ratio_cols].quantile(0.05),
        'Quant95': df_cir_rec.drop(index='All')[ratio_cols].quantile(0.95)}) 

    ## Pre-add/subt to get upper and lower values based on std, for simplicity and thrift later on
    ratio_table['Lower'] = ratio_table.All - ratio_table.Std
    ratio_table['Upper'] = ratio_table.All + ratio_table.Std

    ## Load grid and lake gdfs
    gdf = pyogrio.read_dataframe(gdf_bawld_pth, read_geometry=True, use_arrow=True) # grid cells
    df = pyogrio.read_dataframe(gdf_HL_jn_pth, columns=['BAWLDCell_', 'Shp_Area', 'BAWLDLong', 'BAWLDLat', '0-5', '5-50', '50-95','95-100'], read_geometry=False, use_arrow=True) # lakes (load as df to save mem)

    # Loop over grid cells (TODO: rewrite as a map or map_async)
    for i in tqdm(range(len(gdf))): # 10 # len(gdf)
        ## Select only lakes in cell (based on pour point)
        cell_id = gdf.loc[i, 'Cell_ID'] # called BAWLD_Cell in df
        df_tmp = df.query(f'BAWLDCell_==@cell_id')

        ## compute area stats:
        ## Occurence stats
        gdf.loc[i, 'HL_area'] = df_tmp.Shp_Area.sum() # all HL-observable lakes
        gdf.loc[i, '0_5_area'] = np.nansum(df_tmp['0-5']/100 * df_tmp.Shp_Area) # Total area in an occurence bin
        gdf.loc[i, '5_50_area'] = np.nansum(df_tmp['5-50']/100 * df_tmp.Shp_Area) # Total area in an occurence bin 
        gdf.loc[i, '50_95_area'] = np.nansum(df_tmp['50-95']/100 * df_tmp.Shp_Area) # Total area in an occurence bin
        gdf.loc[i, '95_100_area'] = np.nansum(df_tmp['95-100']/100 * df_tmp.Shp_Area) # Total area in an occurence bin
        
        ## Occ stats normalized by total lake area
        if gdf.loc[i, 'HL_area'] > 0:
            gdf.loc[i, '0_5_per'] = gdf.loc[i, '0_5_area'] / gdf.loc[i, 'HL_area'] * 100
            gdf.loc[i, '5_50_per'] = gdf.loc[i, '5_50_area'] / gdf.loc[i, 'HL_area'] * 100
            gdf.loc[i, '50_95_per'] = gdf.loc[i, '50_95_area'] / gdf.loc[i, 'HL_area'] * 100
            gdf.loc[i, '95_100_per'] = gdf.loc[i, '95_100_area'] / gdf.loc[i, 'HL_area'] * 100

        ## extrapolations
        gdf.loc[i, 'Ppnd_area'] = df_tmp.query('(Shp_Area >= 0.0001) and  (Shp_Area < 0.01)').Shp_Area.sum() # perl ponds (will always be 0 for HL)
        gdf.loc[i, 'Plk_area'] = df_tmp.query('(Shp_Area >= 0.01) and  (Shp_Area < 1)').Shp_Area.sum() # perl lakes (not ponds)
        gdf.loc[i, 'Mid_lk_area'] = df_tmp.query('(Shp_Area >= 0.1) and  (Shp_Area < 1)').Shp_Area.sum() # Mid lakes (if I used 0.3 cutoff, I could include all CIR sites) to use for extrapolation
        gdf.loc[i, 'Lg_lk_area'] = df_tmp.query('(Shp_Area >= 1)').Shp_Area.sum() # Large lakes (add to extrapolation)
        gdf.loc[i, 'Extrap4'] = (1 + ratio_table.loc['HL_pnd_r_4', 'All'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area'] # Hpond extrap 10 e-4
        gdf.loc[i, 'Extrap4_l'] = (1 + ratio_table.loc['HL_pnd_r_4', 'Lower'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap4_u'] = (1 + ratio_table.loc['HL_pnd_r_4', 'Upper'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap3'] = (1 + ratio_table.loc['HL_pnd_r_3', 'All'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap3_l'] = (1 + ratio_table.loc['HL_pnd_r_3', 'Lower'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap3_u'] = (1 + ratio_table.loc['HL_pnd_r_3', 'Upper'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap2'] = (1 + ratio_table.loc['HL_pnd_r_2', 'All'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap2_l'] = (1 + ratio_table.loc['HL_pnd_r_2', 'Lower'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Extrap2_u'] = (1 + ratio_table.loc['HL_pnd_r_2', 'Upper'] / 100) * gdf.loc[i, 'Mid_lk_area'] + gdf.loc[i, 'Lg_lk_area']
        gdf.loc[i, 'Meg_lk_area'] = df_tmp.query('(Shp_Area >= 5000)').Shp_Area.sum() # Mega lakes (often subtracted from upscaling)

    ## INfo on result
    df[['50-95', 'Shp_Area']] # km2 for HL
    gdf[['5_50_area', '50_95_area', '50_95_per', 'HL_area']]
    gdf.loc[20:25,['5_50_area', '50_95_area', '50_95_per', 'HL_area', 'Cell_ID']]

    ## Write out
    gdf.to_csv(area_extrap_pth)
    print(f'Wrote file: {area_extrap_pth}')

    ## Write out as shapefile
    # pyogrio.write_dataframe(gdf, area_extrap_pth.replace('.csv','.shp')) # Why so slow...150 min??
    area_extrap_pth_shp = area_extrap_pth.replace('.csv','.shp')
    gdf.to_file(area_extrap_pth_shp)
    print(f'Wrote file: {area_extrap_pth_shp}')

    ## Check
    gdf.iloc[250:260,:]
    gdf.Hpnd_extrap4.sum()
    gdf.HL_area.sum()

    ## Plot total area as function of extrapolated min. size

    sums = gdf.sum()
    plt.errorbar([0.1, 0.01, 0.001, 0.0001], [sums.HL_area, sums.Hpnd_extrap2, sums.Hpnd_extrap3, sums.Hpnd_extrap4],
        yerr=[0, sums.Hpnd_extrap2-sums.Hpnd_extrap2_l, sums.Hpnd_extrap3-sums.Hpnd_extrap3_l, sums.Hpnd_extrap4-sums.Hpnd_extrap4_l],
        capsize=4, fmt='.-b', label='Predicted')
    plt.plot([0.1, 0.01, 0.001], [746137.7599999998, 805675.0937146356, 819091.9657548245], '.-r', label='Lake inventories') # Paste summed values from lake databases here
    plt.xscale('log')
    plt.xlabel('Minimum lake size ($km^2$)')
    plt.ylabel('Total lake area ($km^2$)')
    plt.title(f'region: {roi_region}')
    plt.legend()

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
'''
