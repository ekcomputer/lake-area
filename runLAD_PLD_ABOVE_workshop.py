'''
Applies CH4 estimate without small lake extrapolation. Customize to choose which reference emissions dataset and whether to use LAV mask. Finally, regrids to 30N 0.5 degree grid as netcdf.

Run after runBatchZonalHistERA5_ABOVE_workshop.py
'''
## Imports
from warnings import warn
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import pyogrio
import argparse
from scipy.stats import pearsonr
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import xarray as xr
from LAD.LAD import *
from LAD.IO import loadR21_CH4, loadBAWLD_CH4, load_HR_ABZ
from LAD.util import grid_area

## I/O
# tables output dir
output_base_dir = '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop'
tb_dir = os.path.join(output_base_dir, 'area_tables')
# dir for output data, used for data archive
output_dir = os.path.join(output_base_dir, 'output')
pic_dir = os.path.join(output_base_dir, 'pic')
extreme_regions_lad = [
    'Tuktoyaktuk Peninsula', 'sur00120130802_tsx_nplaea']
areas_netcdf = os.path.join(output_dir, 'gridarea.nc')

## for output file name
v = 33  # Version number for file naming
ch4_ref = 'R21'
# ch4_ref = 'BAWLD'
ds = 'PLD'
roi_region = 'glob'
year = 2017

## params
area_var = 'ref_area'
temps_var = 'ERA5_stl1'
trunc = 0.1
# trunc = 0.01
inventory_join_clim_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta_temps2017.gdb'
# inventory_join_clim_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta_temps2017.gdb'
# HL shapefile with ID of nearest BAWLD cell (still uses V3)

if __name__ == '__main__':
    ## Loading from CIR gdf
    print('Load HR...')
    # lad_ref = load_HR_ABZ()

    ####################################
    ## LEV Analysis
    ####################################

    # ## Load csv and shapefiles
    # ref_names = ['CSB', 'CSD', 'PAD', 'YF']
    # extreme_regions_lev_for_extrap = ['CSD', 'PAD']
    # lad_lev_cat, ref_dfs = loadUAVSAR(ref_names)

    # ## Create binned ref LEV distribution from UAVSAR
    # binned_lev = BinnedLAD(lad_lev_cat, 0.0001, 0.5, compute_ci_lev=True,
    #                        extreme_regions_lev=extreme_regions_lev_for_extrap)  # 0.000125 is native

    # ## LEV estimate: Load UAVSAR/GSW overlay stats
    # print('Load HL with joined occurrence...')
    # # lad_hl_oc = pyogrio.read_dataframe('/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_full.shp', read_geometry=False, use_arrow=True) # load shapefile with full histogram of zonal stats occurrence values # outdated version
    # # read smaller csv gzip version of data.
    # lad_hl_oc = pd.read_csv(
    #     df_HL_jn_full_pth, compression='gzip', low_memory=False)
    # lev = computeLAV(lad_hl_oc, ref_dfs, ref_names, extreme_regions_lev=extreme_regions_lev_for_extrap,
    #                  use_low_oc=use_low_oc)  # use same extreme regions for est as for extrap

    # ## Set high arctic lakes LEV to 0 (no GSW present above 78 degN)
    # lev.loc[lev.Pour_lat >= 78, ['LEV_MEAN',
    #                              'LEV_MIN', 'LEV_MAX']] = 0  # LEV_MEAN

    # ## Turn into a LAD
    # # main dataset for analysis
    # lad = LAD(lev, area_var='Lake_area', idx_var='Hylak_id', name='HL')

    # # ## Plot LEV CDF by lake area (no extrap) and report mean LEV fraction
    # # lad.plot_lev_cdf_by_lake_area()
    # # lad.plot_lev_cdf_by_lake_area(normalized=False)

    ####################################
    ## Climate Analysis: join in temperature
    ####################################
    print('Loading lake inventory and climate data...')

    inventory_join_clim_pth = Path(inventory_join_clim_pth)
    if inventory_join_clim_pth.suffix == 'csv.gz':
        df_clim = pd.read_csv(inventory_join_clim_pth, compression='gzip')
    elif np.isin(inventory_join_clim_pth.suffix, ['.shp', '.gdb']):
        df_clim = gpd.read_file(inventory_join_clim_pth,
                                engine='pyogrio', read_geometry=True)

    # ## Add binned occurrence values
    # for var in ['0-5', '5-50', '50-95', '95-100']:
    #     lad[var] = lad_m[var]

    # ## Compute double-counting
    # lad['d_counting_frac'] = (
    #     lad['0-5'] + lad['5-50']) / 100

    # ## Compute cell-area-weighted average of climate as FYI
    # # print(f'Mean JJA temperature across {roi_region} domain: {np.average(df_clim.jja, weights=df_clim.Shp_Area)}')
    # # months = ['ann','djf','mam','jja','son']
    # # print(pd.DataFrame(np.average(df_clim[months], weights=df_clim.Shp_Area, axis=0), index=months))

    ####################################
    ## Load and extrapolate inventory
    ####################################

    ## Load dataset (named 'trunc' for compatability purposes)
    lad = LAD.from_shapefile(inventory_join_clim_pth, area_var=area_var,
                             idx_var=None,
                             name=ds,
                             region_var=None, other_vars=[temps_var, 'lat', 'lon', 'lake_id', 'lake_num'])
    lad_trunc = lad.truncate(trunc)

    ## Plot LAD
    ax = lad_trunc.plot_lad(plotLegend=False)
    [ax.get_figure().savefig(
        os.path.join(pic_dir, f'areas_v{v}' + ext), transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    ####################################
    ## Compute CH4 emissions
    ####################################

    ## Flux prediction from observed and extrap lakes
    if ch4_ref == 'R21':
        model = loadR21_CH4(temperature_metric=temps_var)
    elif ch4_ref == 'BAWLD':
        model = loadBAWLD_CH4()

    ## Harmonize ERA5.stl1 with reported water temp by adding 2 K (if using Rosentreter)
    lad_trunc['Temp_K'] = lad_trunc[temps_var] + 2

    del lad_trunc[temps_var]
    lad_trunc.predictFlux(model, includeExtrap=False)

    print(
        f"Estimated annual flux: {lad_trunc._total_flux_Tg_yr:.3} Tg/yr")

    ## Plot  fluxes
    ax = lad_trunc.plot_flux(plotLegend=False, reverse=False, normalized=False)
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim([ymin, ymax / lad_trunc._total_flux_Tg_yr])
    ax2.set_ylabel('Cumulative emissions fraction')
    plt.tight_layout()
    [ax.get_figure().savefig(
        os.path.join(pic_dir, f'fluxes_v{v}' + ext), transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    ####################################
    ## Write out datasets for archive
    ####################################

    ## Add temperatures to HL_lev dataset (don't use truncated, because data users can easily truncate by lake area)
    keys = [temps_var]
    values = ['Temp_' + key for key in keys]
    rename_dict = {k: v for k, v in zip(keys, values)}
    # keys_oc = ['0-5', '5-50', '50-95', '95-100']
    # values_oc = ['Oc_' + key.replace('-', '_') for key in keys_oc]
    # oc_dict = {k: v for k, v in zip(keys_oc, values_oc)}
    # rename_dict.update(oc_dict)
    # keys = ['LEV_MAX', 'LEV_MEAN', 'LEV_MIN']
    # values = [key.replace('LEV', 'LAV') for key in keys]
    # lav_dict = {k: v for k, v in zip(keys, values)}  # lake aquatic veg
    # rename_dict.update(lav_dict)
    # rename_dict.update({'idx_HL': 'Hylak_id'})

    # Format data types
    lad_save = lad_trunc.drop(columns=[col for col in ['Region', 'Temp_K_wght_sum', 'LEV_MAX_km2',
                                                       'LEV_MEAN_km2', 'LEV_MIN_km2', 'd_counting_km2'] if col in lad_trunc.columns]).rename(columns=rename_dict)
    # Get a list of columns with float data type
    float_columns = lad_save.select_dtypes(
        include=['float']).columns.tolist()
    lad_save[float_columns] = lad_save[float_columns].round(
        4)  # Apply rounding to float columns to reduce output file size

    ## output name
    output_name = f'{ds}_trunc{trunc}_ch4_emissions_{ch4_ref}_{year}_v{v}'

    ## Write out
    # print('Writing csv...')
    # lad_save.to_csv(os.path.join(
    #     output_dir, output_name + '.csv'))

    # TODO: merge to og gdf and write out spatially
    '''
    ogr2ogr -sql "select SWOT_PLD_v103_beta_temps.*, PLD_emissions_v32.* from SWOT_PLD_v103_beta_temps left join '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop/output/PLD_emissions_v32.csv'.PLD_emissions_v32  on SWOT_PLD_v103_beta_temps.lake_id = PLD_emissions_v32.lake_id" '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop/output/PLD_emissions_v32_jn.gdb' '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta_temps.gdb'
    '''

    ## Regrid v1
    print('Regrid...')
    grid_lon = np.arange(-179.75, 180, 0.5)
    grid_lat = np.arange(30.25, 90, 0.5)
    mg_lon, mg_lat = np.meshgrid(grid_lon, grid_lat)
    mg_lon_idx, mg_lat_idx = np.meshgrid(
        range(len(grid_lon)), range(len(grid_lat)))

    # ch4_profile_n_rg = interpolate.griddata((lad_trunc.lat, lad_trunc.lon),
    #                                         lad_trunc.est_g_day,
    #                                            (mg_lat.flatten(),
    #                                             mg_lon.flatten()),
    #                                             method='nearest'
    #                                            )
    # ch4_profile_n_rg.reshape(720, 124)

    ## Regrid v2
    lon_edges = np.arange(-180, 180.5, 0.5)
    lat_edges = np.arange(28, 90.5, 0.5)
    lad_trunc['lat_idx'] = np.digitize(lad_trunc['lat'], lat_edges)
    lad_trunc['lon_idx'] = np.digitize(lad_trunc['lon'], lon_edges)
    lad_trunc = lad_trunc[lad_trunc.lat_idx > 0]
    lad_trunc = lad_trunc[lad_trunc.lon_idx > 0]
    lad_trunc['bin_latitude'] = grid_lat[np.digitize(
        lad_trunc['lat'], lat_edges) - 1]
    lad_trunc['bin_longitude'] = grid_lon[np.digitize(
        lad_trunc['lon'], lon_edges) - 1]

    grouped = lad_trunc.groupby(['bin_latitude', 'bin_longitude'])
    summed_est_g_day = grouped['est_g_day'].sum()
    # / grouped['Area_km2'].sum()
    summed_mgC_day = (summed_est_g_day * 1000 * 12.011 / 16.04).astype('float')
    summed_mgC_day.rename('CH4_mgC_day')
    # mean_mgC_m2_day = summed_mgC_day / grid_area
    # df = pd.DataFrame(
    #     {'lat_idx': mg_lat_idx.flatten(), 'lon_idx': mg_lon_idx.flatten(), 'time': 2022}).set_index(['lon_idx', 'lat_idx', 'time'])
    # , left_on=['lon', 'lat'], right_on=['lon', 'lat'])
    df = pd.DataFrame(
        {'bin_latitude': grid_lat[mg_lat_idx].flatten(), 'bin_longitude': grid_lon[mg_lon_idx].flatten(), 'time': 2022}).set_index(['bin_longitude', 'bin_latitude', 'time'])

    df_merge = pd.merge(df, summed_mgC_day, left_index=True,
                        right_index=True, how='left')
    df_merge.index.names = ['longitude', 'latitude', 'time']
    df_merge['area'] = grid_area(
        df_merge.index.get_level_values('latitude').values, 0.5)
    # continue to xarray
    da = df_merge.to_xarray()
    da['fch4'] = da.est_g_day / da.area
    # da = da.assign_coords({'lon': ('lon_idx', grid_lon),
    #                       'lat': ('lat_idx', grid_lat)})
    # gridarea = xr.load_dataset(areas_netcdf)
    # da['mean_mgC_m2_day'] = da.summed_mgC_day.data / gridarea.

    # da.set_index({0:'lon', 1:'lat'})
    # da.reset_coords(['lon_idx', 'lat_idx'], drop=True)
    # da.assign_coords('lon', 'lat'])
    # da.set_coords(['lon', 'lat'])
    # da = da.assign_coords({'time':[2022]})
    # da = da.transpose('lon_idx','lat_idx','time')

    da = da.assign_attrs({'Provider': 'Ethan Kyzivat',
                          'Citation': 'Kyzivat, E. D., & Smith, L. C. (2023). A Closer Look at the Effects of Lake Area, Aquatic Vegetation, and Double - Counted Wetlands on Pan - Arctic Lake Methane Emissions Estimates. Geophysical Research Letters, 50(24), e2023GL104825. https://doi.org/10.1029/2023GL104825',
                          'Data citation': 'Ethan D Kyzivat, & Laurence C Smith. (2023). Parameters and code for estimating methane emissions from Arctic-boreal lakes, 2022. Arctic Data Center. https://doi.org/10.18739/A27M04222.'})
    da.fch4.attrs.update({'Description': 'CH4 flux per unit area due to lake emissions, averaged over a grid cell',
                          'Units': 'mgC m-2 d-1'})
    da.area.attrs.update({'Description': 'Grid cell area', 'Units': 'm2'})
    nc_pth_out = os.path.join(
        output_dir, output_name + '.nc')
    da.to_netcdf(nc_pth_out)
    print(f'Wrote file: {nc_pth_out}')

    pass
