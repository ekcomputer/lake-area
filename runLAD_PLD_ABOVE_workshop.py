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

## I/O
# tables output dir
output_base_dir = '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop'
tb_dir = os.path.join(output_base_dir, 'area_tables')
# dir for output data, used for data archive
output_dir = os.path.join(output_base_dir, 'output')
pic_dir = os.path.join(output_base_dir, 'pic')

v = 32  # Version number for file naming
ds = 'PLD'  # dataset
extreme_regions_lad = [
    'Tuktoyaktuk Peninsula', 'sur00120130802_tsx_nplaea']

# ## BAWLD domain
# dataset = 'PLD'
# roi_region = '40N'
# gdf_bawld_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
# # above, but with all ocurrence values, not binned
# # main data source
# # from utils.py - has water occurrence values
# df_HL_jn_full_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full.csv.gz'
# # hl_area_var = 'Shp_Area'
# inventory_join_clim_pth = '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/joined_ERA5/HL_ERA5_stl1_v3.csv.gz'
# bawld_join_clim_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1___Shapefile_jn_clim.csv'
# # HL shapefile with ID of nearest BAWLD cell (still uses V3)
# hl_nearest_bawld_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned_jnBAWLD.shp'
# bawld_hl_output = os.path.join(output_dir, f'BAWLD_V1_LEV_v{v}.shp')

## Global domain
dataset = 'PLD'
roi_region = 'glob'
# above, but with all ocurrence values, not binned
# main data source
# from utils.py - has water occurrence values
# df_HL_jn_full_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full.csv.gz'
area_var = 'ref_area'
temps_var = 'ERA5_stl1'
inventory_join_clim_pth = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta_temps.gdb'
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
    lad_trunc = LAD.from_shapefile(inventory_join_clim_pth, area_var=area_var,
                                   idx_var=None, name=dataset, region_var=None, other_vars=[temps_var, 'lat', 'lon', 'lake_id', 'lake_num'])

    ## Plot LAD
    ax = lad_trunc.plot_lad(plotLegend=False)
    [ax.get_figure().savefig(
        os.path.join(pic_dir, f'areas_v{v}' + ext), transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    ####################################
    ## Compute CH4 emissions
    ####################################

    ## Flux prediction from observed and extrap lakes
    model = loadR21_CH4(temperature_metric=temps_var)
    # model = loadBAWLD_CH4()

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
    ## Map Analysis
    ####################################

    ## Rescale to km2
    for col in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']:
        lad[col + '_km2'] = lad[col] * \
            lad['Area_km2']  # add absolute area units
    # lad.to_csv('/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v5/HL_BAWLD_LEV.csv')

    ## Rescale double-counting to km2 for data archival purposes
    lad['d_counting_km2'] = lad.d_counting_frac * \
        lad['Area_km2']

    ## Prep weighted avgs
    lad.predictFlux(model, includeExtrap=False)
    lad['Temp_K_wght_sum'] = lad.Temp_K * lad.Area_km2

    ## Groupby bawld cell and compute sum of LEV and weighted avg of LEV
    df_bawld_sum_lev = lad.groupby('BAWLD_Cell', observed=False).sum(
        numeric_only=True)  # Could add Occ

    ## Lake count
    df_bawld_sum_lev['lake_count'] = lad[['Area_km2', 'BAWLD_Cell']].groupby(
        'BAWLD_Cell', observed=False).count().astype('int')

    ## Rescale back to LEV fraction (of lake) as well (equiv to lake area-weighted mean of LEV fraction within grid cell)
    for col in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']:
        df_bawld_sum_lev[(col + '_km2').replace('_km2', '_frac')] = df_bawld_sum_lev[col +
                                                                                     '_km2'] / df_bawld_sum_lev['Area_km2']  # add absolute area units
        # remove summed means, which are meaningless
        df_bawld_sum_lev.drop(columns=col, inplace=True)

    ## add averages of T and est_mg_m2_day
    df_bawld_sum_lev['Temp_K'] = df_bawld_sum_lev['Temp_K_wght_sum'] / \
        df_bawld_sum_lev.Area_km2
    df_bawld_sum_lev['est_mg_m2_day'] = df_bawld_sum_lev['est_g_day'] / \
        1e3 / df_bawld_sum_lev.Area_km2

    ## remove meaningless sums
    df_bawld_sum_lev.drop(
        columns=['idx_HL', 'Temp_K_wght_sum', 'd_counting_frac', '0-5', '5-50', '50-95', '95-100'], inplace=True)  # 'Hylak_id',

    ## Join to BAWLD in order to query cell areas
    gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio')
    gdf_bawld_sum_lev = df_bawld_sum_lev.merge(
        gdf_bawld, how='outer', right_on='Cell_ID', left_index=True)  # [['Cell_ID', 'Shp_Area']]

    ## Rescale to LEV fraction and double counting fraction (of grid cell)
    for col in ['LEV_MEAN', 'LEV_MIN', 'LEV_MAX']:
        gdf_bawld_sum_lev[(col + '_km2').replace('_km2', '_grid_frac')] = gdf_bawld_sum_lev[col + '_km2'] / (
            gdf_bawld_sum_lev['Shp_Area'] / 1e6)  # add cell LEV fraction (note BAWLD units are m2)
    gdf_bawld_sum_lev['d_counting_grid_frac'] = gdf_bawld_sum_lev['d_counting_km2'] / \
        (gdf_bawld_sum_lev['Shp_Area'] / 1e6)

    ## Mask out high Glacier or barren grid cells with no lakes
    gdf_bawld_sum_lev.loc[(gdf_bawld_sum_lev.GLA + gdf_bawld_sum_lev.ROC) > 75,
                          ['LEV_MEAN_km2', 'LEV_MIN_km2', 'LEV_MAX_km2', 'LEV_MEAN_frac', 'LEV_MIN_frac', 'LEV_MAX_frac', 'LEV_MEAN_grid_frac', 'LEV_MIN_grid_frac', 'LEV_MAX_grid_frac']] = 0

    ## and write out full geodataframe as shapefile with truncated field names
    gdf_bawld_sum_lev['Shp_Area'] = gdf_bawld_sum_lev['Shp_Area'].astype(
        'int')  # convert area to int
    gpd.GeoDataFrame(gdf_bawld_sum_lev).to_file(
        bawld_hl_output, engine='pyogrio')

    ## Stats from BAWLD LEV
    s = gdf_bawld_sum_lev.drop(columns=['geometry']).sum()
    print(
        f"BAWLD domain: {s.LEV_MEAN_km2/1e6:0.3} [{s.LEV_MIN_km2/1e6:0.3}-{s.LEV_MAX_km2/1e6:0.3}] Mkm2 lake vegetation (excluding non-inventoried lakes).")
    print(
        f"BAWLD domain is {s.LEV_MEAN_km2/(s.Shp_Area/1e6):0.4%} [{s.LEV_MIN_km2/(s.Shp_Area/1e6):0.4%}-{s.LEV_MAX_km2/(s.Shp_Area/1e6):0.4%}] lake vegetation (excluding non-inventoried lakes).")
    print(
        f"BAWLD domain: {np.dot(gdf_bawld_sum_lev.WET/100,gdf_bawld_sum_lev.Shp_Area/1e6)/1e6:0.3} [{np.dot(gdf_bawld_sum_lev.WET_L/100,gdf_bawld_sum_lev.Shp_Area/1e6)/1e6:0.3}-{np.dot(gdf_bawld_sum_lev.WET_H/100,gdf_bawld_sum_lev.Shp_Area/1e6)/1e6:0.3}] Mkm2  wetlands.")
    print(
        f"BAWLD domain is {np.average(gdf_bawld_sum_lev.WET, weights=gdf_bawld_sum_lev.Shp_Area):0.4} [{np.average(gdf_bawld_sum_lev.WET_L, weights=gdf_bawld_sum_lev.Shp_Area):0.4}-{np.average(gdf_bawld_sum_lev.WET_H, weights=gdf_bawld_sum_lev.Shp_Area):0.4}%] wetlands.")

    ####################################
    ## Write out datasets for archive
    ####################################

    ## Add temperatures to HL_lev dataset (don't use truncated, because data users can easily truncate by lake area)
    # keys = [temps_var]
    # values = ['Temp_' + key for key in keys]
    # rename_dict = {k: v for k, v in zip(keys, values)}
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

    ## Write out
    lad_save.to_csv(os.path.join(
        output_dir, f'{ds}_emissions_v{v}.csv'))

    # TODO: merge to og gdf and write out spatially
    '''
    ogr2ogr -sql "select SWOT_PLD_v103_beta_temps.*, PLD_emissions_v32.* from SWOT_PLD_v103_beta_temps left join '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop/output/PLD_emissions_v32.csv'.PLD_emissions_v32  on SWOT_PLD_v103_beta_temps.lake_id = PLD_emissions_v32.lake_id" '/Volumes/metis/global_lake_ch4/ABoVE_flux_workshop/output/PLD_emissions_v32_jn.gdb' '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/SWOT_PLD_v103_beta_temps.gdb'
    '''

    ## Regrid v1
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

    grouped = lad_trunc.groupby(['lat_idx', 'lon_idx'])
    summed_est_g_day = grouped['est_g_day'].sum()
    df = pd.DataFrame(
        {'lat_idx': mg_lat_idx.flatten(), 'lon_idx': mg_lon_idx.flatten(), 'time': 2022}).set_index(['lon_idx', 'lat_idx', 'time'])
    # , left_on=['lon', 'lat'], right_on=['lon', 'lat'])
    df_merge = pd.merge(df, summed_est_g_day, left_index=True,
                        right_index=True, how='left')

    # continue to xarray
    da = df_merge.to_xarray()
    da = da.assign_coords({'lon': ('lon_idx', grid_lon),
                          'lat': ('lat_idx', grid_lat)})

    # da.set_index({0:'lon', 1:'lat'})
    # da.reset_coords(['lon_idx', 'lat_idx'], drop=True)
    # da.assign_coords('lon', 'lat'])
    # da.set_coords(['lon', 'lat'])
    # da = da.assign_coords({'time':[2022]})
    # da = da.transpose('lon_idx','lat_idx','time')
    da = da.assign_attrs({'Provider': 'Ethan Kyzivat',
                          'Citation': 'Kyzivat, E. D., & Smith, L. C. (2023). A Closer Look at the Effects of Lake Area, Aquatic Vegetation, and Double - Counted Wetlands on Pan - Arctic Lake Methane Emissions Estimates. Geophysical Research Letters, 50(24), e2023GL104825. https://doi.org/10.1029/2023GL104825'})
    da.to_netcdf(os.path.join(
        output_dir, f'{ds}_emissions_v{v}.nc'))

    pass
