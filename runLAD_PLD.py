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
from sklearn.metrics import mean_squared_error
from LAD.LAD import *
from LAD.IO import loadR21_CH4, loadBAWLD_CH4, load_HR_ABZ

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
# tables output dir
tb_dir = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/edk_out/CH4/area_tables'
hydrolakes_pth = '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
# dir for output data, used for data archive
output_dir = '/Volumes/metis/Datasets/SWOT_PLD/SWOT_PLD_v103_beta/edk_out/CH4/output'
v = 32  # Version number for file naming
ds = 'PLD'  # dataset
extreme_regions_lad = [
    'Tuktoyaktuk Peninsula', 'sur00120130802_tsx_nplaea']
# Truncation limits for ref LAD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
tmin, tmax = (0.0001, 0.5)
# Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
emax = 0.05

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
    lad_ref = load_HR_ABZ()

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

    ## Load hydrolakes. if not using LEV
    # print('Load HL...')
    lad = LAD.from_shapefile(inventory_join_clim_pth, area_var=area_var,
                             idx_var=None, name=dataset, region_var=None, other_vars=[temps_var, 'lat', 'lon'])

    ## Extrapolate
    binned_ref = BinnedLAD(lad_ref.truncate(tmin, tmax), tmin, emax, compute_ci_lad=True,
                           extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
    # Beware chaining unless I return a new variable. # Try 0.1
    lad_trunc = lad.truncate(emax, np.inf)
    lad_trunc.extrapolate(binned_ref)

    ## Select only Boreal-Arctic lakes (BAWLD domain)
    # lad_trunc.query("lat >= 55", inplace=True)

    meas = lad.sumAreas(includeExtrap=False)
    extrap = lad_trunc.sumAreas()

    limit = 0.01
    print(f'Total measured lake area in {roi_region} domain: {meas:,.0f} km2')
    print(
        f'Total extrapolated lake area in {roi_region} domain: {extrap:,.0f} km2')
    print(f'{1-(meas / extrap):.1%} of lake area is < observation limit of {emax} km2.')
    # frac = lad_trunc.extrapolated_area_fraction(lad, 0.0001, limit, emax=emax, tmax=tmax)
    # print(f'{frac:.1%} of lake area is < {limit} km2.')
    # print(f'{lad_trunc.extrapolated_area_fraction(lad_ref, 0.0001, 0.001, emax=emax, tmax=tmax):.1%} of lake area is < 0.001 km2.')
    print(f'{lad_trunc.extrapolated_area_fraction(lad_ref, 0.0001, 0.01, emax=emax, tmax=tmax):.1%} of lake area is < 0.01 km2.')

    ## Plot HL extrapolation
    # ax = lad_hl.plot_lad(all=False, reverse=False, normalized=False)
    ax = lad_trunc.plot_extrap_lad(
        label='HL-extrapolated', error_bars=False, normalized=False)
    ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')

    ####################################
    ## Compute CH4 emissions
    ####################################

    ## Flux prediction from observed and extrap lakes
    model = loadR21_CH4(temperature_metric=temps_var)
    # model = loadBAWLD_CH4()

    # correct variable name for flux prediction with no harmonization
    # lad_trunc['Temp_K'] = lad_trunc[temps_var]

    ## Harmonize ERA5.stl1 with reported water temp by adding 2 K (if using Rosentreter)
    lad_trunc['Temp_K'] = lad_trunc[temps_var] + 2

    del lad_trunc[temps_var]
    lad_trunc.predictFlux(model, includeExtrap=True)
    print(
        f"Estimated annual flux: {lad_trunc._total_flux_Tg_yr['mean']:.3} Tg/yr")

    # ## Plot combined extrap LAD/LEV
    # fig, ax = plt.subplots(2, 1, sharex=True)
    # lad_trunc.plot_extrap_lad(
    #     ax=ax[0], label='Lake area', error_bars=True, normalized=False, color='blue', plotLegend=False)
    # # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    # ax2 = ax[0].twinx()
    # lad_trunc.plot_extrap_lev(
    #     ax=ax[0], error_bars=True, color='green', label='Lake vegetation area', plotLegend=False, )
    # ymin, ymax = ax[0].get_ylim()
    # # ax2.set_ylim([ymin, ymax / lad_trunc.sumAreas()])
    # ax2.set_ylim([0, 1.1])
    # ax[0].set_ylabel('Cumulative area (million $km^2$)')
    # ax[0].set_xlabel('')
    # ax2.set_ylabel('Cumulative area fraction')
    # # plt.tight_layout()

    # ## Plot extrapolated fluxes
    # lad_trunc.plot_extrap_flux(
    #     ax=ax[1], reverse=False, normalized=False, error_bars=True, plotLegend=False, label='Emissions')
    # ax2 = ax[1].twinx()
    # ymin, ymax = ax[1].get_ylim()
    # ax2.set_ylim([ymin, ymax / lad_trunc._total_flux_Tg_yr['mean']])
    # ax2.set_ylabel('Cumulative emissions fraction')
    # plt.tight_layout()
    # [ax2.get_figure().savefig(
    #     f'/Volumes/thebe/pic/BAWLD_areas_v{v}' + ext, transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    # ## Plot combined extrap LAD/Flux
    # norm = True # False
    # ax = lad_hl_trunc.plot_extrap_lad(label='HL-extrapolated', error_bars=True, normalized=False, color='blue')
    # # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    # ax2=ax.twinx()
    # lad_hl_trunc.plot_extrap_flux(ax=ax2, reverse=False, normalized=norm, error_bars=True)
    # plt.tight_layout()

    # ## Plot inset with just LEV, with normalized second axis
    # sns.set_theme('poster', font='Ariel')
    # sns.set_style('ticks')
    # ax = lad_trunc.plot_extrap_lev(
    #     error_bars=True, color='green', plotLegend=False)
    # ax2 = ax.twinx()
    # ymin, ymax = ax.get_ylim()
    # ax2.set_ylim([ymin, ymax / lad_trunc.sumLev()['mean']])
    # ax.set_ylabel('')  # 'Cumulative aquatic vegetation area (million $km^2$)')
    # ax2.set_ylabel('')  # 'Cumulative aquatic vegetation area fraction')
    # plt.tight_layout()
    # [ax.get_figure().savefig(
    #     f'/Volumes/thebe/pic/BAWLD_areas_inset_v{v}', transparent=True, dpi=300) for ext in ['.png', '.pdf']]
    # sns.set_theme('notebook', font='Ariel')
    # sns.set_style('ticks')

    # ## Retrieve data from plot
    # ax.get_lines()[0].get_ydata() # gives right part of LAD plot # [1] is left part
    # ax2.get_lines()[0].get_ydata() # gives right part of LEV plot
    # X_lev = np.concatenate((ax.get_lines()[1].get_xdata(), ax.get_lines()[0].get_xdata()))
    # S_lev = np.concatenate((ax.get_lines()[1].get_ydata(), ax.get_lines()[0].get_ydata()))

    ## LEV fraction stats, without and with extrap
    lev_est = lad_trunc.sumLev(includeExtrap=False, asFraction=True)
    print(
        f"Mean inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
    lev_est = lad_trunc.extrapLAD.sumLev(asFraction=True)
    print(
        f"Mean non-inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
    lev_est = lad_trunc.sumLev(includeExtrap=True, asFraction=True)
    print(
        f"Mean total LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")

    # ## Area vs LEV plots (TODO: add extrap points)
    # fig, ax = plt.subplots()
    # # ax.scatter(lad_hl_trunc.Area_km2, lad_hl_trunc.LEV_MEAN)
    # sns.scatterplot(lad_trunc, x='Area_km2', y='LEV_MEAN', ax=ax, alpha=0.1)
    # ax.set_xscale('log')
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax})')
    # [ax.get_figure().savefig(
    #     f'/Volumes/thebe/pic/{roi_region}_area_vs_lev_v{v}', transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    ####################################
    ## Global lake area analysis
    ####################################

    fig, ax = plt.subplots()

    ## Load, process, and plot HydroLAKES
    emaxH = 0.5
    ladH = LAD.from_shapefile(hydrolakes_pth, area_var='Lake_area',
                              idx_var='Hylak_id', name='hydrolakes', region_var=None, other_vars=['Pour_lat', 'Pour_long'])
    binned_refH = BinnedLAD(lad_ref.truncate(tmin, 5), tmin, emaxH, compute_ci_lad=True,
                            extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
    ladH_trunc = ladH.truncate(emaxH, 300000)  # exclude Caspian sea
    # ladH_trunc = ladH.truncate(emaxH, np.inf)
    ladH_trunc.extrapolate(binned_refH)
    ladH_trunc.plot_extrap_lad(ax=ax, label='Lake area', error_bars=True,
                               normalized=False, color='grey', plotLegend=False)

    ## Remake plot for LAD
    lad_trunc.plot_extrap_lad(ax=ax, label='Lake area', error_bars=True,
                              normalized=False, color='cyan', plotLegend=False)
    # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
    # ax2=ax.twinx()
    ax.set_ylabel('Cumulative area (M$km^2$)')
    # ax.set_xlabel('')
    # ax2.set_ylabel('Cumulative area (normalized)')

    ## Compare to Downing 2016, re-using code snippet from BinnedLAD and plot over first plot
    btm = 0.001
    top = 100000
    nbins = 8
    # bins computed from nbins and edges
    bin_edges = np.concatenate(
        (np.geomspace(btm, top, nbins + 1), [np.inf])).round(6)
    area_bins = pd.IntervalIndex.from_breaks(bin_edges, closed='left')
    # X = np.array(list(map(interval_geometric_mean, area_bins))) # take geom mean of each interval to get X-val
    X = bin_edges[1:]  # plot against right bin edge
    d06 = [692600, 602100, 523400, 455100,
           392362, 329816, 257856, 607650, 378119]
    # from Downing 2006 paper
    group_sums = pd.Series(d06, index=area_bins, name='Area_km2')
    # Why are lower/upper non NaN?? Ignore.
    binnedAreas = confidence_interval_from_extreme_regions(
        group_sums, None, None, name='Area_km2')

    # ## Put Downing number into my BinnedLAD data structure, just to verify plot
    # lad_d06 = BinnedLAD(btm=btm, top=top, nbins=nbins, binned_areas=binnedAreas, compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
    # # lad_d06_canon = BinnedLAD(btm=bin_edges[4], top=bin_edges[-2], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[4:-1], None, None, name='Area_km2'), compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
    # # lad_d06_extrap = BinnedLAD(btm=bin_edges[0], top=bin_edges[4], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[:4], None, None, name='Area_km2'), compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
    # lad_d06.plot(ax=ax, show_rightmost=False, as_lineplot=True, as_cumulative=True) # plot as binnedLAD, skipping top bin with Caspian Sea

    # fig, ax = plt.subplots()
    # ax.plot(X, np.cumsum(d06)/np.sum(d06)) # units Mkm2 /1e6
    d06_canonical = d06[4:]
    d06_extrap = d06[:4]
    # This time, exclude top bin to better compare with BAWLD domain
    ax.plot(X[:-1], np.cumsum(d06[:-1]) / 1e6,
            color='orange', marker='x', linestyle='dashed')
    ax.plot(X[4:-1], (np.cumsum(d06_canonical[:-1]) +
            np.sum(d06_extrap)) / 1e6, color='orange')  # Plot canonical
    # ax.plot(X[:4], np.cumsum(d06_extrap)/np.sum(d06[:-1]), color='orange', linestyle='dashed') # Plot extrap
    fig.tight_layout()
    ax.set_yscale('linear')
    ax.set_ylim([0, np.sum(d06) / 1e6 + 0.2])
    # ax.set_xscale('log')
    # ax.set_xticks(X)

    # print(f'Area in two smallest bins: {np.sum(d06[:2])/1e6}\nArea in three largest: {np.sum(d06[-3:])/1e6}')
    [ax.get_figure().savefig(
        f'/Volumes/thebe/pic/GlobalLAD_D16_HL_PLD_v{v}', transparent=True, dpi=300) for ext in ['.png', '.pdf']]

    ###########################
    ## Create Table
    ## Create extrap LAD with all bin edge lining up with powers of 10, for Table
    ## All bins are from lad_hl_trunc_log10bins, derived from lad, the original data used to derive plot data
    ## Bottom bins use different extrap bins than lad_hl_trunc, which is used for plots
    ###########################

    ## Extrapolate with log10 bins
    log_bins_lower = [tmin, 0.001, 0.01, emax]
    # Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
    binned_ref_log10bins = BinnedLAD(lad.truncate(tmin, tmax), tmin, emax, bins=log_bins_lower, compute_ci_lad=False,
                                     extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
    # binned_lev_log10bins = BinnedLAD(lad_lev_cat, bins=log_bins_lower, compute_ci_lev=True,
    #                                  extreme_regions_lev=extreme_regions_lev_for_extrap)  # 0.000125 is native
    # Beware chaining unless I return a new variable. # Try 0.1
    lad_hl_trunc_log10bins = lad.truncate(emax, np.inf)
    lad_hl_trunc_log10bins.extrapolate(
        binned_ref_log10bins)  # , binned_lev_log10bins)

    ## Predict flux on extrapolated part (re-computes for observed part)
    lad_hl_trunc_log10bins['Temp_K'] = lad_hl_trunc_log10bins[temps_var]
    del lad_hl_trunc_log10bins[temps_var]
    lad_hl_trunc_log10bins.predictFlux(model, includeExtrap=True)

    ## bin upper with log10 bins
    log_bins_upper = [emax, 0.1, 1, 10, 100, 1000, 10000, 100000]
    # now, bin upper values for Table estimate, use regions as placeholder to get dummy CI
    lad_hl_trunc_log10bins_binned = BinnedLAD(lad_hl_trunc_log10bins, bins=log_bins_upper, compute_ci_lad=False,
                                              compute_ci_lev=False, compute_ci_lev_existing=True, normalize=False)

    ## Add placeholder attributes prior to combining in function
    # Note:extrapolated values use LEV as placeholder, but should be ignored.
    lad_hl_trunc_log10bins.extrapLAD.binnedDC = lad_hl_trunc_log10bins.extrapLAD.binnedLEV.rename(
        'dc')
    lad_hl_trunc_log10bins.extrapLAD.binnedCounts = (
        lad_hl_trunc_log10bins.extrapLAD.binnedLEV * np.nan).rename('Count')

    ## Combine binnedLADs using nifty function
    lad_binned_cmb = combineBinnedLADs(
        (lad_hl_trunc_log10bins.extrapLAD, lad_hl_trunc_log10bins_binned))

    ## Make table: can ignore CI if nan or same as mean
    tb_comb = pd.concat(
        (lad_binned_cmb.binnedAreas / 1e6, lad_binned_cmb.binnedLEV, lad_binned_cmb.binnedG_day * 365.25 / 1e12, lad_binned_cmb.binnedDC, lad_binned_cmb.binnedCounts), axis=1)

    ## Change units
    mean_rows = tb_comb.loc[:, 'mean', :]
    tb_comb['dc'] = tb_comb['dc'] * mean_rows.Area_km2
    tb_comb['LEV_frac'] = tb_comb['LEV_frac'] * mean_rows.Area_km2
    tb_comb.columns = ['Area_Mkm2', 'LEV_Mkm2', 'Tg_yr', 'DC_Mkm2', 'Count']

    ## Report double counting
    # dummy = lad[~np.isnan(lad.d_counting)]
    dummy = lad.fillna(0)  # assuming missing lakes have 0 LEV
    print(
        f"Double counting of inventoried lakes: {np.average(dummy.d_counting_frac, weights = dummy.Area_km2):0.3}%")

    ## Combine intervals
    grouped_tb_mean, grouped_tb_lower, grouped_tb_upper = [tb_comb.loc[:, stat, :].groupby(
        by=interval_group, observed=False).sum() for stat in ['mean', 'lower', 'upper']]

    ## Normalize
    grouped_tb_mean_norm, grouped_tb_lower_norm, grouped_tb_upper_norm = map(
        norm_table, [grouped_tb_mean, grouped_tb_lower, grouped_tb_upper], [grouped_tb_mean] * 3)

    ## Save to Excel sheets
    ## Create a Pandas Excel writer using XlsxWriter as the engine.
    sheets = ['Mean', 'Min', 'Max']
    tbl_pth = os.path.join(tb_dir, f'Size_bin_table_v{v}.xlsx')
    with pd.ExcelWriter(tbl_pth) as writer:
        [df.to_excel(writer, sheet_name=sheets[i]) for i, df in enumerate(
            [grouped_tb_mean, grouped_tb_lower, grouped_tb_upper])]
        print(f'Table written: {tbl_pth}')
    tbl_pth = os.path.join(tb_dir, f'Size_bin_table_norm_v{v}.xlsx')
    with pd.ExcelWriter(tbl_pth) as writer:
        [df.to_excel(writer, sheet_name=sheets[i]) for i, df in enumerate(
            [grouped_tb_mean_norm, grouped_tb_lower_norm, grouped_tb_upper_norm])]
        print(f'Table written: {tbl_pth}')

    ## Print totals
    sums = grouped_tb_mean.sum(axis=0)
    print(f"Total area: {sums['Area_Mkm2']:0.3} Mkm2")
    print(f"Total LEV: {sums['LEV_Mkm2']:0.3} Mkm2")
    print(f"Total flux: {sums['Tg_yr']:0.3} Tg/yr")
    print(f"Total double counting: {sums['DC_Mkm2']:0.3} Mkm2")
    ## print number of ref lakes:
    # len(lad_hl_trunc)
    # lad_hl_trunc.refBinnedLAD.binnedCounts.sum()
    # lad_hl_trunc.extrapLAD.sumAreas()

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

    ## What percentage of HL is 0-50 Oc bin?
    # print('Load HL...')
    # lad_hl = LAD.from_shapefile(gdf_HL_jn_pth, area_var=hl_area_var, idx_var='Hylak_id', name='HL', region_var=None) # Need to load version with joined in Oc stats per lake
    # use df_hl_nearest_bawld

    ###########################
    ## Compare emissions from small lakes if they are considered lakes vs wl
    # 33 marsh studies.
    # Marshes: mean: 171.6
    # Marshes: median: 106.0
    # Median flux from lakes < 0.001 km2: 90.6
    # Mean flux from lakes < 0.001 km2: 185.8
    # Flux ratio (Marsh:OW, method 1): 3.1
    ###########################

    # ## filter out small size bins and compute area-weighted emissions # TODO: run with the actual veg methane estimate I'll be using # TODO: double check why my values are so much lower than BAWLD means, even when I compare my non-inv mean flux to 0.1-1 km2 BAWLD size bin
    # non_inv_lks = tb_comb.loc[(tb_comb.index.get_level_values(
    #     'size_bin')[0:9], 'mean'), :]  # non-inventoried lakes
    # non_inv_lks_mean_flux = non_inv_lks.Tg_yr.sum() / 365.25 * 1e15 / \
    #     (non_inv_lks.Area_Mkm2.sum() * 1e6 *
    #      1e6)  # area-weighted mean flux in mg/m2/day
    # all_lks_mean_flux = tb_comb.Tg_yr.sum() / 365.25 * 1e15 / \
    #     (tb_comb.Area_Mkm2.sum() * 1e6 * 1e6)
    # # emissions factor (ratio) for non-inventoried lakes (compare to 3.1 for wl, from bald marsh:lake emissions ratio)
    # emissions_factor_ni_lks = non_inv_lks_mean_flux / all_lks_mean_flux

    # ## load bawld veg
    # df_terr = pd.read_csv('/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD-CH4/data/ek_out/BAWLD_CH4_Terrestrial.csv',
    #                       encoding="ISO-8859-1", dtype={'CH4.E.FLUX ': 'float'}, na_values='-')
    # df_terr.query('Class == "Marshes"')['CH4Av'].median()

    ####################################
    ## WBD comparison Analysis
    ####################################

    # ## Compare HL extrapolation to WBD:
    # assert roi_region == 'WBD_BAWLD', f"Carefull, you are comparing to WBD, but roi_region is {roi_region}."
    # ax = plotEmaxSens(lad, extreme_regions_lad, lad, lad_wbd, tmin=0.0001, tmax=5, emax=0.5, y_scaler=1e6*1.11)
    # [ax.get_figure().savefig(f'/Volumes/thebe/pic/WBD_HL_compare_v{v}'+ext, transparent=False, dpi=300) for ext in ['.png','.pdf']]

    # ## Sensitivity test for emax
    # emax_vals = [0.1, 0.5, 1, 3]
    # fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
    # sns.set_context('poster')
    # sns.set_style('ticks')
    # for i, emax in enumerate(emax_vals):
    #     plotEmaxSens(lad, extreme_regions_lad, lad, lad_wbd, tmin=0.0001, tmax=5, emax=emax, y_scaler=1e6*1.12, ax=axes.flatten()[i])
    # fig.set_tight_layout(tight=True)
    # sns.set_theme('notebook', font='Ariel')
    # sns.set_style('ticks')
    # [fig.savefig(f'/Volumes/thebe/pic/WBD_HL_emax_sensitivity_v{v}'+ext, transparent=False, dpi=300) for ext in ['.png','.pdf']]

    # ## Report vals (for WBD)
    # wbd_sum = lad_wbd.truncate(0.001, 10000).Area_km2.sum()
    # hl_extrap_sum = lad_hl_trunc.truncate(0, 10000).sumAreas() # This uses lad_hl_truncate used for prediction, earlier in the script.
    # print(f'{wbd_sum:,.0f} vs {hl_extrap_sum:,.0f} km2 ({((hl_extrap_sum - wbd_sum) / hl_extrap_sum):.2%}) difference between observed datasets WBD and HL in {roi_region}.')
    # print(f'WBD area fraction < 0.01 km2: {lad_wbd.truncate(0.001, np.inf).area_fraction(0.01):,.2%}')
    # print(f'WBD area fraction < 0.1 km2: {lad_wbd.truncate(0.001, np.inf).area_fraction(0.1):,.2%}')
    # print(f'WBD area fraction < 0.5 km2: {lad_wbd.truncate(0.001, np.inf).area_fraction(0.5):,.2%}')

    # ## Compare HL to WBD measured lakes in same domain:
    # # lad_hl.truncate(0, 1000).plot_lad(all=False, reverse=False, normalized=False)
    # # lad_hl = LAD.from_shapefile(gdf_HL_jn_pth, area_var='Shp_Area', idx_var='Hylak_id', name='HL', region_var=None) # reload, if needed # don't truncate this time
    # ax = lad_wbd.truncate(0.1, 1000).plot_lad(all=False, reverse=False, normalized=False, color='r')
    # lad.truncate(0.1, 1000).plot_lad(normalized=False, reverse=False, ax=ax, all=False) # need to have loaded proper lad hl bawld
    # ax.set_title(f'[{roi_region}]')
    # ax.get_figure().tight_layout()

    # ## Compare WBD [self-]extrapolation to WBD (control tests):
    # # lad_hl.truncate(0, 1000).plot_lad(all=False, reverse=False, normalized=False)
    # tmin, tmax = (0.001, 30) # Truncation limits for ref LAD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
    # emax = 0.5 # Extrapolation limit emax defines the left bound of the index region (and right bound of the extrapolation region).
    # # binned_ref = BinnedLAD(lad_wbd.truncate(tmin, tmax), tmin, emax) # uncomment to use self-extrap
    # # txt='self-'
    # binned_ref = BinnedLAD(lad.truncate(tmin, tmax), tmin, emax, compute_ci_lad=True, extreme_regions_lad=extreme_regions_lad)
    # txt=''
    # lad_wbd_trunc = lad_wbd.truncate(emax, np.inf)
    # lad_wbd_trunc.extrapolate(binned_ref)
    # ax = lad_wbd.truncate(0.001, 1000).plot_lad(all=False, reverse=False, normalized=False, color='blue', plotLegend=False)
    # lad_wbd_trunc.truncate(0.001, 1000).plot_extrap_lad(label=f'WBD-{txt}extrapolated', normalized=False, ax=ax, error_bars=True, plotLegend=False)

    # ax2=ax.twinx() # Add normalized axis
    # ymin, ymax = ax.get_ylim()
    # ax2.set_ylim([ymin, ymax/lad_wbd_trunc.sumAreas()*1e6*1.28]) # hot fix
    # ax2.set_ylabel('Cumulative area fraction')
    # ax.get_figure().tight_layout()
    # [ax.get_figure().savefig(f'/Volumes/thebe/pic/WBD_WBD_compare_v{v}'+ext, transparent=False, dpi=300) for ext in ['.png','.pdf']]

    ####################################
    ## Write out datasets for archive
    ####################################

    ## Add temperatures to HL_lev dataset (don't use truncated, because data users can easily truncate by lake area)
    keys = [temperature_metric]
    values = ['Temp_' + key for key in keys]
    rename_dict = {k: v for k, v in zip(keys, values)}
    keys_oc = ['0-5', '5-50', '50-95', '95-100']
    values_oc = ['Oc_' + key.replace('-', '_') for key in keys_oc]
    oc_dict = {k: v for k, v in zip(keys_oc, values_oc)}
    rename_dict.update(oc_dict)
    keys = ['LEV_MAX', 'LEV_MEAN', 'LEV_MIN']
    values = [key.replace('LEV', 'LAV') for key in keys]
    lav_dict = {k: v for k, v in zip(keys, values)}  # lake aquatic veg
    rename_dict.update(lav_dict)
    rename_dict.update({'idx_HL': 'Hylak_id'})

    # Format data types
    lad_save = lad.drop(columns=['Region', 'Temp_K_wght_sum', 'LEV_MAX_km2',
                                               'LEV_MEAN_km2', 'LEV_MIN_km2', 'd_counting_km2']).rename(columns=rename_dict)
    lad_save['Hylak_id'] = lad_save['Hylak_id'].astype('int')
    # Get a list of columns with float data type
    float_columns = lad_save.select_dtypes(
        include=['float']).columns.tolist()
    lad_save[float_columns] = lad_save[float_columns].round(
        4)  # Apply rounding to float columns to reduce output file size

    ## Write out
    lad_save.to_csv(os.path.join(
        output_dir, f'{ds}_emissions_v{v}.csv'))

    ## Version of BAWLD_HL for archive (continue from Map Analysis section)
    assert 'gdf_bawld_sum_lev' in locals(), "Need to run Map Analysis segment first"

    ## Rename columns and rm redundant ones to reduce data sprawl
    columns_save = ['Cell_ID', 'Long', 'Lat', 'Shp_Area', 'Area_km2', 'lake_count',
                    'd_counting_km2', 'd_counting_grid_frac', 'Temp_K',
                    'est_mg_m2_day', 'est_g_day', 'LEV_MEAN_km2', 'LEV_MIN_km2',
                    'LEV_MAX_km2', 'LEV_MEAN_frac', 'LEV_MIN_frac', 'LEV_MAX_frac',
                    'LEV_MEAN_grid_frac', 'LEV_MIN_grid_frac', 'LEV_MAX_grid_frac', 'geometry']
    columns_save_lav = [s for s in columns_save if 'LEV' in s]
    values = list(map(lambda s: s.replace('LEV', 'LAV'), columns_save_lav))
    lav_dict = {k: v for k, v in zip(
        columns_save_lav, values)}  # lake aquatic veg
    lav_dict.update({'Area_km2': 'Lake_area_km2'})
    gdf_bawld_sum_lev_save = gdf_bawld_sum_lev[columns_save].rename(
        columns=lav_dict)
    float_columns = gdf_bawld_sum_lev_save.select_dtypes(
        include=['float']).columns.tolist()  # Get a list of columns with float data type
    gdf_bawld_sum_lev_save[float_columns] = gdf_bawld_sum_lev_save[float_columns].round(
        4)  # Apply rounding to float columns

    ## Write out as csv (full column names) and shapefile
    gdf_bawld_sum_lev_save.drop(columns='geometry').to_csv(
        os.path.join(output_dir, 'BAWLD_V1_LAV.csv'))
    gpd.GeoDataFrame(gdf_bawld_sum_lev_save).to_file(
        os.path.join(output_dir, 'BAWLD_V1_LAV.shp'), engine='pyogrio')  # Can ignore "value not written errors"

    ## Save extrapolations table
    lad_trunc.extrapLAD.to_df().to_csv(os.path.join(
        output_dir, f'{ds}_extrapolated_v{v}.csv'))
    pass

    ################
