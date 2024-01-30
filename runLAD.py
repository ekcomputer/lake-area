## Imports
from warnings import warn
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
from LAD import *

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
tb_dir = '/Volumes/thebe/Ch4/area_tables'
# dir for output data, used for data archive
output_dir = '/Volumes/thebe/Ch4/output'
v = 24  # Version number for file naming

## BAWLD domain
dataset = 'HL'
roi_region = 'BAWLD'
gdf_bawld_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
# HL clipped to BAWLD # note V4 is not joined to BAWLD yet
# gdf_HL_jn_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned.shp'
# above, but with all ocurrence values, not binned
# main data source
df_HL_jn_full_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full.csv.gz'
hl_area_var = 'Shp_Area'
hl_join_clim_pth = '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/joined_ERA5/HL_ERA5_stl1_v3.csv.gz'
bawld_join_clim_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1___Shapefile_jn_clim.csv'
# HL shapefile with ID of nearest BAWLD cell (still uses V3)
hl_nearest_bawld_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned_jnBAWLD.shp'
bawld_hl_output = f'/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/joined_lev/BAWLD_V1_LEV_v{v}.shp'

## BAWLD-NAHL domain
# dataset = 'HL'
# roi_region = 'WBD_BAWLD'
# gdf_bawld_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/BAWLD_V1_clipped_to_WBD.shp'
# df_HL_jn_full_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_full_jnBAWLD_roiNAHL.csv.gz' # main data source # HL clipped to BAWLD and WBD
# # gdf_HL_jn_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_binned_jnBAWLD_roiNAHL.shp'
# hl_area_var='Shp_Area'
# hl_join_clim_pth = '/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/out/joined_ERA5/HL_ERA5_stl1_v3.csv.gz'
# hl_nearest_bawld_pth = '/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v4/HL_zStats_Oc_binned_jnBAWLD.shp'

## BAWLD domain (Sheng lakes)
# dataset = 'Sheng'
# roi_region = 'BAWLD'
# gdf_bawld_pth = '/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/BAWLD_V1___Shapefile.zip'
# gdf_Sheng_pth = '/Volumes/thebe/Other/Sheng-Arctic-lakes/edk_out/clips/UCLA_ArcticLakes15_BAWLD.shp' # HL clipped to BAWLD
# sheng_area_var='area'

## dynamic vars
# analysis_dir = os.path.join('/Volumes/thebe/Ch4/Area_extrapolations','v'+str(version))
# area_extrap_pth = os.path.join(analysis_dir, dataset+'_sub'+roi_region+'_extrap.csv')
# os.makedirs(analysis_dir, exist_ok=True)

## Loading from CIR gdf
print('Load HR...')
regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta',
           'Mackenzie River Valley', 'Canadian Shield Margin', 'Canadian Shield', 'Slave River',
           'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North',
           'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
lad_cir = LAD.from_shapefile('/Volumes/thebe/Planet-SR-2/Classification/cir/dcs_fused_hydroLakes_buf_10_sum.shp',
                             area_var='Area', name='CIR', region_var='Region4', regions=regions, idx_var='OID_')

## Loading PeRL LAD
perl_exclude = ['arg0022009xxxx', 'fir0022009xxxx', 'hbl00119540701', 'hbl00119740617',
                'hbl00120060706', 'ice0032009xxxx', 'rog00219740726', 'rog00220070707',
                'tav00119630831', 'tav00119750810', 'tav00120030702', 'yak0012009xxxx',
                'bar00120080730_qb_nplaea.shp']
lad_perl = LAD.from_paths('/Volumes/thebe/PeRL/PeRL_waterbodymaps/waterbodies/*.shp',
                          area_var='AREA', name='perl', _areaConversionFactor=1000000, exclude=perl_exclude)

## Loading from Mullen
lad_mullen = LAD.from_paths('/Volumes/thebe/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/*_3Y_lakes-and-ponds.zip', _areaConversionFactor=1000000,
                            name='Mullen', computeArea=True)  # '/Volumes/thebe/Other/Mullen_AK_lake_pond_maps/Alaska_Lake_Pond_Maps_2134_working/data/[A-Z][A-Z]_08*.zip'

## Combine PeRL and CIR and Mullen
lad = LAD.concat((lad_cir, lad_perl, lad_mullen),
                 broadcast_name=True, ignore_index=True)

# ## plot
# lad.truncate(0.0001, 10).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=True, plotLabels=False)
# lad.truncate(0.0001, 10).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

# ## Plot just CIR
# lad_cir.truncate(0.0001, 5).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=False)
# lad_cir.truncate(0.0001, 5).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

# ## Plot just PeRL
# lad_perl.truncate(0.0001, 5).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=False)
# lad_perl.truncate(0.0001, 5).plot_lad(all=True, plotLegend=False, reverse=False, groupby_name=False, plotLabels=True)

# ## Compute extreme regions and save to spreadsheet
# df_regions = regionStats(lad)
# df_regions.to_csv(os.path.join(tb_dir, 'region_stats.csv'))

# ## YF compare
# LAD(lad.query("Region=='YF_3Y_lakes-and-ponds' or Region=='Yukon Flats Basin'"), name='compare').plot_lad(all=False, plotLegend=True, reverse=False, groupby_name=False)

####################################
## LEV Analysis
####################################

## Load csv and shapefiles
ref_names = ['CSB', 'CSD', 'PAD', 'YF']
extreme_regions_lev_for_extrap = ['CSD', 'PAD']
lad_lev_cat, ref_dfs = loadUAVSAR(ref_names)

## Create binned ref LEV distribution from UAVSAR
binned_lev = BinnedLAD(lad_lev_cat, 0.0001, 0.5, compute_ci_lev=True,
                       extreme_regions_lev=extreme_regions_lev_for_extrap)  # 0.000125 is native

## LEV estimate: Load UAVSAR/GSW overlay stats
print('Load HL with joined occurrence...')
# lad_hl_oc = pyogrio.read_dataframe('/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v3/HL_zStats_Oc_full.shp', read_geometry=False, use_arrow=True) # load shapefile with full histogram of zonal stats occurrence values # outdated version
# read smaller csv gzip version of data.
lad_hl_oc = pd.read_csv(
    df_HL_jn_full_pth, compression='gzip', low_memory=False)
lev = computeLAV(lad_hl_oc, ref_dfs, ref_names, extreme_regions_lev=extreme_regions_lev_for_extrap,
                 use_low_oc=use_low_oc)  # use same extreme regions for est as for extrap

## Set high arctic lakes LEV to 0 (no GSW present above 78 degN)
lev.loc[lev.Pour_lat >= 78, ['LEV_MEAN',
                             'LEV_MIN', 'LEV_MAX']] = 0  # LEV_MEAN

## Turn into a LAD
# main dataset for analysis
lad_hl_lev = LAD(lev, area_var='Lake_area', idx_var='Hylak_id', name='HL')

# ## Plot LEV CDF by lake area (no extrap) and report mean LEV fraction
# lad_hl_lev.plot_lev_cdf_by_lake_area()
# lad_hl_lev.plot_lev_cdf_by_lake_area(normalized=False)

####################################
## Climate Analysis: join in temperature
####################################
print('Loading BAWLD and climate data...')
# Index(['Unnamed: 0', 'BAWLDCell_', 'Hylak_id', 'Shp_Area', 'geometry','index_right', 'id', 'area', 'perimeter', 'lat', 'lon', 'djf', 'mam', 'jja', 'son', 'ann'],
df_clim = pd.read_csv(hl_join_clim_pth, compression='gzip')
# df_clim = pd.read_csv(bawld_join_clim_pth)
# gdf_bawld = gpd.read_file(gdf_bawld_pth, engine='pyogrio')
# df_clim = df_clim.merge(
#     gdf_bawld[['Cell_ID', 'Shp_Area']], how='left', on='Cell_ID')

## Next, load HL with nearest BAWLD:
# The 0-5 etc. columns refer to HL polygon, not BAWLD cell.
df_hl_nearest_bawld = pyogrio.read_dataframe(
    hl_nearest_bawld_pth, read_geometry=False)
# take only first lake (for cases where lake is equidistant from multiple cells)
df_hl_nearest_bawld = df_hl_nearest_bawld.groupby(
    'Hylak_id', observed=False).first().reset_index()
# Need to create new var because output of merge is not LAD
lad_hl_lev_m = lad_hl_lev.merge(df_hl_nearest_bawld[[
    'Hylak_id', 'BAWLD_Cell', '0-5', '5-50', '50-95', '95-100']], left_on='idx_HL', right_on='Hylak_id', how='left')

## Join in ERA5 temperatures from previously-computed lookup table
temperatures = lad_hl_lev_m[['idx_HL', 'BAWLD_Cell']].merge(
    df_clim, how='left', left_on='idx_HL', right_on='Hylak_id')
# Fill any missing data with mean
temperatures.fillna(temperatures.mean(), inplace=True)
lad_hl_lev['Temp_K'] = temperatures[temperature_metric]
lad_hl_lev['BAWLD_Cell'] = temperatures['BAWLD_Cell'].astype('int')

## Add binned occurrence values
for var in ['0-5', '5-50', '50-95', '95-100']:
    lad_hl_lev[var] = lad_hl_lev_m[var]

## Compute double-counting
lad_hl_lev['d_counting_frac'] = (
    lad_hl_lev['0-5'] + lad_hl_lev['5-50']) / 100

## Compute cell-area-weighted average of climate as FYI
# print(f'Mean JJA temperature across {roi_region} domain: {np.average(df_clim.jja, weights=df_clim.Shp_Area)}')
# months = ['ann','djf','mam','jja','son']
# print(pd.DataFrame(np.average(df_clim[months], weights=df_clim.Shp_Area, axis=0), index=months))

####################################
## Holdout Analysis
####################################

## Load measured holdout LEV dataset
# gdf_holdout = gpd.read_file('/Volumes/thebe/Ch4/misc/UAVSAR_polygonized/sub_roi/zonal_hist/v2_5m_bic/YF_train_holdout/zonal_hist_w_UAVSAR/YFLATS_190914_mosaic_rcls_brn_zHist_UAV_holdout.shp', engine='pyogrio')
# gdf_holdout = gpd.read_file('/Volumes/thebe/Ch4/misc/UAVSAR_zonal_hist_w_Oc/YFLATS_190914_mosaic_rcls_lakes_zHist_Oc.shp', engine='pyogrio')
# gdf_holdout = gpd.read_file('/Volumes/thebe/Ch4/misc/UAVSAR_zonal_hist_w_Oc/Merged/4sites_zHist_Oc.shp', engine='pyogrio') # full data, not holdout
gdf_holdout = gpd.read_file(
    '/Volumes/thebe/Ch4/misc/UAVSAR_zonal_hist_w_Oc/Merged/4sites_zHist_Oc_holdout.shp', engine='pyogrio')

## Pre-process to put Occurrence in format the function expects
gdf_holdout = convertOccurrenceFormat(gdf_holdout)

## Convert Occurence to LEV
a_lev_measured = computeLAV(gdf_holdout, ref_dfs, ref_names, extreme_regions_lev=extreme_regions_lev_for_extrap,
                            use_low_oc=use_low_oc)

## rm edge lakes and small lakes below HL limit
a_lev_measured = a_lev_measured.dropna(subset=['em_fractio', 'LEV_MEAN'])
a_lev_measured = a_lev_measured[a_lev_measured['area_px_m2'] >= 100000]
a_lev_measured = a_lev_measured.query('(edge==0) and (cir_observ==1)')

## Take area-weighted mean LEV
a_lev_pred = np.average(a_lev_measured[[
                        'LEV_MEAN', 'LEV_MIN', 'LEV_MAX']], axis=0, weights=a_lev_measured.area_px_m2)
a_lev = np.average(a_lev_measured.em_fractio,
                   weights=a_lev_measured.area_px_m2)

print(f'Measured A_LEV in holdout ds: {a_lev:0.2%}')
print(
    f'Predicted A_LEV in holdout ds: {a_lev_pred[0]:0.2%} ({a_lev_pred[1]:0.2%}, {a_lev_pred[2]:0.2%})')
corr_coeff, p_value = pearsonr(
    a_lev_measured.LEV_MEAN, a_lev_measured.em_fractio)
print(f'Correlation: {corr_coeff:0.2%}')
print(f'P: {p_value:0.2%}')
print(f'RMSE: {mean_squared_error(a_lev_measured.LEV_MEAN, a_lev_measured.em_fractio, squared=False):0.2%}')

## Compare RMSD of model to RMSD of observed UAVSAR holdout subset compared to average (Karianne's check)
print(
    f"RMSD of observed values from mean: {np.sqrt(1 /a_lev_measured.shape[0] * np.sum((a_lev_measured.em_fractio - a_lev_measured.em_fractio.mean())**2)):0.2%}")
print(
    f"RMSD of predicted values from mean: {np.sqrt(1 /a_lev_measured.shape[0] * np.sum((a_lev_measured.LEV_MEAN - a_lev_measured.LEV_MEAN.mean())**2)):0.2%}")

## Plot validation of LEV
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], '--k', alpha=0.7)  # one-to-one line
# Add the one-to-one line # ax.get_xlim(), ax.get_ylim()
ax.plot([0, 0.8], [0, 0.8], ls="--", c=".3")
sns.regplot(a_lev_measured, x='LEV_MEAN', y='em_fractio', ax=ax)
# sns.scatterplot(a_lev_measured, x='LEV_MEAN', y='em_fractio', hue='layer', alpha=0.7, ax=ax)

ax.set_xlabel('Predicted lake aquatic vegetation fraction')
ax.set_ylabel('Measured lake aquatic vegetation fraction')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
[ax.get_figure().savefig(
    f'/Volumes/thebe/pic/A_LEV_validation_v{v}', transparent=False, dpi=300) for ext in ['.png', '.pdf']]

## Load WBD

if roi_region == 'WBD_BAWLD':
    print('Load WBD...')
    lad_wbd = LAD.from_shapefile(
        '/Volumes/thebe/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', area_var='Area', name='WBD', idx_var='OBJECTID')
    lad_wbd.truncate(0.001, inplace=True)

## Plot WBD
# lad_wbd.plot_lad(reverse=False, all=False)

## Combine WBD with HR dataset over small lakes for plotting comparison
# setattr(lad, 'name', 'HR datasets')
# lad['Region'] = 'NaN' # Hot fix to tell it not to plot a curve for each region # ***This is the buggy line!!!! Uncomment to get good curves, but no error bars if I haven't set compute_ci = False.
# lad_compare = LAD.concat((lad.truncate(0.001, 50), lad_wbd.truncate(0.001, 50)), broadcast_name=True, ignore_index=True)
# lad_compare.plot_lad(all=False, plotLegend=True, reverse=False, groupby_name=True)

## Estimate area fraction
# lad_wbd.area_fraction(0.1)
# lad_wbd.area_fraction(0.01)
# lad_wbd.area_fraction(0.001)

# lad.area_fraction(0.1)
# lad.area_fraction(0.01)
# lad.area_fraction(0.001)

## Load hydrolakes. if not using lad_hl_lev
# print('Load HL...')
# lad_hl = LAD.from_shapefile(gdf_HL_jn_pth, area_var=hl_area_var, idx_var='Hylak_id', name='HL', region_var=None)

## Load sheng
# print('Load Sheng...')
# lad_hl = LAD.from_shapefile(gdf_Sheng_pth, area_var=sheng_area_var, idx_var=None, name='Sheng', region_var=None)

## Extrapolate
# ['Tuktoyaktuk Peninsula', 'Peace-Athabasca Delta']
extreme_regions_lad = [
    'Tuktoyaktuk Peninsula', 'sur00120130802_tsx_nplaea']
# Truncation limits for ref LAD. tmax defines the right bound of the index region. tmin defines the leftmost bound to extrapolate to.
tmin, tmax = (0.0001, 5)
# Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
emax = 0.5
binned_ref = BinnedLAD(lad.truncate(tmin, tmax), tmin, emax, compute_ci_lad=True,
                       extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
# Beware chaining unless I return a new variable. # Try 0.1
lad_hl_trunc = lad_hl_lev.truncate(emax, np.inf)
lad_hl_trunc.extrapolate(binned_ref, binned_lev)
meas = lad_hl_lev.sumAreas(includeExtrap=False)
extrap = lad_hl_trunc.sumAreas()

limit = 0.01
frac = lad_hl_trunc.extrapolated_area_fraction(lad, 0.0001, limit)
print(f'Total measured lake area in {roi_region} domain: {meas:,.0f} km2')
print(
    f'Total extrapolated lake area in {roi_region} domain: {extrap:,.0f} km2')
print(f'{1-(meas / extrap):.1%} of lake area is < 0.1 km2.')
print(f'{frac:.1%} of lake area is < {limit} km2.')
print(f'{lad_hl_trunc.extrapolated_area_fraction(lad, 0.0001, 0.001):.1%} of lake area is < 0.001 km2.')

## Report extrapolated area fractions (need method for area fractions on extrapolatedLAD)
# print(f'Area fraction < 0.01 km2: {lad.area_fraction(0.01):,.2%}')
# print(f'Area fraction < 0.1 km2: {lad.area_fraction(0.1):,.2%}')

# ## Plot HL extrapolation
# # ax = lad_hl.plot_lad(all=False, reverse=False, normalized=False)
# ax = lad_hl_trunc.plot_extrap_lad(label='HL-extrapolated', error_bars=False, normalized=False)
# ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')

## Flux prediction from observed and extrap lakes
model = loadBAWLD_CH4()
lad_hl_trunc.predictFlux(model, includeExtrap=True)

## Plot combined extrap LAD/LEV
fig, ax = plt.subplots(2, 1, sharex=True)
lad_hl_trunc.plot_extrap_lad(
    ax=ax[0], label='Lake area', error_bars=True, normalized=False, color='blue', plotLegend=False)
# ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
ax2 = ax[0].twinx()
lad_hl_trunc.plot_extrap_lev(
    ax=ax[0], error_bars=True, color='green', label='Lake vegetation area', plotLegend=False, )
ymin, ymax = ax[0].get_ylim()
ax2.set_ylim([ymin, ymax / lad_hl_trunc.sumAreas()])
ax[0].set_ylabel('Cumulative area (million $km^2$)')
ax[0].set_xlabel('')
ax2.set_ylabel('Cumulative area fraction')
# plt.tight_layout()

## Plot extrapolated fluxes
lad_hl_trunc.plot_extrap_flux(
    ax=ax[1], reverse=False, normalized=False, error_bars=True, plotLegend=False, label='Emissions')
ax2 = ax[1].twinx()
ymin, ymax = ax[1].get_ylim()
ax2.set_ylim([ymin, ymax / lad_hl_trunc._total_flux_Tg_yr['mean']])
ax2.set_ylabel('Cumulative emissions fraction')
plt.tight_layout()
[ax2.get_figure().savefig(
    f'/Volumes/thebe/pic/BAWLD_areas_v{v}' + ext, transparent=True, dpi=300) for ext in ['.png', '.pdf']]

# ## Plot combined extrap LAD/Flux
# norm = True # False
# ax = lad_hl_trunc.plot_extrap_lad(label='HL-extrapolated', error_bars=True, normalized=False, color='blue')
# # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
# ax2=ax.twinx()
# lad_hl_trunc.plot_extrap_flux(ax=ax2, reverse=False, normalized=norm, error_bars=True)
# plt.tight_layout()

## Plot inset with just LEV, with normalized second axis
sns.set_theme('poster', font='Ariel')
sns.set_style('ticks')
ax = lad_hl_trunc.plot_extrap_lev(
    error_bars=True, color='green', plotLegend=False)
ax2 = ax.twinx()
ymin, ymax = ax.get_ylim()
ax2.set_ylim([ymin, ymax / lad_hl_trunc.sumLev()['mean']])
ax.set_ylabel('')  # 'Cumulative aquatic vegetation area (million $km^2$)')
ax2.set_ylabel('')  # 'Cumulative aquatic vegetation area fraction')
plt.tight_layout()
[ax.get_figure().savefig(
    f'/Volumes/thebe/pic/BAWLD_areas_inset_v{v}', transparent=True, dpi=300) for ext in ['.png', '.pdf']]
sns.set_theme('notebook', font='Ariel')
sns.set_style('ticks')

# ## Retrieve data from plot
# ax.get_lines()[0].get_ydata() # gives right part of LAD plot # [1] is left part
# ax2.get_lines()[0].get_ydata() # gives right part of LEV plot
# X_lev = np.concatenate((ax.get_lines()[1].get_xdata(), ax.get_lines()[0].get_xdata()))
# S_lev = np.concatenate((ax.get_lines()[1].get_ydata(), ax.get_lines()[0].get_ydata()))

## LEV fraction stats, without and with extrap
lev_est = lad_hl_trunc.sumLev(includeExtrap=False, asFraction=True)
print(
    f"Mean inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
lev_est = lad_hl_trunc.extrapLAD.sumLev(asFraction=True)
print(
    f"Mean non-inventoried-lake LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")
lev_est = lad_hl_trunc.sumLev(includeExtrap=True, asFraction=True)
print(
    f"Mean total LEV: {lev_est['mean']:0.2%} ({lev_est['lower']:0.2%}, {lev_est['upper']:0.2%})")

## Area vs LEV plots (TODO: add extrap points)
fig, ax = plt.subplots()
# ax.scatter(lad_hl_trunc.Area_km2, lad_hl_trunc.LEV_MEAN)
sns.scatterplot(lad_hl_trunc, x='Area_km2', y='LEV_MEAN', ax=ax, alpha=0.1)
ax.set_xscale('log')
ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax})')
[ax.get_figure().savefig(
    f'/Volumes/thebe/pic/{roi_region}_area_vs_lev_v{v}', transparent=True, dpi=300) for ext in ['.png', '.pdf']]

####################################
## Global lake area analysis
####################################

# ## Remake plot for LAD
# fig, ax = plt.subplots()
# lad_hl_trunc.plot_extrap_lad(ax=ax, label='Lake area', error_bars=True, normalized=True, color='grey', plotLegend=False)
# # ax.set_title(f'[{roi_region}] truncate: ({tmin}, {tmax}), extrap: {emax}')
# # ax2=ax.twinx()
# ax.set_ylabel('Cumulative area (normalized)')
# # ax.set_xlabel('')
# # ax2.set_ylabel('Cumulative area (normalized)')

# ## Compare to Downing 2016, re-using code snippet from BinnedLAD and plot over first plot
# btm = 0.001
# top = 100000
# nbins = 8
# bin_edges = np.concatenate((np.geomspace(btm, top, nbins+1), [np.inf])).round(6) # bins computed from nbins and edges
# area_bins = pd.IntervalIndex.from_breaks(bin_edges, closed='left')
# # X = np.array(list(map(interval_geometric_mean, area_bins))) # take geom mean of each interval to get X-val
# X = bin_edges[1:] # plot against right bin edge
# d06 = [692600, 602100, 523400, 455100, 392362, 329816, 257856, 607650, 378119]
# group_sums = pd.Series(d06, index=area_bins, name='Area_km2') # from Downing 2006 paper
# binnedAreas = confidence_interval_from_extreme_regions(group_sums, None, None, name='Area_km2') # # Why are lower/upper non NaN?? Ignore.

# ## Put Downing number into my BinnedLAD data structure, just to verify plot
# lad_d06 = BinnedLAD(btm=btm, top=top, nbins=nbins, binned_areas=binnedAreas, compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
# # lad_d06_canon = BinnedLAD(btm=bin_edges[4], top=bin_edges[-2], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[4:-1], None, None, name='Area_km2'), compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
# # lad_d06_extrap = BinnedLAD(btm=bin_edges[0], top=bin_edges[4], nbins=4, binned_areas=confidence_interval_from_extreme_regions(group_sums[:4], None, None, name='Area_km2'), compute_ci_lad=False) # give btm, top, nbins, compute_ci_lad and binnedAreas args
# lad_d06.plot(ax=ax, show_rightmost=False, as_lineplot=True, as_cumulative=True) # plot as binnedLAD, skipping top bin with Caspian Sea

# # fig, ax = plt.subplots()
# # ax.plot(X, np.cumsum(d06)/np.sum(d06)) # units Mkm2 /1e6
# d06_canonical = d06[4:]
# d06_extrap = d06[:4]
# ax.plot(X[:-1], np.cumsum(d06[:-1])/np.sum(d06[:-1]), color='orange', marker='x',linestyle='dashed') # This time, exclude top bin to better compare with BAWLD domain
# ax.plot(X[4:-1], (np.cumsum(d06_canonical[:-1])+np.sum(d06_extrap))/(np.sum(d06_canonical[:-1]) + np.sum(d06_extrap)), color='orange') # Plot canonical
# # ax.plot(X[:4], np.cumsum(d06_extrap)/np.sum(d06[:-1]), color='orange', linestyle='dashed') # Plot extrap
# fig.tight_layout()
# ax.set_yscale('linear')
# # ax.set_xscale('log')
# # ax.set_xticks(X)

# # print(f'Area in two smallest bins: {np.sum(d06[:2])/1e6}\nArea in three largest: {np.sum(d06[-3:])/1e6}')

###########################
## Create Table
## Create extrap LAD with all bin edge lining up with powers of 10, for Table
## All bins are from lad_hl_trunc_log10bins, derived from lad_hl_lev, the original data used to derive plot data
## Bottom bins use different extrap bins than lad_hl_trunc, which is used for plots
###########################

## Extrapolate with log10 bins
log_bins_lower = [tmin, 0.001, 0.01, 0.1, emax]
# Extrapolation limits. emax defines the left bound of the index region (and right bound of the extrapolation region).
binned_ref_log10bins = BinnedLAD(lad.truncate(tmin, tmax), bins=log_bins_lower, compute_ci_lad=True,
                                 extreme_regions_lad=extreme_regions_lad)  # reference distrib (try 5, 0.5 as second args)
binned_lev_log10bins = BinnedLAD(lad_lev_cat, bins=log_bins_lower, compute_ci_lev=True,
                                 extreme_regions_lev=extreme_regions_lev_for_extrap)  # 0.000125 is native
# Beware chaining unless I return a new variable. # Try 0.1
lad_hl_trunc_log10bins = lad_hl_lev.truncate(emax, np.inf)
lad_hl_trunc_log10bins.extrapolate(
    binned_ref_log10bins, binned_lev_log10bins)

## Predict flux on extrapolated part (re-computes for observed part)
lad_hl_trunc_log10bins.predictFlux(model, includeExtrap=True)

## bin upper with log10 bins
log_bins_upper = [0.5, 1, 10, 100, 1000, 10000, 100000]
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
# dummy = lad_hl_lev[~np.isnan(lad_hl_lev.d_counting)]
dummy = lad_hl_lev.fillna(0)  # assuming missing lakes have 0 LEV
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
    lad_hl_lev[col + '_km2'] = lad_hl_lev[col] * \
        lad_hl_lev['Area_km2']  # add absolute area units
# lad_hl_lev.to_csv('/Volumes/thebe/Ch4/GSW_zonal_stats/HL/v5/HL_BAWLD_LEV.csv')

## Rescale double-counting to km2 for data archival purposes
lad_hl_lev['d_counting_km2'] = lad_hl_lev.d_counting_frac * \
    lad_hl_lev['Area_km2']

## Prep weighted avgs
lad_hl_lev.predictFlux(model, includeExtrap=False)
lad_hl_lev['Temp_K_wght_sum'] = lad_hl_lev.Temp_K * lad_hl_lev.Area_km2

## Groupby bawld cell and compute sum of LEV and weighted avg of LEV
df_bawld_sum_lev = lad_hl_lev.groupby('BAWLD_Cell', observed=False).sum(
    numeric_only=True)  # Could add Occ

## Lake count
df_bawld_sum_lev['lake_count'] = lad_hl_lev[['Area_km2', 'BAWLD_Cell']].groupby(
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
# ax = plotEmaxSens(lad_hl_lev, extreme_regions_lad, lad, lad_wbd, tmin=0.0001, tmax=5, emax=0.5, y_scaler=1e6*1.11)
# [ax.get_figure().savefig(f'/Volumes/thebe/pic/WBD_HL_compare_v{v}'+ext, transparent=False, dpi=300) for ext in ['.png','.pdf']]

# ## Sensitivity test for emax
# emax_vals = [0.1, 0.5, 1, 3]
# fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
# sns.set_context('poster')
# sns.set_style('ticks')
# for i, emax in enumerate(emax_vals):
#     plotEmaxSens(lad_hl_lev, extreme_regions_lad, lad, lad_wbd, tmin=0.0001, tmax=5, emax=emax, y_scaler=1e6*1.12, ax=axes.flatten()[i])
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
# lad_hl_lev.truncate(0.1, 1000).plot_lad(normalized=False, reverse=False, ax=ax, all=False) # need to have loaded proper lad hl bawld
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
lad_hl_lev_save = lad_hl_lev.drop(columns=['Region', 'Temp_K_wght_sum', 'LEV_MAX_km2',
                                           'LEV_MEAN_km2', 'LEV_MIN_km2', 'd_counting_km2']).rename(columns=rename_dict)
lad_hl_lev_save['Hylak_id'] = lad_hl_lev_save['Hylak_id'].astype('int')
# Get a list of columns with float data type
float_columns = lad_hl_lev_save.select_dtypes(
    include=['float']).columns.tolist()
lad_hl_lev_save[float_columns] = lad_hl_lev_save[float_columns].round(
    4)  # Apply rounding to float columns to reduce output file size

## Write out
lad_hl_lev_save.to_csv(os.path.join(
    output_dir, f'HydroLAKES_emissions_v{v}.csv'))

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
lad_hl_trunc.extrapLAD.to_df().to_csv(os.path.join(
    output_dir, f'HydroLAKES_extrapolated_v{v}.csv'))
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
* Re-define LAD so if called with no args but proper column names it returns a LAD correctly.
* Use fid_as_index argument when loading with pyarrow
* Preserve og index when concatenating so I can look up lakes from raw file (combine with above re: fid)
* Branches for if there is no CI/error bars in binned distrib. Make sure there is still a second index called 'stat' with constant val 'mean'
* Rewrite sumLev() to output a series x
* Go back and add branches for no CI to the various methods. Make sure it still has a second index for 'stat' with constant val 'mean'
* Make predictFlux() calls consistent bw LAD and BinnedLAD, whether they return a value or add an attribute.
* Thorough testing of all function options
* Search for "TODO"
* Code in LEV flux calc from table-2-stats
* Most awkward part of the LAD class is that I can't use any builtin pandas function without returning a DataFrame, so I have developed ways to re-initiate a LAD from a DataFrame to use when needed.
*Possible solution: re-define LAD class to be a genric structre that has an LAD attribute that is simply a dataframe. Re-define operaters print/__repr__ and slicing operations so it still behaves like the base structure is a df.
* Add binnedLAD.truncate() method that removes bins
* Truncate CSV decimals when writing out data files
* Add confusion matrix output steps (in LEV_GSW_overlay.ipynb) to main script.
* replace sklearn 'mean_squared_error' function to make package easier to install.
* Fix runtime div by 0 warnings
* Publish to pypi
* install tests for mac - copy geospatial

NOTES:
* Every time a create an LAD() object in a function from an existing LAD (e.g. making a copy), I should pass it the public attributes of its parent, or they will be lost.
'''
