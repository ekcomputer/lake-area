# tests/test_LAD.py

import sys
import unittest
import numpy as np
import pandas as pd
import pyogrio
from LAD.LAD import *
from LAD.util import *

lad_hr_pth = 'sample_data/CIR_Canadian_Shield.shp'
lad_lr_pth = 'sample_data/HydroLAKESv10_Sweden.shp'
lad_lr_oc_pth = 'sample_data/HydroLAKESv10_Sweden_Occurrence.csv.gz'
lev_pth = 'sample_data/LEV_Canadian_Shield.shp'
lev_csv_pth = 'sample_data/LEV_Canadian_Shield_train_s.csv'


class TestLAD(unittest.TestCase):
    ''' TODO: add more tests, starting with predicting emissions (need to include temperatures in sample data)'''

    def test_LAD_load(self):
        '''Tests a variety of keyword arguments and their defaults for loading.'''
        lad_from_shp = LAD.from_shapefile(
            lad_hr_pth, area_var='Area', name='CIR', region_var='Region4', other_vars=['Category4'])
        lad_from_shp = LAD.from_shapefile(
            lad_lr_pth, area_var='Lake_area', idx_var='Hylak_id', name='HL', region_var=None)
        lad_from_shp = LAD.from_shapefile(lad_lr_pth, area_var='Lake_area')
        lad_from_shp = LAD.from_shapefile(
            lad_hr_pth, computeArea=True, areaConversionFactor=1e6)
        self.assertIsInstance(lad_from_shp, LAD)
        self.assertIn('Area_km2', lad_from_shp.columns)

    def test_LAD_load_from_gdf(self):
        '''Tests a variety of keyword arguments and their defaults for loading from geodataframe'''
        gdf = pyogrio.read_dataframe(
            lad_hr_pth, read_geometry=True, use_arrow=True)
        lad_from_gdf = LAD(gdf, area_var='Area',
                           name='CIR', region_var='Region4')
        regions = ['Sagavanirktok River', 'Yukon Flats Basin', 'Old Crow Flats', 'Mackenzie River Delta', 'Mackenzie River Valley', 'Canadian Shield Margin',
                   'Canadian Shield', 'Slave River', 'Peace-Athabasca Delta', 'Athabasca River', 'Prairie Potholes North', 'Prairie Potholes South', 'Tuktoyaktuk Peninsula', 'All']
        lad_from_gdf = LAD(gdf, area_var='Area', name='CIR',
                           region_var='Region4', regions=regions, idx_var='OID_')
        self.assertIsInstance(lad_from_gdf, LAD)

    def test_concat(self):
        '''Tests concatenation of 2 LAD objects. Concatenation must return an LAD object.'''
        lad_from_shp = LAD.from_shapefile(
            lad_hr_pth, area_var='Area', name='CIR', region_var='Region4')
        lad_concat = LAD.concat((lad_from_shp, lad_from_shp))
        lad_concat = LAD.concat(
            (lad_from_shp, lad_from_shp), broadcast_name=True)
        self.assertIsInstance(lad_concat, LAD)

    def test_truncate(self):
        '''Tests truncation of LAD object. Truncation must return an LAD object.'''
        lad_from_shp = LAD.from_shapefile(lad_lr_pth, area_var='Lake_area')
        lad_from_shp.truncate(0.01, 20)
        self.assertIsInstance(lad_from_shp, LAD)

    def test_area_fraction(self):
        '''Tests area fraction computation on LAD object.'''
        lad_from_shp = LAD.from_shapefile(lad_lr_pth, area_var='Lake_area')
        aF = lad_from_shp.area_fraction(1)
        self.assertLessEqual(aF, 1)
        self.assertGreaterEqual(aF, 0)

    def test_plots(self):
        lad_from_shp = LAD.from_shapefile(lad_lr_pth, area_var='Lake_area')
        # Should produce a plot in debug mode
        plotEPDFByValue(lad_from_shp.Area_km2)

    def test_workflow(self):
        lad_hl_oc = pd.read_csv(lad_lr_oc_pth, compression='gzip')
        lad_lev_cat, ref_dfs = loadUAVSAR(ref_names=['CSB'], pths_shp=[
                                          lev_pth], pths_csv=[lev_csv_pth])
        self.assertIsInstance(lad_lev_cat, LAD)
        self.assertIsInstance(ref_dfs, list)
        self.assertIsInstance(ref_dfs[0], pd.DataFrame)

        ## Load from csv and compute LAV from LEV
        lev = computeLAV(lad_hl_oc, ref_dfs, ['CSB'],
                         # use same region for extremes bc limited sample data
                         extreme_regions_lev=['CSB', 'CSB'])
        lad_lev = LAD(lev, area_var='Lake_area',
                      idx_var='Hylak_id', region_var='Country')
        self.assertIsInstance(lad_lev, LAD)
        self.assertIn('LEV_MEAN', lad_lev.columns)
        self.assertIn('LEV_MIN', lad_lev.columns)

        ## Test plot LEV  CDF
        lad_lev.plot_lev_cdf()

        ## Test plot LEV CDF by lake area
        lad_lev.plot_lev_cdf_by_lake_area()
        mean = lad_lev.sumLev(asFraction=True, includeExtrap=False)['mean']
        print(f"Mean LEV: {mean:.2%}")

        ## Test binned LEV HL LAD (won't actually use this for analysis)
        lad_lev.Region = 'Region 4'
        lad_lev.loc[:4000, 'Region'] = 'Region 3'  # Give dummy regions
        binned = BinnedLAD(lad_lev.truncate(0.1, 1000), 5, 1000, compute_ci_lad=True, extreme_regions_lad=[
                           # compute_ci_lad=False will disable plotting CI.
                           'Region 3', 'Region 4'], normalize=False)

        ## Load HR data for binning a ref LAD
        lad_hr = LAD.from_shapefile(
            lad_hr_pth, area_var='Area', name='CIR', region_var='Region4', other_vars=['Category4'])
        lad_hr.Region = 'Region 2'
        lad_hr.loc[:1500, 'Region'] = 'Region 1'  # Give dummy regions

        ## Test binnedLAD with edges specified
        # compute_ci_lad=False will disable plotting CI.
        binned_manual = BinnedLAD(lad_hr.truncate(0.0001, 1), 0.0001, 0.5, bins=[
            0.001, 0.1, 0.5], compute_ci_lad=False)
        binned_manual.plot()

        ## Test binnedLAD for reference with additional tests for different parameters
        binned = BinnedLAD(lad_hr.truncate(0.0001, 1),
                           0.0001, 0.5, compute_ci_lad=True, extreme_regions_lad=['Region 1', 'Region 2'])
        binned.plot()
        binned.plot(as_lineplot=True)

        ## Test extrapolate on small data
        ref_binned_lev = BinnedLAD(
            lad_lev_cat, 0.0001, 0.5)  # todo: compute CI LEV
        # Beware chaining unless I return a new variable.
        lad_lr_trunc = lad_lev.truncate(0.5, np.inf, inplace=False)
        lad_lr_trunc.extrapolate(binned, ref_binned_lev)
        mean = lad_lr_trunc.sumLev(asFraction=True, includeExtrap=True)['mean']
        print(f"Mean LEV: {mean:.2%}")
        sum = lad_lr_trunc.sumAreas(includeExtrap=True)
        print(f"Total area: {sum:.6} km2")

        ## Test plot extrap LAD, LEV
        ax = lad_lr_trunc.plot_extrap_lad(
            normalized=False, error_bars=True, color='blue')
        ax2 = ax.twinx()
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim([ymin / ymax, ymax / ymax])
        lad_lr_trunc.plot_extrap_lev(error_bars=False, color='green')
        plt.tight_layout()


class TestUtils(unittest.TestCase):
    ''' For utility functions'''
    @unittest.skip("Skipping this test")
    def test_getRequests(self):
        '''Requires Earth Engine authentication.'''
        # Define the latitude and longitude ranges
        lat_range = [30, 40]
        lon_range = [-120, -110]

        # Call the getRequests function to generate the list of requests
        requests = getRequests(lat_range, lon_range)

        # Check the shape of the returned array
        self.assertEquals(requests.shape, (400, 2))

        # Check the values of the first and last few requests
        self.assertListEqual(requests[0].tolist(), [30., -120.])
        self.assertListEqual(requests[-1].tolist(), [39.5, -110.5])
