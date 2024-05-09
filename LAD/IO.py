import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from LAD.LAD import LAD

def loadBAWLD_CH4():
    '''units are in mg CH4/m2/day'''  # TODO: add function args
    ## Load
    df = pd.read_csv('/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD-CH4/data/ek_out/BAWLD_CH4_Aquatic_ERA5.csv',
                     encoding="ISO-8859-1", dtype={'CH4.E.FLUX ': 'float'}, na_values='-')
    len0 = len(df)
    temperature_metric = 'ERA5_stl1'
    eb_scaling = 0.580
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


def loadR21_CH4(pth='/Volumes/thebe/Other/Rosentreter2021/edk_out/Rosentreter_Aquatic_Ecosystems_lakes_temps.csv', flux_var='fch4_mgCH4m2d', surface_area_var='surface_area_km2', temperature_metric='ERA5_stl1'):
    '''
    Loads pre-formatted Rosentreter et al. 2021 emissions dataset and computes a regression model
    
    units are in mg CH4/m2/day
    '''

    ## Load
    df = pd.read_csv(pth,  # encoding="ISO-8859-1",
                     #  dtype={'CH4.E.FLUX ': 'float'},
                     #    na_values='-'
                     )
    ## Linear models (regression)
    # 'Seasonal.Diff.Flux' 'CH4.D.FLUX'
    formula = f"np.log10(Q('{flux_var}')+0.01) ~ np.log10({surface_area_var}) + {temperature_metric}"
    model = ols(formula=formula, data=df).fit()

    return model


def load_HR_ABZ():
    '''
    loadHR loads high resolution reference lake datasets from the Arctic-boreal Zone (ABZ), with input paths defined in this function.

    Returns
    -------
    lad_ref (LAD.LAD)
        LAD of combined datasets
    '''
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
    lad_ref = LAD.concat((lad_cir, lad_perl, lad_mullen),
                         broadcast_name=True, ignore_index=True)
    return lad_ref
