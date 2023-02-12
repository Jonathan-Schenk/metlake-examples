import os
import warnings
from datetime import datetime, timedelta
import itertools
from inspect import signature
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
import anemometer_data
from metlake_utils import Henry_Kh, calculate_LW_radiation
import SMHI_data


def import_chambers_data(path_data_file=None, drop_first_row=True, out='dict'):
    """
    Import chambers data from the METLAKE Flux chambers Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing data.
    drop_first_row : bool, default: True
        Condition for dropping the first row of the table, which contains
        a test case (Testsjön).
    out : {'dict', 'df'}, default: 'dict'
        'dict' : return a dictionary where each item corresponds to one sheet
        'df' : return a pandas.DataFrame combining data from all sheets
    """

    if path_data_file is None:
        path_data_file = os.path.join(
            '..', 'Data',
            'METLAKE_ManualFluxChambers_DBsheet_2018-2020_final.xlsx'
        )

    sheets = {
        'raw_data_in': 'RAW Field info & GC data',
        'mesan': 'MESAN', 'gap_fill': 'Gap-filling info',
        'data_in': 'GAP-FILLED Field info & GC data',
        'w_conc': 'CH4, DIC, N2O aq conc', 'pco2': 'pCO2 & FCO2 from pCO2',
        'ch4_flux': 'CH4 Flux', 'n2o_flux': 'FN2O from conc & k',
        'co2_flux': 'CO2 sensor data'
    }
    data = {}

    print('Opening Excel file...')
    f = pd.ExcelFile(path_data_file)
    print('The file contains the following sheets:')
    for sheet in f.sheet_names:
        print(f'  {sheet}')
    for k, sheet in sheets.items():
        print(f'Reading "{sheet}" sheet...')
        if sheet == 'MESAN':
            d = f.parse(sheet_name=sheet)
            d.columns = pd.MultiIndex.from_product([d.columns, [''], ['']])
            d.index += 1
            data[k] = d
        else:
            data[k] = f.parse(sheet_name=sheet, header=[5, 7, 8])
        empty_cols = [
            col for col in data[k].columns if all(pd.isna(data[k][col]))
        ]
        print(f'   Dropping {len(empty_cols)} empty columns.')
        data[k].drop(empty_cols, axis=1, inplace=True)
        col_ref = data[k].columns[-1]
        flag = 0
        for col in empty_cols:
            cond = col[0] != col_ref[0] and col[1] != col_ref[1] and \
                    col[2].split('.')[0] != col_ref[2]
            if cond:
                print(f'   One dropped empty column was "{col}".')
                flag += 1
        if flag == 0 and len(empty_cols) > flag:
            print('   All dropped empty columns were at the end of the table.')
        elif flag > 0 and len(empty_cols) > flag:
            print((
                '   All other dropped empty columns were at the end of the '
                'table.'
            ))
        if drop_first_row and sheet != 'MESAN':
            lake = data[k].loc[0, ('General', 'Lake', 'Unnamed: 0_level_2')]
            print(f'   Dropping the first row of the table (lake: {lake}).')
            data[k].drop(0, inplace=True)

    if out == 'dict':
        return data
    elif out == 'df':
        return pd.concat(data, axis=1)


def import_depth_profiles_data(path_data_file=None):
    """
    Import depth profiles data from the METLAKE Depth profiles Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing depth profiles data.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_DepthProfiles_2018-2020.xlsx'

    data = pd.read_excel(path_data_file, header=[0, 1, 2])

    return data


def import_depth_profiles_info(path_data_file=None):
    """
    Import depth profiles information from the METLAKE Depth profiles metadata
    Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing depth profiles
        information.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_DepthProfilesMetadata_2018-2020.xlsx'

    data = pd.read_excel(path_data_file, sheet_name=None)

    return data


def import_water_chemistry_data(path_data_file=None, mode='raw'):
    """
    Import water chemistry data from the METLAKE Water chemistry Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing water chemistry
        data.
    mode : {'raw', 'cleaned'}
        If mode='raw', import data tables from the Excel files without
        modification.
        If mode='cleaned', import data tables from the Excel files and
        preprocess them by averaging duplicate values and removing outliers.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_WaterChemistry_2018-2020.xlsx'

    conv_lakes_CN = {
        'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6',
        'Venasjon': 'VEN', 'Parsen': 'PAR', 'SodraTeden': 'SOD',
        'StoraGalten': 'SGA', 'Gundlebosjon': 'GUN', 'Grinnsjon': 'GRI',
        'LjusvattentjarnExperiment': 'LJE', 'LjusvattentjarnReference': 'LJR',
        'Nastjarn': 'NAS', 'NedreBjorntjarn': 'NBJ',
        'Dammsjon': 'DAM', 'Norrtjarn': 'NOR', 'Grastjarn': 'GRA',
        'Lammen': 'LAM', 'Klintsjon': 'KLI', 'Gyslattasjon': 'GYS'
    }
    conv_lakes_P = {
        'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6',
        'Venasjön': 'VEN', 'Parsen': 'PAR', 'SödraTeden': 'SOD',
        'StoraGalten': 'SGA', 'Gundlebosjön': 'GUN', 'Grinnsjön': 'GRI',
        'LjusvattentjärnExperiment': 'LJE', 'LjusvattentjärnReference': 'LJR',
        'Nästjärn': 'NAS', 'NedreBjörntjärn': 'NBJ',
        'Dammsjön': 'DAM', 'Norrtjärn': 'NOR', 'Grästjärn': 'GRA',
        'Lammen': 'LAM', 'Klintsjön': 'KLI', 'Gyslättasjön': 'GYS'
    }
    conv_lakes_chla = {
        'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6',
        'Venasjön': 'VEN', 'Parsen': 'PAR', 'Södra Teden': 'SOD',
        'Stora Galten': 'SGA', 'Gundlebosjön': 'GUN', 'Grinnsjön': 'GRI',
        'Ljusvattentjärn Experiment': 'LJE', 'Ljusvattentjärn Reference': 'LJR',
        'Nästjärn': 'NAS', 'Nedre Björntjärn': 'NBJ',
        'Dammsjön': 'DAM', 'Norrtjärn': 'NOR', 'Grästjärn': 'GRA',
        'Lammen': 'LAM', 'Klintsjön': 'KLI', 'Gyslättasjön': 'GYS'
    }
    conv_lakes = [conv_lakes_CN, conv_lakes_P, conv_lakes_chla]

    data_file = pd.ExcelFile(path_data_file)
    columns_C_and_N = [
        'Anal.', 'Reference_run', 'Sample_type', 'Lake', 'Date_sampling',
        'Depth_m', 'Depth_category', 'Filtered', 'Dilution', 'Remarks',
        'TC_final_mg/L', 'IC_final_mg/L', 'TOC_final_mg/L', 'TN_final_mg/L'
    ]
    data_CN = data_file.parse('TC_IC_TOC_TN')[columns_C_and_N]
    data_P = data_file.parse('TP')
    data_Chla = data_file.parse('ChlorophyllA')
    data_modif = []
    for n, data in enumerate([data_CN, data_P, data_Chla]):
        cond = data['Lake'].apply(lambda x: x in conv_lakes[n].keys())
        data = data[cond].copy()
        data['Lake'].replace(conv_lakes[n], inplace=True)
        data_modif.append(data)
    # There are some NaN values in the column 'Filtered' of the CN table.
    # These values correspond to samples from the pond near GUN, taken from
    # the shore in SOD, PAR, VEN during lake selection and taken during
    # lake selection in SGA, GUN and GRI. These samples were most likely
    # not filtered.
    data_modif[0].loc[np.isnan(data_modif[0]['Filtered']), 'Filtered'] = 0
    data_modif[0] = data_modif[0].astype({ 'Filtered': 'bool'})

    if mode == 'raw':
        return data_modif
    elif mode == 'cleaned':
        infos = [
            (0, 'TOC_final_mg/L', 'TOC_mg/L',
             data_modif[0].loc[np.logical_and(np.logical_and(
                 data_modif[0]['Anal.'].apply(lambda x: x in ['TC', 'TOC']),
                 data_modif[0]['Reference_run']),
                 np.logical_not(data_modif[0]['Filtered'])
             )]
            ),
            (0, 'TOC_final_mg/L', 'DOC_mg/L',
             data_modif[0].loc[np.logical_and(np.logical_and(
                 data_modif[0]['Anal.'].apply(lambda x: x in ['TC', 'TOC']),
                 data_modif[0]['Reference_run']),
                 data_modif[0]['Filtered']
             )]
            ),
            (0, 'TN_final_mg/L', 'TN_mg/L',
             data_modif[0].loc[np.logical_and(np.logical_and(
                 data_modif[0]['Anal.'].apply(lambda x: x == 'TN'),
                 data_modif[0]['Reference_run']),
                 np.logical_not(data_modif[0]['Filtered'])
             )]
            ),
            (0, 'TN_final_mg/L', 'DN_mg/L',
             data_modif[0].loc[np.logical_and(np.logical_and(
                 data_modif[0]['Anal.'].apply(lambda x: x == 'TN'),
                 data_modif[0]['Reference_run']),
                 data_modif[0]['Filtered']
             )]
            ),
            (1, 'TP_ug/L', 'TP_ug/L', data_modif[1]),
            (2, 'Chlorophyll A concentration [ug/L]', 'Chla_ug/L',
             data_modif[2]
            )
        ]
        res = None
        col_lake = 'Lake'
        for n, col_val, col_res, data in infos:
            if n == 0 or n == 1:
                col_date = 'Date_sampling'
                col_depth = 'Depth_category'
            elif n == 2:
                col_date = 'Date sampling'
                col_depth = 'Depth category'
            series = []
            for lake, d_lake in data.groupby(col_lake):
                for date, d_date in d_lake.groupby(col_date):
                    for depth, d_depth in d_date.groupby(col_depth):
                        d_g = d_depth[col_val].dropna()
                        if d_g.shape[0] == 0:
                            val = np.nan
                        elif d_g.shape[0] == 1:
                            val = d_g.iloc[0]
                        elif d_g.shape[0] == 2:
                            diff_val = np.abs(d_g.diff().iloc[1])
                            if diff_val < d_g.mean()/3:
                                val = d_g.mean()
                            else:
                                dates = d_lake.loc[
                                    d_lake[col_depth] == depth, col_date
                                ]
                                dates = list(sorted(set(dates)))
                                ind_date = dates.index(date)
                                date_min = dates[max(0, ind_date - 2)]
                                date_max = dates[min(len(dates) - 1,
                                                     ind_date + 2)]
                                vals_ref = d_lake.loc[
                                    np.logical_and(np.logical_and(
                                        d_lake[col_date] >= date_min,
                                        d_lake[col_date] <= date_max),
                                        d_lake[col_depth] == depth
                                    ), col_val
                                ]
                                ind = np.argmin(np.abs(d_g - vals_ref.mean()))
                                val = d_g.iloc[ind]
                        else:
                            subvals = []
                            for ind, v in d_g.iteritems():
                                other_vals = d_g.drop(ind)
                                diff_val = np.abs(v - other_vals.mean())
                                if diff_val < other_vals.mean()/3:
                                    subvals.append(v)
                            if len(subvals) > 0:
                                val = np.mean(subvals)
                            else:
                                val = np.nan
                        series.append(pd.Series({
                            'Lake': lake, 'Date_sampling': date,
                            'Depth_category': depth, col_res: val
                        }))
            data_clean = pd.concat(series, axis=1).T.\
                    set_index(['Lake', 'Date_sampling', 'Depth_category']).\
                    astype(np.float64)
            if res is None:
                res = data_clean
            else:
                res = res.merge(
                    data_clean, 'outer', left_index=True, right_index=True
                )
        res = res.sort_values(['Lake', 'Date_sampling', 'Depth_category'],
                              ascending=[True, True, False])
        return res


def import_water_chemistry_data_cleaned_manually(path_data_file=None):
    """
    Import cleaned water chemistry data from all lakes and all depths and
    calculate yearly average values.

    Data includes total and dissolved organic carbon, total and dissolved
    nitrogen, total phosphorus and chlorophyll a concentrations.

    The manual cleaning step was achieved by first importing data using
    the function 'import_water_chemistry_data' when using mode='cleaned'.
    Data were then cleaned further by looking at the figures obtained
    using the function 'plot_timeseries_water_chemistry', for example
    by adjusting which values were used to calculate daily average values
    when more than one daily value were available. The mask table was
    created in the process to filter out some values before calculating
    yearly average values. The manually cleaned data has also been corrected
    for double dates compared to the table returned by the function
    'import_water_chemistry_data' using mode='cleaned'. Depths (in meters)
    at which samples were taken were also added in the cleaned data when
    available. In addition, total phosphorus concentration values were
    replaced with values from trace elements analysis for the lakes sampled
    in 2020 in the cleaned data.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing manually cleaned
        water chemistry data.
    """

    if path_data_file is None:
        path_data_file = os.path.join(
            '..', 'Data',
            'METLAKE_WaterChemistryWithSummary_2018-2020.xlsx'
        )

    water_chem_clean = pd.read_excel(
        path_data_file, sheet_name='SummaryFromPython', index_col=[0, 1, 2]
    )
    water_chem_mask = pd.read_excel(
        path_data_file, sheet_name='ValuesUsedForYearlyAverage',
        index_col=[0, 1, 2]
    )
    water_chem_clean[water_chem_mask == 0] = np.nan
    d_avg = []
    for lake, d_lake in water_chem_clean.groupby(level='Lake'):
        if lake in ['BD3', 'BD4', 'BD6']:
            summer_start = datetime(2018, 7, 1)
            summer_end = datetime(2018, 9, 1)
        elif lake in ['PAR', 'VEN', 'SOD']:
            summer_start = datetime(2018, 6, 1)
            summer_end = datetime(2018, 10, 1)
        elif lake in ['SGA', 'GUN', 'GRI']:
            summer_start = datetime(2019, 6, 1)
            summer_end = datetime(2019, 10, 1)
        elif lake in ['LJE', 'LJR', 'NAS', 'NBJ']:
            summer_start = datetime(2019, 7, 1)
            summer_end = datetime(2019, 9, 1)
        elif lake in ['NOR', 'GRA', 'DAM']:
            summer_start = datetime(2020, 6, 15)
            summer_end = datetime(2020, 9, 15)
        elif lake in ['KLI', 'LAM', 'GYS']:
            summer_start = datetime(2020, 6, 1)
            summer_end = datetime(2020, 10, 1)
        d_summer = d_lake.loc[
            (lake, slice(summer_start, summer_end), slice(None)), 'Chla_ug/L'
        ]
        for depth in ['surface', 'middle', 'bottom']:
            if depth not in d_lake.index.get_level_values(2):
                continue
            d_year = d_lake.loc[(lake, slice(None), depth)]
            if depth in d_summer.index.get_level_values(2):
                Chla_summer = d_summer.loc[(lake, slice(None), depth)].mean()
            else:
                Chla_summer = np.nan
            d = d_year.mean()
            d['Chla_summer_ug/L'] = Chla_summer
            d_avg.append(pd.Series(d, name=(lake, depth)))
    water_chem_avg = pd.concat(d_avg, axis=1).T

    return water_chem_clean, water_chem_avg


def import_trace_elements_data(path_data_file=None, convert_bdl_to_nan=True):
    """
    Import trace elements data from the METLAKE Trace Elements Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing trace elements data.
    convert_bdl_to_nan : bool, default: True
        If True, convert 'below detection limit' values such as '<0.005'
        to numpy.nan values.
        If False, convert 'below detection limit' values such as '<0.005'
        to float number values without the '<' symbol.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_TraceElements.xlsx'

    data = []
    surface_labels = ['surface', '1,0m', 0.1, 1, 'surf']
    for m in ['June', 'Nov']:
        df = pd.read_excel(path_data_file, sheet_name=f'Analyses {m} 2021')
        df.columns = range(df.shape[1])
        df.iloc[8, df.iloc[8].apply(lambda x: x in surface_labels)] = 'surface'
        d = df.iloc[12:-4, 2:]
        ind = df.iloc[12:-4, 0:2].apply(
            lambda row: f'{row.iloc[0]}, {row.iloc[1]}', axis=1
        )
        cols = pd.MultiIndex.from_frame(
            df.iloc[6:9, 2:].T,
            names=['Lake', 'Date_sampling', 'Depth_category']
        )
        data.append(pd.DataFrame(data=d.values, index=ind.values, columns=cols))
    data = pd.concat(data, axis=1)

    if convert_bdl_to_nan:
        f = lambda x: np.nan if isinstance(x, str) else x
    else:
        f = lambda x: float(x[1:]) if isinstance(x, str) else x
    data = data.applymap(f)

    return data


def import_absorbance_data(path_data_file=None, interpolate=True):
    """
    Import absorbance data from the METLAKE Aqualog Absorbance CSV file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the CSV file containing absorbance data.
    interpolate : bool, default: True
        If True, interpolate the measurements at a 1 nm wavelength resolution.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_AqualogAbsorbance_2018-2020.csv'

    data = pd.read_csv(path_data_file, header=[1, 2, 3], index_col=0)
    data = data.T.sort_index()
    if interpolate:
        data = data.interpolate(axis=1)
    data_avg = data.groupby(level=0).mean()

    return data, data_avg


def import_info_lakes(
    path_data_file=None, str_to_nan=False, only_bay_BD6=True,
    add_area_2008=False
):
    """
    Import lakes information and bathymetry from the METLAKE Info lakes file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing information about
        lakes.
    str_to_nan : bool
        If True, convert "<1%" values to NaN in columns with soil type
        and land use information in the table with lake information.
    only_bay_BD6 : bool
        If True, replace the area data for BD6 by the area derived only
        from the eastern basin of the lake (where sampling took place).
    add_area_2008 : bool
        If True, add the bathymetry of the three lakes near Uppsala which
        were sampled in 2008 to the table containing the bathymetry of lakes.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_InfoLakes.xlsx'

    data_file = pd.ExcelFile(path_data_file)
    info_lakes = data_file.parse('lake_catchment_info', header=[0])
    info_lakes.replace({'na': np.nan}, inplace=True)
    info_lakes.dropna(axis=1, how='all', inplace=True)
    area = data_file.parse('lake_2d_area', header=[0, 1], index_col=0)
    area.dropna(axis=1, how='all', inplace=True)
    volume = data_file.parse('lake_3d_volume', header=[0, 1], index_col=0)
    volume.dropna(axis=1, how='all', inplace=True)

    if str_to_nan:
        s = slice('Artificial fill', 'Wacke')
        func = lambda x: isinstance(x, str)
        cond = info_lakes.loc[:, s].applymap(func)
        info_lakes[cond] = np.nan
        info_lakes.loc[:, s] = info_lakes.loc[:, s].astype('float64')

    if only_bay_BD6:
        area_new = pd.read_csv(
            '../Data/Bathymetry_BD6_EasternBay.csv', index_col=0, header=[0, 1]
        )
        area['BD6'] = area_new['BD6']

    if add_area_2008:
        area_tot = {'LA': 989600.0, 'LO': 575000.0, 'ST': 1230300.0}
        bathy = {
            'LA': {0.0: 1.0, 1.0: 0.737, 2.0: 0.432, 3.0: 0.179, 4.0: 0.021,
                   4.5: 0.0},
            'LO': {0.0: 1.0, 3.0: 0.897, 4.0: 0.793, 5.0: 0.693, 6.0: 0.557,
                   7.0: 0.316, 8.0: 0.171, 9.0: 0.111, 10.0: 0.035, 11.0: 0.003,
                   11.2: 0.0},
            'ST': {0.0: 1.0, 1.0: 0.816, 2.0: 0.490, 3.0: 0.003, 4.0: 0.0}
        }
        for lake in ['LA', 'LO', 'ST']:
            b = pd.Series(bathy[lake])
            f_area = interp1d(b.index, b.values, bounds_error=False)
            area[(lake, 'Area_[m2]')] = area_tot[lake]*f_area(area.index)

    return info_lakes, area, volume


def import_gmx531_data(path_folder=None, lake='all'):
    """
    Import weather data from a GMX531 weather station CSV file.

    Below are some notes about daylight saving time based on inspection of raw
    (1 second resolution) data. Among other checks, logger datetime and weather
    station datetime were compared.
    - The weather station in BD6 was started with GMT+02:00 and used this time
      zone until the end of the deployment.
    - The weather station in SGA was started with GMT+01:00 and used this time
      zone until 2019-03-31 01:59:59 (GMT+01:00). After that, it switched
      automatically to GMT+02:00, so it skipped one hour between 2019-03-31
      02:00:00 (GMT+01:00) and 2019-03-31 03:00:00 (GMT+01:00). It continued
      using GMT+02:00 until 2019-10-15 01:39:41 (GMT+02:00), when it stopped.
      When it was restarted on 2019-11-04 09:31:00 (GMT+01:00), the time was set
      to GMT+01:00.
    - The weather station in NBJ was started with GMT+02:00 and used this time
      zone until 2019-10-27 02:59:59 (GMT+02:00). After that, it switched
      automatically to GMT+01:00, so it recorded another hour of data with
      similar timestamps. It means that data between 2019-10-27 02:00:00
      (GMT+02:00) and 2019-10-27 03:00:00 (GMT+02:00) have been aggregated with
      data between 2019-10-27 03:00:00 (GMT+02:00) and 2019-10-27 04:00:00
      (GMT+02:00) when resampling values at a 1-minute frequency.
    - The weather station in NOR was started with GMT+02:00 and used this time
      zone until 2020-09-29 17:08:51 (GMT+02:00) when it stopped. When it was
      restarted on 2020-11-02 14:59:37 (GMT+01:00), the time was set to
      GMT+01:00.
    - The weather station in GYS was started with GMT+02:00 and used this time
      zone until 2020-10-21 14:14:13 (GMT+02:00) when it stopped. When it was
      restarted on 2020-11-16 12:21:05 (GMT+01:00), the time was set to
      GMT+01:00.

    Parameters
    ----------
    path_folder : str or None
        Full or relative path to the folder containing GMX531 weather stations
        CSV files.
    lake : {'BD6', 'SGA', 'NBJ', 'NOR', 'GYS', 'all'}
        Lake where the weather station was deployed. If lake = 'all', import all
        files.
    """

    if path_folder is None:
        path_folder = '../Data/GMX531/'

    noDST = [
        (datetime(2017, 10, 29, 3, 0), datetime(2018, 3, 25, 2, 0)),
        (datetime(2018, 10, 28, 3, 0), datetime(2019, 3, 31, 2, 0)),
        (datetime(2019, 10, 27, 3, 0), datetime(2020, 3, 29, 2, 0)),
        (datetime(2020, 10, 25, 3, 0), datetime(2021, 3, 28, 2, 0))
    ]

    data = {}
    for f in os.listdir(path_folder):
        l = f.split('_')[2]
        if lake != 'all' and l != lake:
            continue
        # Load data file and convert column 'Datetime' to datetime type
        d = pd.read_csv(os.path.join(path_folder, f))
        d['Datetime'] = pd.to_datetime(d['Datetime'])
        # Change time to GMT+02:00 when it is GMT+01:00. Add/remove rows
        # to compensate for gaps/overlaps in datetime due to the time conversion
        # from GMT+01:00 to GMT+02:00.
        for nodst in noDST:
            cond = np.logical_and(
                d['Datetime'] >= nodst[0], d['Datetime'] <= nodst[1]
            )
            d.loc[cond, 'Datetime'] = \
                    d.loc[cond, 'Datetime'] + timedelta(hours=1)
        if l == 'SGA':
            d.drop(range(14832, 14892), inplace=True)
            missing_hour = pd.DataFrame(
                index=[-1]*60, columns=d.columns, data=np.nan
            )
            missing_hour['Datetime'] = pd.DatetimeIndex(
                [datetime(2019, 10, 27, 3, 0) + timedelta(minutes=n)
                 for n in range(60)]
            )
            d = pd.concat([d, missing_hour])
        elif l == 'NBJ':
            missing_hour = pd.DataFrame(
                index=[-1]*60, columns=d.columns[1:],
                data=np.tile(
                    d.loc[208106:208107, d.columns[1:]].mean().values, (60, 1)
                )
            )
            missing_hour['Datetime'] = pd.DatetimeIndex(
                [datetime(2019, 10, 27, 3, 0) + timedelta(minutes=n)
                 for n in range(60)]
            )
            d = pd.concat([d, missing_hour])
        elif l == 'NOR':
            missing_hour = pd.DataFrame(
                index=[-1]*60, columns=d.columns, data=np.nan
            )
            missing_hour['Datetime'] = pd.DatetimeIndex(
                [datetime(2020, 10, 25, 3, 0) + timedelta(minutes=n)
                 for n in range(60)]
            )
            d = pd.concat([d, missing_hour])
        elif l == 'GYS':
            missing_hour = pd.DataFrame(
                index=[-1]*60, columns=d.columns, data=np.nan
            )
            missing_hour['Datetime'] = pd.DatetimeIndex(
                [datetime(2020, 10, 25, 3, 0) + timedelta(minutes=n)
                 for n in range(60)]
            )
            d = pd.concat([d, missing_hour])
        # Sort table according to date and time
        d = d.sort_values('Datetime')
        data[l] = d

    if len(data) == 1:
        data = data[0]

    return data


def import_mesan_data(path_data_file=None, lake='all'):
    """
    Import MESAN data from the METLAKE MESAN Excel file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the Excel file containing MESAN data.
    lake : {'all', 'Venasjon', 'Parsen', 'SodraTeden', 'BD03', 'BD04', 'BD06',
            'StoraGalten', 'Gundlebosjon', 'Grinnsjon', 'LjusvattentjarnExp',
            'LjusvattentjarnRef', 'Nastjarn', 'NedreBjorntjarn', 'Norrtjarn',
            'Grastjarn', 'Dammsjon', 'Klintsjon', 'Gyslattasjon', 'Lammen'}
        Name of the lake of interest. If lake = 'all', import data from all
        lakes into a dictionary.
    """

    if path_data_file is None:
        path_data_file = '../Data/METLAKE_2018-2020_MESAN.xlsx'

    if lake == 'all':
        data = pd.read_excel(path_data_file, sheet_name=None, index_col=0)
    else:
        data = pd.read_excel(path_data_file, sheet_name=lake, index_col=0)

    return data


def import_and_combine_mesan_and_strang_data(
    lake, path_data_file_mesan=None, path_data_files_strang=None
):
    """
    Import and combine MESAN data from the METLAKE MESAN Excel file and
    STRANG data from a CSV file.

    Parameters
    ----------
    lake : {'VEN', 'PAR', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GUN', 'GRI, 'LJE',
            'LJR', 'NAS', 'NBJ', 'NOR', 'GRA', 'DAM', 'KLI', 'GYS', 'LAM'}
        Name of the lake of interest.
    path_data_file_mesan : str or None
        Full or relative path to the Excel file containing MESAN data.
    path_data_files_strang : str or None
        Full or relative path to the folder where STRANG CSV files are located.
    """

    if path_data_file_mesan is None:
        path_data_file_mesan = '../Data/METLAKE_2018-2020_MESAN.xlsx'

    if path_data_files_strang is None:
        path_data_files_strang = '../Data/STRANG'

    conv_lake = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExp', 'LJR': 'LjusvattentjarnRef',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }

    mesan = import_mesan_data(lake=conv_lake[lake])
    lw_clear, lw_tot = calculate_LW_radiation(
        mesan['Temperature'], mesan['Relative humidity'],
        mesan['tcc']*mesan['c_sigfr']*100, mesan['cb_sig_b']
    )
    mesan['lw_radiation'] = lw_tot
    mesan.index.name = 'datetime'
    strang = pd.read_csv(os.path.join(
        path_data_files_strang, f'STRANG_{lake}.csv'
    ))
    strang['datetime'] = pd.to_datetime(strang['datetime'])
    strang = strang[['datetime', 'value']].set_index('datetime')
    strang = strang.resample('H').mean()
    for ind, row in strang.iterrows():
        if np.isnan(row['value']):
            cond = strang.index == (ind - timedelta(days=1))
            strang.loc[ind, 'value'] = strang.loc[cond, 'value'].squeeze()
    strang.rename(columns={'value': 'solar_radiation'}, inplace=True)
    data = pd.merge(mesan, strang, on='datetime', how='outer')

    return data


def import_thermistor_chain_data(path_data_file, mode='raw', year=2000):
    """
    Import water temperature data from a thermistor chain MAT file.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the MAT file containing thermistor chain data.
    mode : {'raw', 'dataframe'}, default: 'raw'
        Type of output.
        - 'raw': Separate numpy.ndarray for temperature, time, depth and info.
        - 'dataframe': pandas.DataFrame with time in index and depth in columns.
    year : int, default: 2000
        Year when measurement starts.
    """

    data = loadmat(path_data_file)
    T = data['T']
    time = data['time']
    depth = data['depth'][0]
    readme = data['readme']

    if mode == 'raw':
        return T, time, depth, readme
    elif mode == 'dataframe':
        time = pd.DatetimeIndex(
            datetime(year, 1, 1) + \
            np.vectorize(timedelta)(hours=24*(time-1)-2).reshape(-1)
        )
        time = time.map(lambda x: x.replace(microsecond=0))
        data = pd.DataFrame(data=T, columns=depth, index=time)
        return data


def import_minidot_data(path_data_file):
    """
    Import oxygen concentration data from a miniDOT TXT file.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the TXT file containing miniDOT data.
    """

    data = pd.read_csv(path_data_file, header=[5, 6], skipinitialspace=True)
    data.iloc[:, 1] = pd.to_datetime(data.iloc[:, 1])
    data.iloc[:, 2] = pd.to_datetime(data.iloc[:, 2])

    return data


def import_hobo_data(path_data_file):
    """
    Generic function to import HOBO CSV data files.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the CSV file containing HOBO data.
    """

    data = pd.read_csv(path_data_file, header=1)
    col_dt = data.columns[1]
    data[col_dt] = pd.to_datetime(data[col_dt])
    if col_dt != 'Date Time, GMT+02:00':
        tz = int(col_dt[-4])
        data['Date Time, GMT+02:00'] = data[col_dt] + timedelta(hours=2-tz)

    return data


def import_hobo_weather_station_data(path_data_file=None):
    """
    Import weather data from a HOBO weather station CSV file.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the CSV file containing HOBO weather station
        data.
        If 'LJT' or 'PAR' is passed, a default path to files for Ljusvattentjarn
        and Parsen, respectively, will be used.
    """

    if path_data_file == 'LJT':
        path_data_file = os.path.join(
            '..', 'Data', 'HOBOWeatherStation',
            'HOBOWeatherStation_LJT_20190524-20191017.csv'
        )
    elif path_data_file == 'PAR':
        path_data_file = os.path.join(
            '..', 'Data', 'HOBOWeatherStation',
            'HOBOWeatherStation_PAR_20180505-20181207_9855141.csv'
        )

    data = import_hobo_data(path_data_file)

    return data


def import_sediment_thermistor_data(path_data_file):
    """
    Import shallow sediment water temperature data from a HOBO thermistor CSV
    file.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the CSV file containing shallow sediment water
        temperature data.
    """

    data = import_hobo_data(path_data_file)

    return data


def import_water_level_data(path_data_file=None):
    """
    Import hydrostatic pressure data from a HOBO water level logger CSV file.

    Parameters
    ----------
    path_data_file : str
        Full or relative path to the CSV file containing hydrostatic pressure
        data.
    """

    if path_data_file is None:
        path_data_file = os.path.join(
            '..', 'Data', 'WaterLevel',
            'HOBOWaterLevel_GUN_20190521-20200418_10329402.csv'
        )

    data = import_hobo_data(path_data_file)

    return data


def import_anemometer_data(path_data_file):
    """
    Import wind data from a HOBO anemometer CSV file.

    Parameters
    ----------
    path_data_file : str or None
        Full or relative path to the CSV file containing HOBO weather station
        data.
    """

    data = import_hobo_data(path_data_file)

    return data


def import_merged_anemometer_data(path_folder=None):
    """
    Import wind data from all HOBO anemometer CSV files.

    Parameters
    ----------
    path_folder : str or None
        Full or relative path to the folder containing anemometer CSV files.
    """

    if path_folder is None:
        path_folder = '../Data/Anemometers/'

    data, data_cor = anemometer_data.merge_data(path_folder)

    return data_cor


def format_ch4_flux_columns(chambers):
    """
    Change column names in CH4 flux data table.

    Parameters
    ----------
    chambers : dict
        Dictionary of data table returned by the function
        'import_chambers_data'.
    """

    fc = chambers['ch4_flux'].copy()
    fc[('General', 'Lake', 'Unnamed: 0_level_2')].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    fc.columns = [
        'lake', 'chamber', 'dt_ini', 'dt_fin', 'depth_ini', 'comment_ini',
        'lat_ini', 'lon_ini', 'depth_fin', 'comment_fin', 'lat_fin',
        'lon_fin', 'T_air_ini', 'T_wat_ini', 'P_ini', 'sal_ini', 'T_air_fin',
        'T_wat_fin', 'P_fin', 'sal_fin', 'R', 'P_conv', 'T_conv', 'CH4_air_ini',
        'CH4_air_fin', 'CH4_chamber', 'pCH4_air_ini', 'pCH4_air_fin',
        'pCH4_chamber_P_fin', 'pCH4_chamber_P_ini', 'pCH4_chamber_P_ini_T_corr',
        'T_wat_ini_K', 'KH_ini', 'T_wat_fin_K', 'KH_fin', 'CH4aq_ini',
        'CH4aq_fin', 'pCH4aq_ini', 'pCH4aq_fin', 'area', 'volume', 'n_CH4_ini',
        'n_CH4_fin', 'dt', 'dCH4', 'T_wat_mean', 'T_wat_mean_K', 'KH_mean',
        'pCH4aq_mean', 'pCH4_air_mean', 'K', 'k', 'Sc', 'k600', 'ebul_flag',
        'CH4_flux_nonlinear', 'CH4_flux_linear', 'k600_filt1', 'k600_median',
        'k600_median_thld_fraction', 'k600_too_low', 'k600_filt2', 'k600_min',
        'ebul_thld', 'k_diff', 'k_diff_avg', 'k_diff_gapfilled',
        'CH4_diff_flux', 'CH4_ebul_flux', 'CH4_tot_flux', 'unknown'
    ]

    return fc


def create_deployment_groups_indices(
    data, col_lake=None, col_dt_ini=None, col_dt_fin=None, col_new=None,
    dt_delta=1
):
    """
    Add a column containing distinct indices for each deployment group.

    A deployment group is a group of chambers deployed in the same lake and
    over the same period.

    Parameters
    ----------
    data : pandas.DataFrame
        Data table containing the merged field notes and GC analysis results.
        This function will modify it in-place by sorting it according to lakes
        and chambers deployment dates and times, and adding a new column
        indicating the ID of each deployment group.
    col_lake : str or None, default: None
        Column in table 'data' containing lakes IDs.
    col_dt_ini : str or None, default: None
        Column in table 'data' containing chambers deployment dates and times.
    col_dt_fin : str or None, default: None
        Column in table 'data' containing chambers sampling dates and times.
    col_new : str or None, default: None
        Name of the new column to add and that contains the deployment groups
        IDs.
    dt_delta : float, default: 1
        Time gap used to distinguish different deployment groups (in hour).
        One hour works well to distinguish individual deployment groups.
        Seven days work well to distinguish individual sampling 'weeks'.
    """

    if col_lake is None:
        col_lake = ('General', 'Lake', 'Unnamed: 0_level_2')
    if col_dt_ini is None:
        col_dt_ini = ('General', 'Initial sampling', 'Date and time')
    if col_dt_fin is None:
        col_dt_fin = ('General', 'Final sampling', 'Date and time')
    if col_new is None:
        col_new = ('General', 'Deployment group', '')

    data.sort_values([col_lake, col_dt_ini], ignore_index=True, inplace=True)

    n = 0
    group = []
    dt_ini_prev = datetime(2000, 1, 1)
    dt_fin_prev = datetime(2000, 1, 1)

    for ind, row in data.iterrows():
        dt_ini = row[col_dt_ini]
        dt_fin = row[col_dt_fin]
        if pd.isna(dt_ini):
            if not pd.isna(dt_ini_prev):
                data.loc[group, col_new] = n
            data.loc[ind, col_new] = -1
            group = []
            n += 1
        elif dt_ini - dt_ini_prev < timedelta(hours=0):
            data.loc[group, col_new] = n
            group = [ind]
            n += 1
        elif dt_ini - dt_ini_prev > timedelta(hours=dt_delta):
            if not pd.isna(dt_fin) and not pd.isna(dt_fin_prev):
                if abs(dt_fin - dt_fin_prev) > timedelta(hours=dt_delta):
                    data.loc[group, col_new] = n
                    group = [ind]
                    n += 1
                else:
                    group.append(ind)
            else:
                data.loc[group, col_new] = n
                group = [ind]
                n += 1
        else:
            group.append(ind)
        dt_ini_prev = dt_ini
        dt_fin_prev = dt_fin
    else:
        data.loc[group, col_new] = n


def combine_all_data_at_chamber_deployment_scale(
    chambers=None, anemo=None, gmx=None, hobo_PAR=None, hobo_LJT=None,
    water_chem=None, thermistors_data=None
):
    """
    Create a data table containing CH4 flux data and a range of other variables
    at the scale of individual chamber deployments.

    Parameters
    ----------
    chambers : dict or None, default: None
        Dictionary of data tables returned by the function
        'import_chambers_data'.
    anemo : dict or None, default: None
        Dictionary of data tables returned by the function
        'import_merged_anemometer_data'.
    gmx : dict or None, default: None
        Dictionary of data tables returned by the function
        'import_gmx531_data'.
    hobo_PAR : pandas.DataFrame or None, default: None
        Data table returned by the function 'import_hobo_weather_station_data'.
    hobo_LJT : pandas.DataFrame or None, default: None
        Data table returned by the function 'import_hobo_weather_station_data'.
    water_chem : pandas.DataFrame or None, default: None
        First of the two data tables returned by the function
        'import_water_chemistry_data_cleaned_manually'.
    thermistors_data : dict or None, default: None
        Dictionary of data tables. Individual data tables are returned by
        the function 'import_thermistor_chain_data'.
    """

    # PARAMETERS
    # Anemometer data
    params_anemo = {
        'WindSpeed': 'Wind Speed, m/s', 'GustSpeed': 'Gust Speed, m/s',
        'WindDirection': 'Wind Direction, ø'
    }
    funcs_anemo = {}
    #funcs_anemo = {'mean': np.mean}
    #funcs_anemo = {'mean': np.mean, 'median', np.nanmedian, 'std': np.nanstd,
    #               'min': np.min, 'max': np.max}

    # Weather station data
    params_weather_station = {
        'AirTemperature': 'oC', 'RelativeHumidity': '%',
        'Precipitation': 'mm/h', 'WindSpeed': 'm/s', 'GustSpeed': 'm/s',
        'WindDirection': 'deg', 'SolarRadiation': 'W/m2'
    }
    funcs_weather_station = {}
    #funcs_weather_station = {'mean': np.mean}
    #funcs_weather_station = {
    #    'mean': np.mean, 'median', np.nanmedian, 'std': np.nanstd,
    #    'min': np.min, 'max': np.max
    #}

    # Water chemistry data
    params_water_chemistry = []
    #params_water_chemistry = ['DOC', 'TOC', 'DN', 'TN', 'TP', 'Chla']
    depth_cats_water_chemistry = []
    #depth_cats_water_chemistry = ['surface', 'middle', 'bottom']

    # Thermistor chains temperature data
    folder_thermistors = '../Data/ThermistorChains'

    # FUNCTIONS
    def extract_anemo_data(row, func, param):
        lake = row['lake']
        if lake not in ['VEN', 'SOD', 'NOR', 'LJE']:
            d = anemo[lake]
        elif lake in ['VEN', 'SOD', 'NOR']:
            return np.nan
        elif lake == 'LJE':
            d = anemo['LJR']
        cond = np.logical_and(
                d[col_time] >= row['dt_ini'], d[col_time] <= row['dt_fin']
        )
        if sum(cond) == 0:
            return np.nan
        elif param == 'WindSpeed' or param == 'GustSpeed':
            return func(d.loc[cond, param])
        elif param == 'WindDirection':
            if func == np.mean:
                return 180/np.pi*np.arctan2(
                    func(np.cos(d[param]*np.pi/180)),
                    func(np.sin(d[param]*np.pi/180))
                )
            else:
                return np.nan

    def extract_weather_data(row, func, param):
        lake = row['lake']
        if lake in ['BD3', 'BD4', 'BD6']:
            d = gmx['BD6']
            d_time = d['Datetime']
        elif lake in ['PAR', 'VEN', 'SOD']:
            d = hobo_PAR
            d_time = d['Date Time, GMT+02:00']
        elif lake in ['SGA', 'GUN', 'GRI']:
            d = gmx['SGA']
            d_time = d['Datetime']
        elif lake in ['NAS', 'NBJ']:
            d = gmx['NBJ']
            d_time = d['Datetime']
        elif lake in ['LJE', 'LJR']:
            d = hobo_LJT
            d_time = d['Date Time, GMT+02:00']
        elif lake in ['NOR', 'GRA', 'DAM']:
            d = gmx['NOR']
            d_time = d['Datetime']
        elif lake in ['KLI', 'GYS', 'LAM']:
            d = gmx['GYS']
            d_time = d['Datetime']
        cond = np.logical_and(
            d_time >= row['dt_ini'], d_time <= row['dt_fin']
        )
        if sum(cond) == 0:
            return np.nan
        if param == 'AirTemperature':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                return func(d.loc[cond, 'Temp, °C'])
            else:
                return func(d.loc[cond, 'temperature'])
        elif param == 'RelativeHumidity':
            if lake in ['PAR', 'VEN', 'SOD']:
                return np.nan
            elif lake in ['LJE', 'LJR']:
                return func(d.loc[cond, 'RH, %'])
            else:
                return func(d.loc[cond, 'humidity'])
        elif param == 'Precipitation':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                return func(d.loc[cond, 'Rain, mm'])
            else:
                return func(d.loc[cond, 'precipitation_intensity'])
        elif param == 'WindSpeed':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                return func(d.loc[cond, 'Wind Speed, m/s'])
            else:
                return func(d.loc[cond, 'wind_speed'])
        elif param == 'GustSpeed':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                return func(d.loc[cond, 'Gust Speed, m/s'])
            else:
                return func(d.loc[cond, 'gust_speed'])
        elif param == 'WindDirection':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                col = 'Wind Direction, ø'
            else:
                col = 'wind_dir'
            if func == np.mean:
                return 180/np.pi*np.arctan2(
                    func(np.cos(d[col]*np.pi/180)),
                    func(np.sin(d[col]*np.pi/180))
                )
            else:
                return np.nan
        elif param == 'SolarRadiation':
            if lake in ['PAR', 'VEN', 'SOD', 'LJE', 'LJR']:
                return func(d.loc[cond, 'PAR, µmol/m²/s'])
            else:
                return func(d.loc[cond, 'solar_radiation'])

    def extract_chemistry_data(row, param, depth_cat):
        lake = row['lake']
        date = row['dt_ini'] + (row['dt_fin'] - row['dt_ini'])/2
        time_gap = timedelta(days=7)
        if param in ['TOC', 'DOC', 'TN', 'DN']:
            param = f'{param}_mg/L'
        elif param in ['TP', 'Chla']:
            param = f'{param}_ug/L'
        if depth_cat in water_chem.loc[lake].index.get_level_values(1):
            d_lake_depth = water_chem.loc[(lake, slice(None), depth_cat)]
            deltat = d_lake_depth.index.get_level_values(1) - date
            cond = np.abs(deltat) < time_gap
            return d_lake_depth.loc[cond, param].mean()
        else:
            return np.nan

    def extract_temperature_data(row, thermistors_data):
        lake = row['lake']
        chamber = row['chamber']
        year = row['dt_ini'].year
        if lake in ['LA', 'LO', 'ST']:
            return np.nan
        if chamber in range(1, 5):
            transect = 'T1'
        elif chamber in range(5, 9):
            transect = 'T2'
        elif chamber in range(9, 13) or chamber == '12b':
            transect = 'T3'
        elif chamber in ['par11', 'par8', 'sod12']:
            transect = 'T1'
        if lake == 'VEN' and transect == 'T3':
            transect = 'T1'
        elif lake == 'GUN' and transect == 'T1':
            transect = 'T2'
        elif lake == 'LJR' and transect == 'T2':
            transect = 'T1'
        d = thermistors_data[f'Thermistors_{lake}_{transect}_raw.mat']
        cond = np.logical_and(
            d.index >= row['dt_ini'], d.index <= row['dt_fin']
        )
        return d.loc[cond, min(d.columns)].mean()

    # DATA PROCESSING
    # Rename columns in the CH4 flux data table
    print('Importing and processing CH4 flux data...')
    if chambers is None:
        chambers = import_chambers_data()
    fc = format_ch4_flux_columns(chambers)
    create_deployment_groups_indices(fc, 'lake', 'dt_ini', 'dt_fin', 'group', 1)

    # Add MESAN data
    print('Adding MESAN data (from "chambers" data)...')
    mesan = chambers['mesan'].copy()
    mesan.index = fc.index
    mesan['Lake_ID'].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    fc = pd.concat([fc, mesan], axis=1)

    # Add anemometer data
    print('Adding anemometer data...')
    if anemo is None:
        anemo = import_merged_anemometer_data()
    col_time = 'Date Time, GMT+02:00'
    for param_name, param in params_anemo.items():
        units = param.split(', ')[1]
        for func_name, func in funcs_anemo.items():
            print(f'    Adding {func_name} values of {param_name}...')
            fc[f'{param_name}Anemometer_{func_name}_[{units}]'] = fc.apply(
                lambda row: extract_anemo_data(row, func, param), axis=1
            )

    # Add weather station data
    print('Adding weather station data...')
    if gmx is None:
        gmx = import_gmx531_data()
    if hobo_PAR is None:
        hobo_PAR = import_hobo_weather_station_data('PAR')
    if hobo_LJT is None:
        hobo_LJT = import_hobo_weather_station_data('LJT')
    for param, units in params_weather_station.items():
        for func_name, func in funcs_weather_station.items():
            print(f'    Adding {func_name} values of {param}...')
            fc[f'{param}WeatherStation_{func_name}_[{units}]'] = fc.apply(
                lambda row: extract_weather_data(row, func, param), axis=1
            )

    # Add C, N, P, Chl a data
    print('Adding chemistry data...')
    if water_chem is None:
        water_chem, _ = import_water_chemistry_data_cleaned_manually()
    for param in params_water_chemistry:
        for depth_cat in depth_cats_water_chemistry:
            print(f'    Adding {depth_cat} layer {param} data...')
            fc[f'{param}_{depth_cat}'] = fc.apply(
                lambda row: extract_chemistry_data(row, param, depth_cat),
                axis=1
            )

    # Add thermistor chains temperature data
    print('Adding thermistor chains temperature data...')
    if thermistors_data is None:
        thermistors_data = {}
        for f in os.listdir(folder_thermistors):
            if f == 'METLAKE_ThermistorChains_Info.txt':
                continue
            lake = f.split('_')[1]
            if lake in ['BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD']:
                year = 2018
            elif lake in ['SGA', 'GRI', 'GUN', 'LJE', 'LJR', 'NAS', 'NBJ']:
                year = 2019
            elif lake in ['NOR', 'GRA', 'DAM', 'LAM', 'KLI', 'GYS']:
                year = 2020
            thermistors_data[f] = import_thermistor_chain_data(
                os.path.join(folder_thermistors, f), 'dataframe', year
            )
    fc[f'WaterTemperatureThermistor'] = fc.apply(
        lambda row: extract_temperature_data(row, thermistors_data), axis=1
    )

    return fc


def create_water_concentration_table(data, ignore_gap_filled=True):
    """
    Build a separate table containing only water concentrations.

    Parameters
    ----------
    data : dict
        Dictionary of pandas.DataFrame returned by the function
        'import_data_chambers'.
    ignore_gap_filled : bool, default: True
        If True, use numpy.nan values when no GC sample is available and
        nearby values were used to fill the gap in the Excel file.
    """

    columns=[
        'Lake', 'Chamber', 'Datetime', 'Depth_[m]', 'Twat_[oC]',
        'CH4_[uM]', 'CH4_[uatm]', 'DIC_[uM]',
        'N2O_[uM]', 'N2O_[uatm]','CO2_[uM]', 'CO2_[uatm]'
    ]
    sort_columns = ['Lake', 'Chamber', 'Datetime']

    gap_fill_info = data['gap_fill'].copy()
    gap_fill_info.index = range(gap_fill_info.shape[0])
    data_vial = data['w_conc'].copy()
    data_syringe = data['pco2'].copy()

    data_vial[('General', 'Lake', 'Unnamed: 0_level_2')] = \
            data_vial[('General', 'Lake', 'Unnamed: 0_level_2')]\
            .replace({'LA': 'zLA', 'LO': 'zLO', 'ST': 'zST'})
    data_syringe[('General', 'Lake', 'Unnamed: 0_level_2')] = \
            data_syringe[('General', 'Lake', 'Unnamed: 0_level_2')]\
            .replace({'LA': 'zLA', 'LO': 'zLO', 'ST': 'zST'})

    # The function 'create_deployment_groups_indices' sorts rows according
    # to lakes and initial sampling date and time. However, as the data
    # is already sorted this way in the Excel sheet, it should not change
    # anything here. It might be important to check it again in the future
    # if modifications are done in the Excel sheet.
    create_deployment_groups_indices(
        data_vial, ('General', 'Lake', 'Unnamed: 0_level_2'),
        ('General', 'Initial sampling', 'Date and time'),
        ('General', 'Final sampling', 'Date and time'),
        ('General', 'Sampling_day', ''), 1
    )
    create_deployment_groups_indices(
        data_vial, ('General', 'Lake', 'Unnamed: 0_level_2'),
        ('General', 'Initial sampling', 'Date and time'),
        ('General', 'Final sampling', 'Date and time'),
        ('General', 'Sampling_month', ''), 24*7
    )
    create_deployment_groups_indices(
        data_syringe, ('General', 'Lake', 'Unnamed: 0_level_2'),
        ('General', 'Initial sampling', 'Date and time'),
        ('General', 'Final sampling', 'Date and time'),
        ('General', 'Sampling_day', ''), 1
    )
    create_deployment_groups_indices(
        data_syringe, ('General', 'Lake', 'Unnamed: 0_level_2'),
        ('General', 'Initial sampling', 'Date and time'),
        ('General', 'Final sampling', 'Date and time'),
        ('General', 'Sampling_month', ''), 24*7
    )

    rows = []
    for ind, row in data_vial.iterrows():
        day = row[('General', 'Sampling_day', '')]
        month = row[('General', 'Sampling_month', '')]
        days = set(data_vial.loc[
            data_vial[('General', 'Sampling_month', '')] == month,
            ('General', 'Sampling_day', '')
        ])
        if len(days) == 0:
            continue
        elif day == max(days):
            samples = ['Initial', 'Final']
        else:
            samples = ['Initial']
        for s in samples:
            if s == 'Initial' and np.isnan(row[(s, 'Depth', 'm')]):
                depth = row[('Final', 'Depth', 'm')]
            elif s == 'Final' and np.isnan(row[(s, 'Depth', 'm')]):
                depth = row[('Initial', 'Depth', 'm')]
            else:
                depth = row[(s, 'Depth', 'm')]
            Twat = row[(s, 'T H2O', 'oC')]
            values = [
                row[('General', 'Lake', 'Unnamed: 0_level_2')],
                row[('General', 'Chamber ID', 'Unnamed: 1_level_2')],
                row[('General', f'{s} sampling', 'Date and time')],
                depth, Twat
            ]
            for g in ['CH4', 'DIC', 'N2O']:
                if g == 'DIC':
                    col = (s, f'CO2 in water vial headspace', 'ppm')
                else:
                    col = (s, f'{g} in water vial headspace', 'ppm')
                if gap_fill_info.loc[ind, col] == 1 or not ignore_gap_filled:
                    val = row[(s, f'{s} conc {g}', 'µM')]
                    values.append(val)
                    if g == 'CH4' or g == 'N2O':
                        kHenry = Henry_Kh(g, row[(s, 'T H2O', 'oC')])
                        values.append(val/kHenry)
                else:
                    values.append(np.nan)
                    if g == 'CH4' or g == 'N2O':
                        values.append(np.nan)
            col = (s, 'CO2 in syringe extr. vial', 'ppm')
            if gap_fill_info.loc[ind, col] == 1 or not ignore_gap_filled:
                val = data_syringe.loc[ind, (s, 'CO2aq field', 'µM')]
                values.append(val)
                kHenry = Henry_Kh(
                    'CO2', data_syringe.loc[ind, (s, 'T H2O', 'oC')]
                )
                values.append(val/kHenry)
            else:
                values.append(np.nan)
                values.append(np.nan)
            rows.append(pd.Series(data=values, index=columns))

    water_conc = pd.concat(rows, axis=1).T.sort_values(sort_columns)
    water_conc['Lake'] = water_conc['Lake'].replace(
        {'zLA': 'LA', 'zLO': 'LO', 'zST': 'ST'}
    )
    water_conc = water_conc.astype({
        'Depth_[m]': 'float64', 'Twat_[oC]': 'float64',
        'CH4_[uM]': 'float64', 'CH4_[uatm]': 'float64', 'DIC_[uM]': 'float64',
        'N2O_[uM]': 'float64', 'N2O_[uatm]': 'float64',
        'CO2_[uM]': 'float64', 'CO2_[uatm]': 'float64'
    })

    return water_conc


def calculate_average_flux_per_deployment_group(
    data, area_lakes, use_quantiles_depth=False, col_group=None, col_lake=None,
    col_date_ini=None, col_date_fin=None, col_depth_ini=None,
    col_depth_fin=None, col_diff_flux=None, col_ebul_flux=None,
    col_tot_flux=None, col_T=None, col_k=None, col_k600=None, col_k_thld=None,
    col_k_diff=None
):
    """
    Calculate whole-lake average flux values weighted by the area of different
    depth zones for all deployment groups.

    A deployment group is a group of chambers deployed in the same lake and
    over the same period.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing data related to methane fluxes measured with
        manual flux chambers.
    area_lakes : pandas.DataFrame
        Data table containing depth vs. area information for all lakes in
        the table 'data'. The indices must contain depth values in [m] and
        the columns must contain lake names as they appear in the table 'data'.
        If the area of the deepest layer is not 0, a new layer with area 0 is
        added 0.1 m below the deepest layer.
    use_quantiles_depth : bool, default: False
        - False: Use 0 m, 1 m, 2 m, 4 m, and the maximum depth according
            to the 'area' table as boundaries for the areas to which different
            fluxes apply.
            For lakes that are shallower than 4 m, remove the 4 m boundary.
        - True: Use 0 m, the 0.25, 0.5 and 0.75 quantiles of the depths of
            all chambers in the deployment group, and the maximum depth
            according to the 'area' table as boundaries for the areas to which
            different fluxes apply.
            If less than 12 depth values are available, falls back on
            the 'False' option.
    col_group : str or tuple or None, default: None
        Column in table 'data' containing deployment group IDs.
    col_lake : str or tuple or None, default: None
        Column in table 'data' containing lake names.
    col_date_ini : str or tuple or None, default: None
        Column in table 'data' containing the date and time of the chamber's
        deployment.
    col_date_fin : str or tuple or None, default: None
        Column in table 'data' containing the data and time of the chamber's
        sampling.
    col_depth_ini : str or tuple or None, default: None
        Column in table 'data' containing initial depth of chambers.
    col_depth_fin : str or tuple or None, default: None
        Column in table 'data' containing final depth of chambers.
    col_diff_flux : str or tuple or None, default: None
        Column in table 'data' containing diffusive fluxes.
    col_ebul_flux : str or tuple or None, default: None
        Column in table 'data' containing ebullitive fluxes.
    col_tot_flux : str or tuple or None, default: None
        Column in table 'data' containing total fluxes.
    col_T : str or tuple or None, default: None
        Column in table 'data' containing surface water temperature data.
    col_k : str or tuple or None, default: None
        Column in table 'data' containing k values.
    col_k600 : str or tuple or None, default: None
        Column in table 'data' containing k600 values.
    col_k_thld : str or tuple or None, default: None
        Column in table 'data' containing the threshold k value used
        to distinguish chambers receiving ebullition from chambers receiving
        only diffusion of CH4.
    col_k_diff : str or tuple or None, default: None
        Column in table 'data' containing k values for diffusion part of fluxes.
    """

    if col_group is None:
        col_group = ('General', 'Deployment group', '')
    if col_lake is None:
        col_lake = ('General', 'Lake', 'Unnamed: 0_level_2')
    if col_date_ini is None:
        col_date_ini = ('General', 'Initial sampling', 'Date and time')
    if col_date_fin is None:
        col_date_fin = ('General', 'Final sampling', 'Date and time')
    if col_depth_ini is None:
        col_depth_ini = ('Initial', 'Depth', 'm')
    if col_depth_fin is None:
        col_depth_fin = ('Final', 'Depth', 'm')
    if col_diff_flux is None:
        col_diff_flux = \
                ('Flux calculation', 'CH4 diffusive flux', 'mmol m-2 d-1')
    if col_ebul_flux is None:
        col_ebul_flux = \
                ('Flux calculation', 'CH4 ebullitive flux', 'mmol m-2 d-1')
    if col_tot_flux is None:
        col_tot_flux = ('Flux calculation', 'CH4 total flux', 'mmol m-2 d-1')
    if col_T is None:
        col_T = ('Flux calculation', 'Mean T H2O', 'oC')
    if col_k is None:
        col_k = ('Flux calculation', 'k', 'm d-1')
    if col_k600 is None:
        col_k600 = ('Flux calculation', 'k600', 'm d-1')
    if col_k_thld is None:
        col_k_thld = ('Flux calculation', 'Min k600', 'm d-1')
    if col_k_diff is None:
        col_k_diff = ('Flux calculation', 'k_diff', 'm d-1')

    # Keep only non-NaN values from each lake's bathymetry and add a layer
    # with area equal to zero 10 cm below the deepest layer if the area of
    # the deepest layer is not equal to zero. Create interpolation functions.
    f_area_lakes = {}
    for lake in area_lakes.columns:
        area = area_lakes[lake].dropna()
        if area.sort_index().iloc[-1] != 0:
            area[area.sort_index().index[-1] + 0.1] = 0
        f_area_lakes[lake] = interp1d(area.index, area)

    groups = pd.Index(sorted(set(data[col_group]))).drop(-1)

    var_names = [
        'T', 'k', 'k600', 'k_diff', 'diff_flux', 'ebul_flux', 'tot_flux']
    var_cols = [
        col_T, col_k, col_k600, col_k_diff, col_diff_flux, col_ebul_flux,
        col_tot_flux
    ]

    columns = [
        'lake', 'initial_sampling_start', 'initial_sampling_end',
        'final_sampling_start', 'final_sampling_end',
        'deployment_start', 'deployment_end', 'deployment_time',
        'area_zone1', 'area_zone2', 'area_zone3', 'area_zone4', 'k_thld'
    ]
    for var in var_names + ['diff+ebul_flux']:
        for n in range(1, 5):
            columns.append(f'{var}_zone{n}')
    for var in var_names + ['diff+ebul_flux']:
        columns.append(var)

    results = pd.DataFrame(index=groups, columns=columns)

    for g in groups:

        d = data[data[col_group] == g].copy()

        # Extract some general information about the deployment group
        lake = d.loc[d.index[0], col_lake]
        initial_sampling_start = np.min(d[col_date_ini].dropna())
        initial_sampling_end = np.max(d[col_date_ini].dropna())
        final_sampling_start = np.min(d[col_date_fin].dropna())
        final_sampling_end = np.max(d[col_date_fin].dropna())
        t_ini = initial_sampling_start \
                + (initial_sampling_end - initial_sampling_start)/2
        t_fin = final_sampling_start \
                + (final_sampling_end - final_sampling_start)/2
        deltat = (t_fin - t_ini).total_seconds()/86400
        k_thld = d.loc[d.index[0], col_k_thld]

        # Get k, k600 and diffusive, ebullitive and total fluxes
        var_vals = {}
        for n, var in enumerate(var_names):
            var_vals[var] = d[var_cols[n]].values

        # Do not calculate anything for deployment groups containing too few
        # chambers.
        cond_nan = all(np.isnan(var_vals['diff_flux'])) and \
                all(np.isnan(var_vals['ebul_flux']))
        if d.shape[0] < 8 or cond_nan:
            results.loc[g, 'lake'] = lake
            results.loc[g, 'initial_sampling_start'] = initial_sampling_start
            results.loc[g, 'initial_sampling_end'] = initial_sampling_end
            results.loc[g, 'final_sampling_start'] = final_sampling_start
            results.loc[g, 'final_sampling_end'] = final_sampling_end
            results.loc[g, 'deployment_start'] = t_ini
            results.loc[g, 'deployment_end'] = t_fin
            results.loc[g, 'deployment_time'] = deltat
            results.loc[g, results.columns[8:]] = np.nan
            continue

        # Set the depth boundaries to use to split the lake in different depth
        # zones.
        depths = np.nanmean(d[[col_depth_ini, col_depth_fin]], axis=1)
        if use_quantiles_depth and pd.notna(depths).sum() >= 12:
            depth_lim = [
                0.0, np.nanquantile(depths, 0.25), np.nanquantile(depths, 0.5),
                np.nanquantile(depths, 0.75), max(f_area_lakes[lake].x)
            ]
        else:
            if max(f_area_lakes[lake].x) <= 4.0:
                depth_lim = [0.0, 1.0, 2.0, max(f_area_lakes[lake].x)]
            else:
                depth_lim = [0.0, 1.0, 2.0, 4.0, max(f_area_lakes[lake].x)]

        # Calculate the area and average diffusive and ebullitive fluxes
        # in each depth zone.
        n_zones = len(depth_lim) - 1
        var_zones = {}
        for var in ['area'] + var_names:
            var_zones[var] = np.zeros(n_zones)*np.nan
        for n in range(n_zones):
            area_above = f_area_lakes[lake](depth_lim[n])
            area_below = f_area_lakes[lake](depth_lim[n+1])
            var_zones['area'][n] = area_above - area_below
            if n == n_zones - 1:
                cond_depth = depths >= depth_lim[n]
            else:
                cond_depth = np.logical_and(
                    depths >= depth_lim[n], depths < depth_lim[n+1]
                )
            if cond_depth.sum() > 0:
                # No value if the only chambers that are in the depth zone do
                # not provide any flux measurement (for example if they were
                # flipped and could not be sampled).
                for var in var_names:
                    if not all(np.isnan(var_vals[var][cond_depth])):
                        var_zones[var][n] = \
                                np.nanmean(var_vals[var][cond_depth])

        # Use flux values from a nearby depth zone for depth zones where
        # no average flux could be calculated. This can arise if no chamber was
        # located in a specific depth zone during the deployment (or at least
        # if none of them provided any flux measurement in this depth zone).
        for var in var_names:
            if all(np.isnan(var_zones[var])):
                print((
                    f'No value available for "{var}" in any depth zone '
                    f'in deployment group {g}.'
                ))
            for n in range(n_zones - 1, 0, -1):
                if all(np.isnan(var_zones[var][:n])):
                    var_zones[var][:n] = var_zones[var][n]
            for n in range(1, n_zones):
                if np.isnan(var_zones[var][n]):
                    var_zones[var][n] = var_zones[var][n-1]
            if any(var_zones[var] == 0):
                print((
                    f'"{var}" is equal to 0 in some depth zone(s) '
                    f'({np.arange(1, n_zones + 1)[var_zones[var] == 0]}) '
                    f'in deployment group {g}.'
                ))

        # Calculate average fluxes over the entire lake as the average of fluxes
        # in all depth zones weighted by the area of each zone. For an array
        # containing only np.nan values, np.nansum returns 0, so the cases where
        # all values are np.nan in 'diff_flux_zones', 'ebul_flux_zones', and
        # 'tot_flux_zones' must be treated separately.
        area_tot = f_area_lakes[lake](0)
        var_lake = {}
        for var in var_names:
            if all(np.isnan(var_zones[var])):
                var_lake[var] = np.nan
            else:
                v = np.nansum(var_zones[var]*var_zones['area'])/area_tot
                var_lake[var] = v

        # Introduce data in the results table.
        results.loc[g, 'lake'] = lake
        results.loc[g, 'initial_sampling_start'] = initial_sampling_start
        results.loc[g, 'initial_sampling_end'] = initial_sampling_end
        results.loc[g, 'final_sampling_start'] = final_sampling_start
        results.loc[g, 'final_sampling_end'] = final_sampling_end
        results.loc[g, 'deployment_start'] = t_ini
        results.loc[g, 'deployment_end'] = t_fin
        results.loc[g, 'deployment_time'] = deltat
        results.loc[g, 'k_thld'] = k_thld
        for var in ['area'] + var_names:
            for n in range(1, 4):
                results.loc[g, f'{var}_zone{n}'] = var_zones[var][n-1]
            if n_zones == 4:
                results.loc[g, f'{var}_zone4'] = var_zones[var][3]
            else:
                results.loc[g, f'{var}_zone4'] = np.nan
        for n in range(1, 4):
            results.loc[g, f'diff+ebul_flux_zone{n}'] = \
                    var_zones['diff_flux'][n-1] + var_zones['ebul_flux'][n-1]
            if n_zones == 4:
                results.loc[g, f'diff+ebul_flux_zone4'] = \
                        var_zones['diff_flux'][3] + var_zones['ebul_flux'][3]
            else:
                results.loc[g, f'diff+ebul_flux_zone4'] = np.nan
        for var in var_names:
            results.loc[g, var] = var_lake[var]
        results.loc[g, 'diff+ebul_flux'] = \
                var_lake['diff_flux'] + var_lake['ebul_flux']

    dict_types = {
        'lake': 'object',
        'initial_sampling_start': 'datetime64',
        'initial_sampling_end': 'datetime64',
        'final_sampling_start': 'datetime64',
        'final_sampling_end': 'datetime64',
        'deployment_start': 'datetime64',
        'deployment_end': 'datetime64'
    }
    for col in results.columns:
        if col not in dict_types.keys():
            dict_types[col] = 'float64'
    results = results.astype(dict_types)

    return results


def calculate_average_flux_per_lake_and_year(avg_flux_groups):
    """
    Calculate whole-lake and whole-year average flux values.

    Parameters
    ----------
    avg_flux_groups : pandas.DataFrame
        Data table returned by the function
        'calculate_average_flux_per_deployment_group'.
    """

    d = avg_flux_groups.copy()
    flux_cols = [
        'k', 'k600', 'k_diff', 'diff_flux', 'ebul_flux', 'tot_flux',
        'diff+ebul_flux'
    ]
    d[flux_cols] = d.loc[:, ['deployment_time']].values*d[flux_cols].values
    d = d.groupby('lake')[['deployment_time'] + flux_cols].agg(np.sum)
    d[flux_cols] = d[flux_cols].values/d.loc[:, ['deployment_time']].values

    return d


def calculate_average_surface_concentration_per_deployment_group(
    data, area_lakes, use_quantiles_depth=False, col_group=None, col_lake=None,
    col_date=None, col_depth=None, col_T=None, col_conc=None
):
    """
    Calculate whole-lake average surface water concentration values weighted by
    the area of different depth zones for all deployment groups.

    A deployment group is a group of chambers deployed in the same lake and
    over the same period.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing surface water concentration data.
    area_lakes : pandas.DataFrame
        Data table containing depth vs. area information for all lakes in
        the table 'data'. The indices must contain depth values in [m] and
        the columns must contain lake names as they appear in the table 'data'.
        If the area of the deepest layer is not 0, a new layer with area 0 is
        added 0.1 m below the deepest layer.
    use_quantiles_depth : bool, default: False
        - True: Use 0 m, the 0.25, 0.5 and 0.75 quantiles of the depths of
            all chambers in the deployment group, and the maximum depth
            according to the 'area' table as boundaries for the areas to which
            different fluxes apply.
            If less than 12 depth values are available, falls back on
            the 'False' option.
        - False: Use 0 m, 1 m, 2 m, 4 m, and the maximum depth according
            to the 'area' table as boundaries for the areas to which different
            fluxes apply.
            For lakes that are shallower than 4 m, remove the 4 m boundary.
    col_group : str or tuple or None, default: None
        Column in table 'data' containing deployment group IDs.
    col_lake : str or tuple or None, default: None
        Column in table 'data' containing lake names.
    col_date : str or tuple or None, default: None
        Column in table 'data' containing the date and time of the chamber's
        deployment.
    col_depth : str or tuple or None, default: None
        Column in table 'data' containing initial depth of chambers.
    col_T : str or tuple or None, default: None
        Column in table 'data' containing surface water temperature data.
    col_conc : str or tuple or None, default: None
        Column in table 'data' containing surface water concentration.
    """

    if col_group is None:
        col_group = 'Group'
    if col_lake is None:
        col_lake = 'Lake'
    if col_date is None:
        col_date = 'Datetime'
    if col_depth is None:
        col_depth = 'Depth_[m]'
    if col_T is None:
        col_T = 'Twat_[oC]'
    if col_conc is None:
        col_conc = 'CH4_[uM]'

    # Keep only non-NaN values from each lake's bathymetry and add a layer
    # with area equal to zero 10 cm below the deepest layer if the area of
    # the deepest layer is not equal to zero. Create interpolation functions.
    f_area_lakes = {}
    for lake in area_lakes.columns:
        area = area_lakes[lake].dropna()
        if area.sort_index().iloc[-1] != 0:
            area[area.sort_index().index[-1] + 0.1] = 0
        f_area_lakes[lake] = interp1d(area.index, area)

    groups = pd.Index(sorted(set(data[col_group]))).drop(-1)

    var_names = ['T', col_conc]
    var_cols = [col_T, col_conc]

    columns = ['lake', 'sampling_start', 'sampling_end', 'sampling_middle']
    for var in ['area'] + var_names:
        for n in range(1, 5):
            columns.append(f'{var}_zone{n}')
    columns.append('T')
    columns.append(col_conc)

    results = pd.DataFrame(index=groups, columns=columns)

    for g in groups:

        d = data[data[col_group] == g].copy()

        # Extract some general information about the deployment group
        lake = d.loc[d.index[0], col_lake]
        sampling_start = np.min(d[col_date].dropna())
        sampling_end = np.max(d[col_date].dropna())
        sampling_middle = sampling_start \
                + (sampling_end - sampling_start)/2

        # Get surface water temperature and concentrations
        var_vals = {}
        for n, var in enumerate(var_names):
            var_vals[var] = d[var_cols[n]].values

        # Do not calculate anything for deployment groups containing too few
        # chambers.
        if d.shape[0] < 8 or all(np.isnan(var_vals[col_conc])):
            results.loc[g, 'lake'] = lake
            results.loc[g, 'sampling_start'] = sampling_start
            results.loc[g, 'sampling_end'] = sampling_end
            results.loc[g, 'sampling_middle'] = sampling_middle
            results.loc[g, results.columns[4:]] = np.nan
            continue

        # Set the depth boundaries to use to split the lake in different depth
        # zones.
        depths = d[col_depth]
        if use_quantiles_depth and pd.notna(depths).sum() >= 12:
            depth_lim = [
                0.0, np.nanquantile(depths, 0.25), np.nanquantile(depths, 0.5),
                np.nanquantile(depths, 0.75), max(f_area_lakes[lake].x)
            ]
        else:
            if max(f_area_lakes[lake].x) <= 4.0:
                depth_lim = [0.0, 1.0, 2.0, max(f_area_lakes[lake].x)]
            else:
                depth_lim = [0.0, 1.0, 2.0, 4.0, max(f_area_lakes[lake].x)]

        # Calculate the area and average surface water concentration in each
        # depth zone.
        n_zones = len(depth_lim) - 1
        var_zones = {}
        for var in ['area'] + var_names:
            var_zones[var] = np.zeros(n_zones)*np.nan
        for n in range(n_zones):
            area_above = f_area_lakes[lake](depth_lim[n])
            area_below = f_area_lakes[lake](depth_lim[n+1])
            var_zones['area'][n] = area_above - area_below
            if n == n_zones - 1:
                cond_depth = depths >= depth_lim[n]
            else:
                cond_depth = np.logical_and(
                    depths >= depth_lim[n], depths < depth_lim[n+1]
                )
            if cond_depth.sum() > 0:
                # No value if the only chambers that are in the depth zone were
                # not sampled for surface water concentration (for example
                # if they were flipped).
                for var in var_names:
                    if not all(np.isnan(var_vals[var][cond_depth])):
                        var_zones[var][n] = \
                                np.nanmean(var_vals[var][cond_depth])

        # Use surface water concentration values from a nearby depth zone for
        # depth zones where no average surface water concentration could be
        # calculated. This can arise if no chamber was located in a specific
        # depth zone during the deployment (or at least if none of them were
        # sampled for surface water concentration in this depth zone).
        for var in var_names:
            if all(np.isnan(var_zones[var])):
                print((
                    f'No value available for "{var}" in any depth zone '
                    f'in deployment group {g}.'
                ))
            for n in range(n_zones - 1, 0, -1):
                if all(np.isnan(var_zones[var][:n])):
                    var_zones[var][:n] = var_zones[var][n]
            for n in range(1, n_zones):
                if np.isnan(var_zones[var][n]):
                    var_zones[var][n] = var_zones[var][n-1]
            if any(var_zones[var] == 0):
                print((
                    f'"{var}" is equal to 0 in some depth zone(s)'
                    f'({np.arange(1, n_zones + 1)[var_zones[var] == 0]}) '
                    f'in deployment group {g}.'
                ))

        # Calculate average surface water concentration over the entire lake as
        # the average of surface water concentrations in all depth zones
        # weighted by the area of each zone. For an array containing only np.nan
        # values, np.nansum returns 0, so the case where all values are np.nan
        # in 'conc_zones' must be treated separately.
        area_tot = f_area_lakes[lake](0)
        var_lake = {}
        for var in var_names:
            if all(np.isnan(var_zones[var])):
                var_lake[var] = np.nan
            else:
                v = np.nansum(var_zones[var]*var_zones['area'])/area_tot
                var_lake[var] = v

        # Introduce data in the results table.
        results.loc[g, 'lake'] = lake
        results.loc[g, 'sampling_start'] = sampling_start
        results.loc[g, 'sampling_end'] = sampling_end
        results.loc[g, 'sampling_middle'] = sampling_middle
        for var in ['area'] + var_names:
            for n in range(1, 4):
                results.loc[g, f'{var}_zone{n}'] = var_zones[var][n-1]
            if n_zones == 4:
                results.loc[g, f'{var}_zone4'] = var_zones[var][3]
            else:
                results.loc[g, f'{var}_zone4'] = np.nan
        for var in var_names:
            results.loc[g, var] = var_lake[var]

    dict_types = {
        'lake': 'object',
        'sampling_start': 'datetime64',
        'sampling_end': 'datetime64',
        'sampling_middle': 'datetime64',
    }
    for col in results.columns:
        if col not in dict_types.keys():
            dict_types[col] = 'float64'
    results = results.astype(dict_types)

    return results


def calculate_average_surface_concentration_per_lake_and_year(
    avg_conc_groups, col_conc
):
    """
    Calculate whole-lake and whole-year average water surface concentration.

    Parameters
    ----------
    avg_conc_groups : pandas.DataFrame
        Data table returned by the function
        'calculate_average_surface_concentration_per_deployment_group'.
    """

    lakes = sorted(set(avg_conc_groups['Lake']))
    d = pd.DataFrame(index=lakes, columns=[col_conc], dtype='float64')
    d.index.name = 'Lake'
    for lake, group in avg_conc_groups.groupby('Lake'):
        month_avg = group.set_index('sampling_middle').resample('M').mean()
        d.loc[lake, col_conc] = month_avg.mean()[col_conc]

    return d


def calculate_average_flux_or_concentration_per_lake_and_year_using_temperature(
    d_avg, T_lakes, cols
):
    """
    Calculate whole-lake and whole-year average flux or surface water
    concentration values using depth zones to interpolate in space
    and temperature data to interpolate in time.

    Parameters
    ----------
    d_avg : pandas.DataFrame
        Data table returned by the function
        'calculate_average_flux_per_deployment_group' or
        'calculate_average_surface_concentration_per_deployment_group'.
    T_lakes : dict
        Dictionary of pandas.DataFrame tables obtained using the function
        'calculate_average_daily_water_temperature_lakes'.
    cols : list
        List of columns in table 'd_avg' that will be averaged.
    """

    info_lakes, _, _ = import_info_lakes()

    f = lambda x, a, b: a*np.exp(b*x)

    res = pd.DataFrame(index=sorted(set(d_avg['lake'])), columns=cols)

    for lake, d_lake in d_avg.groupby('lake'):
        print(f'\n{lake}')
        if lake in ['LA', 'LO', 'ST']:
            dti = datetime(2008, 1, 1)
            dte = datetime(2008, 12, 31)
        else:
            dti = info_lakes.loc[
                info_lakes['Lake'] == lake, 'Open water season start'
            ].values[0]
            dtf = info_lakes.loc[
                info_lakes['Lake'] == lake, 'Open water season end'
            ].values[0]
        if isinstance(dti, str):
            # year-1 because the only lakes for which the open water season
            # start dates are missing are the lakes in Bergslagen, and
            # the open water season end dates in this region are set at
            # the beginning of the following year.
            dti = datetime(dtf.year - 1, 1, 1)
        if isinstance(dtf, str):
            dtf = datetime(dti.year, 12, 31)
        T = T_lakes[lake]
        T = T.loc[np.logical_and(T.index >= dti, T.index <= dtf), 'T_water']
        if lake in ['SOD', 'BD6', 'LAM', 'DAM', 'ST']:
            n_zones = 3
        else:
            n_zones = 4
        cols_area = [f'area_zone{n}' for n in range(1, n_zones+1)]
        area_zones = d_lake[cols_area].dropna().iloc[0].values
        for col in cols:
            avg_vals = np.zeros(n_zones)*np.nan
            for n in range(n_zones):
                nz = n + 1
                d_zone = d_lake[[f'T_zone{nz}', f'{col}_zone{nz}']].dropna()
                if col.endswith('flux') and lake == 'PAR':
                    # Previously (before adding three lakes from Uppsala),
                    # dropped rows 282 and 283.
                    # d_zone = d_zone.drop([282, 283])
                    d_zone = d_zone.iloc[np.r_[0:18, 20:24]]
                elif col == 'ebul_flux' and lake == 'GRA' and nz == 1:
                    # Previously (before adding three lakes from Uppsala),
                    # dropped rows 93 and 94.
                    # d_zone = d_zone.drop([93, 94])
                    d_zone = d_zone.iloc[np.r_[0:5, 7:15]]
                elif col == 'ebul_flux' and lake == 'GRI' and nz == 4:
                    # This is not an optimal solution because the resulting
                    # regression line increases steeply at high
                    # temperatures. This effect is reduced by removing row 112.
                    # Previously (before adding three lakes from Uppsala),
                    # dropped rows 112 and 114.
                    # d_zone = d_zone.drop([112, 114])
                    d_zone = d_zone.iloc[np.r_[0:8, 9:10, 11:17]]
                elif col == 'ebul_flux' and lake == 'KLI' and nz == 1:
                    # This is not an optimal solution because the resulting
                    # regression line underestimates the average flux
                    # obtained when the very high flux value at the
                    # highest temperature is considered.
                    # Previously (before adding three lakes from Uppsala),
                    # dropped row 166.
                    # d_zone = d_zone.drop(166)
                    d_zone = d_zone.iloc[np.r_[0:8, 9:16]]
                elif col == 'tot_flux' and lake == 'KLI' and nz == 1:
                    # This is not an optimal solution because the resulting
                    # regression line underestimates the average flux
                    # obtained when the very high flux value at the
                    # highest temperature is considered. Dropping row 167
                    # instead leads to a regression that increases fast
                    # at high temperatures. Row 166 is removed here for
                    # consistence with the ebullition case.
                    # Previously (before adding three lakes from Uppsala),
                    # dropped row 166.
                    # d_zone = d_zone.drop(166)
                    d_zone = d_zone.iloc[np.r_[0:8, 9:16]]
                elif col == 'CH4_[uM]' and lake == 'PAR':
                    # Previously (before adding three lakes from Uppsala),
                    # dropped rows 423, 424, and 425.
                    # d_zone = d_zone.drop([423, 424, 425])
                    d_zone = d_zone.iloc[np.r_[0:23, 26:32]]
                try:
                    popt, pcov = curve_fit(
                        f, d_zone.iloc[:, 0], d_zone.iloc[:, 1],
                        [1e-2, 1e-2]
                    )
                    r2 = r2_score(d_zone.iloc[:, 0], d_zone.iloc[:, 1])
                    if popt[1] < 0:
                        avg_val = np.mean(d_zone.iloc[:, 1])
                    else:
                        avg_val = np.mean(f(T, *popt))
                except RuntimeError:
                    avg_val = np.mean(d_zone.iloc[:, 1])
                avg_vals[n] = avg_val
                print(f'zone {n+1}', col, f'{r2:.3f}', f'{avg_val:.3f}')
            res.loc[lake, col] = sum(avg_vals*area_zones)/np.sum(area_zones)

    dict_types = {}
    for col in cols:
        dict_types[col] = 'float64'
    res = res.astype(dict_types)

    return res


def calculate_average_daily_water_temperature_lakes(mesan):
    """
    Use average daily MESAN air temperature data to fill gaps in timeseries
    of average daily surface water temperature obtained from the average
    of the thermistors located within the 10 first centimeters of the water
    column in each thermistor chain. Calculations are done for all lakes.

    Parameters
    ----------
    mesan : dict
        Dictionary of pandas.DataFrame tables obtained using the function
        'import_mesan_data'.
    """

    p = '../Data/ThermistorChains/'
    lakes = [
        'BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM'
    ]
    lakes_uppsala = ['LA', 'LO', 'ST']
    conv_lake = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExp', 'LJR': 'LjusvattentjarnRef',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }
    T_lakes = {}
    # Use METLAKE thermistor and weather station data for METLAKE lakes
    for lake in lakes:
        if lake in ['BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD']:
            year = 2018
        elif lake in ['SGA', 'GRI', 'GUN', 'LJE', 'LJR', 'NAS', 'NBJ']:
            year = 2019
        elif lake in ['NOR', 'GRA', 'DAM', 'LAM', 'KLI', 'GYS']:
            year = 2020
        T_surf_all = []
        for n in range(1, 4):
            f = f'Thermistors_{lake}_T{n}_raw.mat'
            if f in os.listdir(p):
                T = import_thermistor_chain_data(
                    os.path.join(p, f), mode='dataframe', year=year
                )
                if min(T.columns) < 0.1:
                    T_surf = T[min(T.columns)]
                else:
                    continue
                if lake == 'GUN' and n == 2:
                    datetime_failure = datetime(2019, 9, 12, 6, 0)
                    T_surf[T_surf.index >= datetime_failure] = np.nan
                T_surf.name = f'T{n}'
                T_surf_all.append(T_surf)
        T_surf_all = pd.concat(T_surf_all, axis=1).resample('1D').mean()
        T_surf_all_avg = T_surf_all.mean(axis=1)
        mesan_lake = mesan[conv_lake[lake]]
        T_mesan_lake = mesan_lake.loc[
            np.logical_and(
                mesan_lake.index >= T_surf_all.index[0],
                mesan_lake.index <= T_surf_all.index[-1]
            ), 'Temperature'
        ].resample('1D').mean()
        T_air_wat = pd.concat([T_mesan_lake, T_surf_all_avg], axis=1)
        T_air_wat = T_air_wat.dropna()
        X = T_air_wat.iloc[:, 0].values.reshape(-1, 1)
        y = T_air_wat.iloc[:, 1].values.reshape(-1, 1)
        lm = LinearRegression().fit(X, y)
        T_mesan_lake = mesan_lake.loc[
            mesan_lake.index.map(lambda x: x.year == year), 'Temperature'
        ].resample('1D').mean()
        T_mesan_lake.name = 'T_air_MESAN'
        T_surf_all_avg.name = 'T_water_meas'
        T_water_pred = pd.Series(
            lm.predict(T_mesan_lake.values.reshape(-1, 1)).reshape(-1),
            index=T_mesan_lake.index, name='T_water_pred'
        )
        T_lake = pd.concat(
            [T_mesan_lake, T_water_pred, T_surf_all_avg], axis=1
        )
        T_lake['T_water_pred_rolling_avg'] = \
                T_lake['T_water_pred'].rolling(7, center=True).mean()
        def assign_T(row):
            if np.isnan(row['T_water_meas']):
                if np.isnan(row['T_water_pred_rolling_avg']):
                    return row['T_water_pred']
                else:
                    return row['T_water_pred_rolling_avg']
            else:
                return row['T_water_meas']
        T_lake['T_water'] = T_lake.apply(assign_T, axis=1)
        T_lake.loc[T_lake['T_water'] < 0, 'T_water'] = 0.0
        T_lakes[lake] = T_lake
    # Use air temperature measured at Uppsala Flygplats and add 3.5 oC
    # for the three lakes that were sampled near Uppsala in 2008.
    # The 3.5 oC correction is obtained from linear regressions derived
    # above in lakes at similar latitudes and with similar colors
    # in the METLAKE project.
    T_uppsala = SMHI_data.load_station_data(1, 97530, 'corrected-archive')
    T_uppsala.set_index('Datetime', inplace=True)
    T_uppsala = T_uppsala\
            .loc[T_uppsala.index.map(lambda x: x.year == 2008)]\
            .resample('1D').mean()\
            .rename({'Value': 'T_water'}, axis=1)
    T_uppsala['T_water'] += 3.5
    T_uppsala.loc[T_uppsala['T_water'] < 0, 'T_water'] = 0.0
    for lake in lakes_uppsala:
        T_lakes[lake] = T_uppsala

    return T_lakes


def calculate_average_chemistry_per_lake_and_year_OLD(CN, P, Chla, depth):
    """
    Calculate yearly average values for total and dissolved carbon, total
    nitrogen, total phosphorus and chlorophyll a concentration in the surface
    water of all lakes.

    Parameters
    ----------
    CN : pandas.DataFrame
        Table with carbon and nitrogen data returned by the function
        'import_water_chemistry_data' when using mode='raw'.
    P : pandas.DataFrame
        Table with phosphorus data returned by the function
        'import_water_chemistry_data' when using mode='raw'.
    Chla : pandas.DataFrame
        Table with chlorophyll a data returned by the function
        'import_water_chemistry_data' when using mode='raw'.
    depth : {'surface', 'middle', 'bottom'}
        Depth category from which to extract data.
    """

    cond_totC = np.logical_and(
        np.logical_and(
            CN['Anal.'].apply(lambda x: x in ['TOC', 'TC']), CN['Reference_run']
        ),
        np.logical_and(
            CN['Depth_category'] == depth, np.logical_not(CN['Filtered'])
        )
    )
    cond_disC = np.logical_and(
        np.logical_and(
            CN['Anal.'].apply(lambda x: x in ['TOC', 'TC']), CN['Reference_run']
        ),
        np.logical_and(CN['Depth_category'] == depth, CN['Filtered'])
    )
    cond_totN = np.logical_and(
        np.logical_and(CN['Anal.'] == 'TN', CN['Reference_run']),
        np.logical_and(
            CN['Depth_category'] == depth, np.logical_not(CN['Filtered'])
        )
    )
    cond_disN = np.logical_and(
        np.logical_and(CN['Anal.'] == 'TN', CN['Reference_run']),
        np.logical_and(CN['Depth_category'] == depth, CN['Filtered'])
    )

    TC_avg = CN[cond_totC].groupby('Lake').agg(np.nanmean)['TC_final_mg/L']
    TC_avg.rename('TC_mg/L', inplace=True)
    TOC_avg = CN[cond_totC].groupby('Lake').agg(np.nanmean)['TOC_final_mg/L']
    TOC_avg.rename('TOC_mg/L', inplace=True)
    DOC_avg = CN[cond_disC].groupby('Lake').agg(np.nanmean)['TOC_final_mg/L']
    DOC_avg.rename('DOC_mg/L', inplace=True)
    TN_avg = CN[cond_totN].groupby('Lake').agg(np.nanmean)['TN_final_mg/L']
    TN_avg.rename('TN_mg/L', inplace=True)
    DN_avg = CN[cond_disN].groupby('Lake').agg(np.nanmean)['TN_final_mg/L']
    DN_avg.rename('DN_mg/L', inplace=True)
    P_avg = P[P['Depth_category'] == depth].groupby('Lake').agg(np.nanmean)
    P_avg = P_avg['TP_ug/L']
    Chla_avg = Chla[Chla['Depth category'] == depth].groupby('Lake')
    Chla_avg = Chla_avg.agg(np.nanmean)['Chlorophyll A concentration [ug/L]']
    Chla_avg = Chla_avg.rename('chla_ug/L', inplace=True)

    chem_avg = pd.concat(
        [TC_avg, TOC_avg, DOC_avg, TN_avg, DN_avg, P_avg, Chla_avg], axis=1
    )

    return chem_avg


def calculate_light_attenuation_from_profiles(data):
    """
    Calculate light attenuation in water from depth profiles measurements.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing depth profiles data imported with the function
        'import_depth_profiles_data'.
    """

    results = pd.DataFrame(
        index=['Lake', 'Date', 'Attenuation_m-1', 'R2', 'n_values']
    )

    col_lake = ('General', 'Lake', 'Unnamed: 1_level_2')
    col_date = ('General', 'Date', 'Unnamed: 2_level_2')
    col_depth = ('General', 'Depth', 'm')
    col_par = ('LI-COR and HACH probes', 'PAR', 'µmol s-1 m-2')

    for lake, group_lake in data.groupby(col_lake):
        for date, group_date in group_lake.groupby(col_date):
            d = group_date[[col_depth, col_par]].dropna()
            d = d[np.logical_and(d[col_depth] > 0.0, d[col_par] > 0.0)]
            n = d.shape[0]
            if n == 0:
                results = pd.concat(
                    [results, pd.Series({
                        'Lake': lake, 'Date': date.date(),
                        'Attenuation_m-1': np.nan, 'R2': np.nan, 'n_values': n
                    })], axis=1
                )
            else:
                reg = LinearRegression().fit(
                    d[col_depth].values.reshape(-1, 1),
                    np.log(d[col_par].values)
                )
                r2 = reg.score(
                    d[col_depth].values.reshape(-1, 1),
                    np.log(d[col_par].values)
                )
                results = pd.concat(
                    [results, pd.Series({
                        'Lake': lake, 'Date': date.date(),
                        'Attenuation_m-1': -reg.coef_[0], 'R2': r2, 'n_values': n
                    })], axis=1
                )

    results = results.T
    results.index = range(results.shape[0])

    return results


def calculate_water_retention_time_lakes(mode):
    """
    Calculate the water retention time of the METLAKE lakes using recent data.

    Parameters
    ----------
    mode : {'average', 'sampling_year'}
        If mode is 'average', calculate the average retention time for
        the years 1991-2020.
        If mode is 'sampling_year', calculate the retention time for
        the year when each lake was sampled.
    """

    path_data = '../Data/S-Hype'
    info, _, _ = import_info_lakes('../Data/METLAKE_InfoLakes_withDBlakes.xlsx')

    res = []
    for f in os.listdir(path_data):
        lakes = f[:-4].split('_')[1:]
        if mode == 'average':
            d_avg = pd.read_excel(
                os.path.join(path_data, f), sheet_name='Områdesinformation',
                header=57, index_col=0, usecols='A:C', nrows=3
            )
            for lake in lakes:
                P = d_avg.loc['Nederbörd [mm/år]', 'Delavrinningsområdet']
                EP = d_avg.loc['Evapotranspiration [mm/år]',
                               'Delavrinningsområdet']
                runoff = d_avg.loc['Avrinning [mm/år]', 'Delavrinningsområdet']
                info_lake_cond = info['Lake'] == lake
                catchment_area = info.loc[info_lake_cond, 'CatchmentArea_[m2]']
                lake_volume = info.loc[info_lake_cond, 'LakeVolume_[m3]']
                residence_time = lake_volume/(1e-3*runoff*catchment_area)*365
                res.append(pd.Series(
                    {'P': P, 'EP': EP, 'runoff': runoff,
                     'residence_time': residence_time.values[0]},
                    name=lake
                ))
        elif mode == 'sampling_year':
            d_catchment = pd.read_excel(
                os.path.join(path_data, f), sheet_name='Områdesinformation',
                header=10, index_col=0, usecols='A:B', nrows=10
            )
            d_year = pd.read_excel(
                os.path.join(path_data, f), sheet_name='Årsvärden',
                index_col=0, nrows=17
            )
            for lake in lakes:
                if lake in ['LA', 'LO', 'ST']:
                    year = 2008
                elif lake in ['BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD']:
                    year = 2018
                elif lake in ['SGA', 'GUN', 'GRI', 'LJE', 'LJR', 'NAS', 'NBJ']:
                    year = 2019
                elif lake in ['NOR', 'GRA', 'DAM', 'LAM', 'GYS', 'KLI']:
                    year = 2020
                smhi_catchment_area = d_catchment.loc['Area [km²]:',
                                                      'Unnamed: 1']
                Q = d_year.loc[year, 'Lokal\nvattenföring\n[m³/s]']
                P = d_year.loc[year, 'Lokal\nnederbörd\n[mm]']
                EP = d_year.loc[year, 'Lokal\nevapotranspiration\n[mm]']
                runoff = Q*3600*24*365/(1e6*smhi_catchment_area)*1000
                info_lake_cond = info['Lake'] == lake
                catchment_area = info.loc[info_lake_cond, 'CatchmentArea_[m2]']
                lake_volume = info.loc[info_lake_cond, 'LakeVolume_[m3]']
                residence_time = lake_volume/(1e-3*runoff*catchment_area)*365
                res.append(pd.Series(
                    {'P': P, 'EP': EP, 'runoff': runoff,
                     'residence_time': residence_time.values[0]},
                    name=lake
                ))
    res = pd.concat(res, axis=1).T

    return res


def calculate_water_retention_time_lakes_SVAR2016():
    """
    Calculate the water retention time of the METLAKE lakes using SVAR data
    (outdated).
    """

    id_table = pd.read_csv(os.path.join(
        '..', 'Data', 'SvensktVattenarkiv_2016_3',
        'METLAKE_lakes_SvensktVattenarkiv.txt'
    ))
    id_table['Lake'] = id_table['LAKE_CODE']
    waterbalance_svar = pd.read_excel(
        '../Data/SvensktVattenarkiv_2016_3/Vattenbalans.xls',
        sheet_name='Vattenbalans Lokalt'
    )
    info_lakes, area_lakes, volume_lakes = import_info_lakes()
    volume_lakes = volume_lakes.droplevel(1, axis=1)
    volume_lakes = volume_lakes.T
    volume_lakes['Lake'] = volume_lakes.index
    summary = pd.merge(id_table, waterbalance_svar, on='AROID')
    summary = pd.merge(summary, info_lakes, on='Lake')
    summary = pd.merge(summary, volume_lakes, on='Lake')
    summary = summary[
        ['Lake', 'Nederbörd', 'Avrinning', 'CatchmentArea_[m2]', 0]
    ]
    summary.rename(
        columns={
            0: 'LakeVolume_[m3]', 'Avrinning': 'Runoff_[mm/year]',
            'Nederbörd': 'Precipitation_[mm/year]'
        }, inplace=True
    )
    summary['RunoffVolume_[m3/year]'] = \
            1e-3*summary['Runoff_[mm/year]']*summary['CatchmentArea_[m2]']
    summary['RetentionTime_[year]'] = \
            summary['LakeVolume_[m3]']/summary['RunoffVolume_[m3/year]']

    return summary


def calculate_storage_and_VFAN_in_lake(
    dp, volume, lake, date, from_depth=0.0, comp='CH4', thld_o2=0.2
):
    """
    Calculate the amount of a gas or compound stored in a lake and the fraction
    of the lake's volume that is anoxic.

    Parameters
    ----------
    dp : pd.DataFrame
        Data table returned by the function 'import_depth_profiles_data'.
    volume : pd.DataFrame
        Data table returned by the function 'import_info_lakes' and containing
        volumes of lakes by layers.
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake of interest.
    date : datetime.datetime
        Date of the profile of interest.
    from_depth : float or 'anoxic', default: 0.0
        If a float is passed, calculate storage below this depth.
        If 'anoxic' is passed, calculate storage below the point where
        dissolved oxygen concentration is lower than 'thld_o2' (in mg/L).
    comp : {'CH4', 'CO2', 'N2O'}, default: 'CH4'
        Compound of interest. Note that using CO2 returns DIC values.
    thld_o2 : float, default: 0.2
        Dissolved oxygen concentration [mg/L] threshold to consider that anoxic
        conditions prevail.
    """

    conv_lake = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExperiment', 'LJR': 'LjusvattentjarnReference',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }

    cond = np.logical_and(
        dp[('General', 'Lake', 'Unnamed: 1_level_2')] == conv_lake[lake],
        dp[('General', 'Date', 'Unnamed: 2_level_2')] == date
    )
    col_depth = ('General', 'Depth', 'm')
    col_comp = ('CH4, DIC, N2O aq conc', f'{comp}aq', 'µM')
    col_o2 = ('LI-COR and HACH probes', 'DissolvedOxygen', 'mg/L')
    dp_comp = dp.loc[cond, [col_depth, col_comp]]
    dp_comp = dp_comp.dropna().sort_values(col_depth)
    dp_o2 = dp.loc[cond, [col_depth, col_o2]]
    dp_o2 = dp_o2.dropna().sort_values(col_depth)
    volume_depths = (volume.index[:-1] + volume.index[1:])/2
    volume_depths = volume_depths.append(pd.Index([np.nan], dtype='float64'))
    volume_cum = volume.droplevel(1, axis=1)[lake].copy()
    volume_cum.index = volume_depths
    volume_layers = volume_cum.diff(-1).dropna()
    flag_print_profiles = False

    if (dp_comp.shape[0] == 0 and dp_o2.shape[0] == 0) or \
       (from_depth == 'anoxic' and dp_o2.shape[0] == 0) or \
       (from_depth == 'anoxic' and min(dp_o2[col_o2]) > thld_o2):
        return np.nan, np.nan
    elif dp_o2.shape[0] > 0:
        if min(dp_o2[col_o2]) > thld_o2:
               anoxic_depth = 1e9
               vfan = 0.0
        else:
            interp_o2 = interp1d(
                dp_o2[col_depth], dp_o2[col_o2], bounds_error=False,
                fill_value=(dp_o2.iloc[0][col_o2], dp_o2.iloc[-1][col_o2])
            )
            ind = np.argmax(interp_o2(volume_layers.index) <= thld_o2)
            anoxic_depth = volume_layers.index[ind]
            vfan = volume_cum[anoxic_depth]/volume_cum[min(volume_cum.index)]
        if any(dp_o2.loc[dp_o2[col_depth] >= anoxic_depth, col_o2] > thld_o2):
            print('High O2 concentration in the assumed anoxic zone.')
            flag_print_profiles = True
        if dp_comp.shape[0] == 0:
            if flag_print_profiles:
                print(f'Threshold depth for anoxic zone: {anoxic_depth:.2f} m')
                print(dp_o2)
                print(dp_comp)
            return np.nan, vfan
        elif any(dp_comp.loc[dp_comp[col_depth] >= anoxic_depth, col_comp] < 2.0)\
                and comp == 'CH4':
            print('Low CH4 concentration in the assumed anoxic zone.')
            flag_print_profiles = True
        if flag_print_profiles:
            print(f'Threshold depth for anoxic zone: {anoxic_depth:.2f} m')
            print(dp_o2)
            print(dp_comp)
    else:
        vfan = np.nan

    interp_comp = interp1d(
        dp_comp[col_depth], dp_comp[col_comp], bounds_error=False,
        fill_value=(dp_comp.iloc[0][col_comp], dp_comp.iloc[-1][col_comp])
    )

    # Total amount of compound is in [mol] (multiplication of [m3] and [µM]
    # and division by 1000 for conversion to [mol]).
    # Multiply the concentration in the middle of each layer with the layer's
    # volume.
    if from_depth == 'anoxic':
        from_depth = anoxic_depth
    storage = volume_layers*interp_comp(volume_layers.index)/1000
    storage = storage[volume_layers.index >= from_depth].sum()

    return storage, vfan


def calculate_storage_and_VFAN_in_all_lakes_for_all_dates(
    dp=None, volume=None, from_depth=0.0, comp='CH4', thld_o2=0.2
):
    """
    Calculate the amount of a compound stored in all lakes and the fraction
    of the lakes' volume that is anoxic (only if 'from_depth'='anoxic') for
    all sampling dates.

    Parameters
    ----------
    dp : pd.DataFrame
        Data table returned by the function 'import_depth_profiles_data'.
    volume : pd.DataFrame
        Data table returned by the function 'import_info_lakes' and containing
        volumes of lakes by layers.
    from_depth : float or 'anoxic', default: 0.0
        If a float is passed, calculate storage below this depth.
        If 'anoxic' is passed, calculate storage below the point where
        dissolved oxygen concentration is lower than 'thld_o2' (in mg/L).
    comp : {'CH4', 'CO2', 'N2O'}, default: 'CH4'
        Compound of interest. Note that using CO2 returns DIC values.
    thld_o2 : float, default: 0.2
        Dissolved oxygen concentration [mg/L] threshold to consider that anoxic
        conditions prevail.
    """

    if dp is None:
        dp = import_depth_profiles_data()
    if volume is None:
        _, _, volume = import_info_lakes()

    conv_lakes = {
        'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6',
        'Venasjon': 'VEN', 'Parsen': 'PAR', 'SodraTeden': 'SOD',
        'StoraGalten': 'SGA', 'Gundlebosjon': 'GUN', 'Grinnsjon': 'GRI',
        'LjusvattentjarnExperiment': 'LJE', 'LjusvattentjarnReference': 'LJR',
        'Nastjarn': 'NAS', 'NedreBjorntjarn': 'NBJ',
        'Dammsjon': 'DAM', 'Norrtjarn': 'NOR', 'Grastjarn': 'GRA',
        'Lammen': 'LAM', 'Klintsjon': 'KLI', 'Gyslattasjon': 'GYS'
    }

    results = {}
    for lake, group in dp.groupby(('General', 'Lake', 'Unnamed: 1_level_2')):
        print(f'\n{lake}')
        dates = sorted(set(group[('General', 'Date', 'Unnamed: 2_level_2')]))
        lake_data = pd.DataFrame(
            index=dates, columns=['storage', 'anoxic volume fraction'],
            dtype='float64'
        )
        for date in dates:
            print(date.date())
            storage, vfan = calculate_storage_and_VFAN_in_lake(
                dp, volume, conv_lakes[lake], date,
                from_depth=from_depth, comp=comp, thld_o2=thld_o2
            )
            if not np.isnan(storage):
                lake_data.loc[date, 'storage'] = storage
            if not np.isnan(vfan):
                lake_data.loc[date, 'anoxic volume fraction'] = vfan
        results[conv_lakes[lake]] = lake_data

    return pd.concat(results)


def combine_all_yearly_average_data(
    chambers=None, water_conc=None, profiles=None, chem_avg=None,
    info=None, area=None, volume=None
):
    """
    Combine yearly average data from a variety of sources.

    Parameters
    ----------
    chambers : dict or None, default: None
        Dictionary of data tables imported with the function
        'import_chambers_data'.
    water_conc : pd.DataFrame or None, default: None
        Data table created with the function 'create_water_concentration_table'.
    profiles : pd.DataFrame or None, default: None
        Data table imported with the function 'import_depth_profiles_data'.
    chem_avg : pd.DataFrame or None, default: None
        Second data table imported with the function
        'import_water_chemistry_data_cleaned_manually.
    info : pd.DataFrame or None, default: None
        First data table imported with the function 'import_info_lakes'.
    area : pd.DataFrame or None, default: None
        Second data table imported with the function 'import_info_lakes'.
    volume : pd.DataFrame or None, default: None
        Third data table imported with the function 'import_info_lakes'.
    """

    # Import data tables that are not passed to the function
    if chambers is None:
        chambers = import_chambers_data(
            '~/OneDrive/VM/Metlake/Data/METLAKE_ManualFluxChambers'
            '_DBsheet_2018-2020_final_withDBlakes.xlsx'
        )
    if water_conc is None:
        water_conc = create_water_concentration_table(chambers)
    if profiles is None:
        profiles = import_depth_profiles_data()
    if chem_avg is None:
        chem, chem_avg = import_water_chemistry_data_cleaned_manually()
    if info is None or area is None or volume is None:
        info_new, area_new, volume_new = import_info_lakes(
            '~/OneDrive/VM/Metlake/Data/METLAKE_InfoLakes_withDBlakes.xlsx',
            str_to_nan=True, only_bay_BD6=True, add_area_2008=True,
        )
        if info is None:
            info = info_new
        if area is None:
            area = area_new
            area = area.droplevel(1, axis=1)
        if volume is None:
            volume = volume_new

    # Rename columns in chambers data
    fc = chambers['ch4_flux'].copy()
    fc[('General', 'Lake', 'Unnamed: 0_level_2')].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    water_conc['Lake'].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    # Calculate average fluxes and average surface water concentrations
    create_deployment_groups_indices(
        fc,
        ('General', 'Lake', 'Unnamed: 0_level_2'),
        ('General', 'Initial sampling', 'Date and time'),
        ('General', 'Final sampling', 'Date and time'),
        ('General', 'Deployment group', '')
    )
    create_deployment_groups_indices(
        water_conc, 'Lake', 'Datetime', 'Datetime', 'Group', 6
    )
    avg_fluxes_groups = calculate_average_flux_per_deployment_group(fc, area)
    avg_fluxes_year = calculate_average_flux_per_lake_and_year(
        avg_fluxes_groups
    )
    avg_conc_groups_uM = \
            calculate_average_surface_concentration_per_deployment_group(
                water_conc, area, col_conc='CH4_[uM]'
            )
    avg_conc_groups_uM.rename({'lake': 'Lake'}, axis=1, inplace=True)
    avg_conc_year_uM = \
            calculate_average_surface_concentration_per_lake_and_year(
                avg_conc_groups_uM, 'CH4_[uM]'
            )
    avg_conc_groups_uatm = \
            calculate_average_surface_concentration_per_deployment_group(
                water_conc, area, col_conc='CH4_[uatm]'
            )
    avg_conc_groups_uatm.rename({'lake': 'Lake'}, axis=1, inplace=True)
    avg_conc_year_uatm = \
            calculate_average_surface_concentration_per_lake_and_year(
                avg_conc_groups_uatm, 'CH4_[uatm]'
            )
    # Reshape average chemistry data and add chemistry data for the three
    # lakes near Uppsala that were sampled in 2008
    cols = ['TOC_mg/L', 'DOC_mg/L', 'TN_mg/L', 'DN_mg/L', 'TP_ug/L',
            'Chla_ug/L', 'Chla_summer_ug/L']
    chem_avg = chem_avg.xs('surface', level=1)[cols].copy()
    chem_avg_OC = chem \
            .loc[chem["DOC_mg/L"]/chem["TOC_mg/L"] < 1.1, cols[:2]] \
            .loc[(slice(None), slice(None), 'surface')] \
            .groupby("Lake") \
            .mean()
    chem_avg[cols[:2]] = chem_avg_OC[cols[:2]]
    chem_avg.loc['LA', 'TOC_mg/L'] = 15.0
    chem_avg.loc['LO', 'TOC_mg/L'] = 12.1
    chem_avg.loc['ST', 'TOC_mg/L'] = 20.8
    chem_avg.loc['LA', 'TP_ug/L'] = 37.0
    chem_avg.loc['LO', 'TP_ug/L'] = 28.1
    chem_avg.loc['ST', 'TP_ug/L'] = 41.3
    chem_avg.index.name = 'Lake'
    # Import pH data
    ph = pd.read_csv('../Data/pH_AverageYearPerLake.csv', index_col=0)
    ph.loc['LJE', 'pH'] = np.nan
    ph.loc['LA', 'pH'] = np.nan
    ph.loc['LO', 'pH'] = np.nan
    ph.loc['ST', 'pH'] = np.nan
    # Import absorbance (420 nm) data
    _, absorbance = import_absorbance_data()
    absorbance = absorbance[420]
    absorbance.loc['LA'] = np.nan
    absorbance.loc['LO'] = np.nan
    absorbance.loc['ST'] = np.nan
    absorbance.index.name = 'Lake'
    # Rename index of the avg_fluxes_year table
    avg_fluxes_year.index.name = 'Lake'
    # Extract total volume of lakes
    # UPDATE: replaced with manually extracted values
    #volume_lakes = volume.droplevel(1, axis=1).iloc[0].T
    #volume_lakes.name = 'lake_volume'
    #volume_lakes.index.name = 'Lake'
    # Extract yearly average precipitation
    # UPDATE: replaced with manually extracted values
    #precip = calculate_water_retention_time_lakes_SVAR2016()
    #precip = precip[['Lake', 'Precipitation_[mm/year]']]
    # Calculate meteorological parameters for the lakes sampled in 2008
    # near Uppsala using SMHI weather station data but storing it in
    # the MESAN columns of the table because reference data for all other
    # lakes is from MESAN. The Uppsala SMHI weather station was not measuring
    # precipitation in 2008.
    m = [('Temperature_mean', 1), ('WindSpeed_mean', 4)]
    for param, code in m:
        d_smhi = SMHI_data.load_station_data(code, 97530, 'corrected-archive')
        cond = np.logical_and(
            d_smhi['Datetime'] >= datetime(2008, 1, 1, 0, 0),
            d_smhi['Datetime'] < datetime(2009, 1, 1, 0, 0)
        )
        d_smhi_mean = d_smhi.loc[cond, 'Value'].mean()
        for lake in ['LA', 'LO', 'ST']:
            info.loc[info['Lake'] == lake, f'{param}_MESAN'] = d_smhi_mean
    # Calculate aggregated land type and land use values
    cols_forest = [
        'Forest_NotOnWetland_Deciduous', 'Forest_OnWetland_Deciduous',
        'Forest_NotOnWetland_Evergreen', 'Forest_OnWetland_Evergreen',
        'Forest_NotOnWetland_Mixed', 'Forest_OnWetland_Mixed'
    ]
    cols_openfield = [
        'Forest_NotOnWetland_Clearcut', 'Forest_OnWetland_Clearcut',
        'OpenField_WithoutVegetation', 'OpenField_WithVegetation', 'Farmland'
    ]
    info['Forest'] = info[cols_forest].sum(axis=1)
    info['OpenField'] = info[cols_openfield].sum(axis=1)
    # Update some lake info about BD6
    bathy_BD6 = area['BD6'].dropna().values
    bathy_vol_BD6 = (bathy_BD6[1:] + bathy_BD6[:-1])/2*0.1
    bathy_cumvol_BD6 = np.cumsum(bathy_vol_BD6[::-1])[::-1]
    volume_BD6 = bathy_cumvol_BD6[0]
    bathy_cumvol_BD6 = pd.Series(bathy_cumvol_BD6,
                                 area['BD6'].dropna().index[:-1])
    bathy_cumvol_BD6[max(bathy_cumvol_BD6.index) + 0.1] = 0.0
    volume['BD6'] = bathy_cumvol_BD6
    bathy_BD6 = area['BD6'].diff(-1).dropna()
    mean_depth_BD6 = np.average(bathy_BD6.index, weights=bathy_BD6.values)
    max_depth_BD6 = area['BD6'].dropna().index.max() + 0.1
    info.loc[info['Lake'] == 'BD6', 'LakeShoreline_[m]'] = 800.0
    info.loc[info['Lake'] == 'BD6', 'LakeArea_[m2]'] = area['BD6'].loc[0]
    info.loc[info['Lake'] == 'BD6', 'LakeVolume_[m3]'] = volume_BD6
    info.loc[info['Lake'] == 'BD6', 'LakeDepth_Mean_[m]'] = mean_depth_BD6
    info.loc[info['Lake'] == 'BD6', 'LakeDepth_Max_[m]'] = max_depth_BD6
    # Calculate light attenuation coefficient in water
    conv_lakes_abs = {
        'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6',
        'Venasjon': 'VEN', 'Parsen': 'PAR', 'SodraTeden': 'SOD',
        'StoraGalten': 'SGA', 'Gundlebosjon': 'GUN', 'Grinnsjon': 'GRI',
        'LjusvattentjarnExperiment': 'LJE', 'LjusvattentjarnReference': 'LJR',
        'Nastjarn': 'NAS', 'NedreBjorntjarn': 'NBJ',
        'Dammsjon': 'DAM', 'Norrtjarn': 'NOR', 'Grastjarn': 'GRA',
        'Lammen': 'LAM', 'Klintsjon': 'KLI', 'Gyslattasjon': 'GYS'
    }
    attenuation = calculate_light_attenuation_from_profiles(profiles)
    attenuation['Lake'].replace(conv_lakes_abs, inplace=True)
    attenuation = attenuation.groupby('Lake')['Attenuation_m-1'].median()
    for lake in ['LA', 'LO', 'ST']:
        attenuation[lake] = np.nan
    # Calculate length of ice-free seasons and average daily incoming
    # shortwave radiation during the ice-free season
    # UPDATE: replaced ice-free season lengths with manually extracted values
    #icefree_period = pd.DataFrame(
    #    index=info['Lake'].values, columns=['icefree_period', 'SWavg'],
    #    dtype='float64'
    #)
    icefree_period = pd.DataFrame(
        index=info['Lake'].values, columns=['SWavg'], dtype='float64'
    )
    icefree_period.index.name = 'Lake'
    for n, row in info.iterrows():
        if row['Lake'] in ['LA', 'LO', 'ST']:
            icefree_period.loc[row['Lake'], 'SWavg'] = np.nan
            continue
        strang = pd.read_csv(os.path.join(
            '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake', 'Data',
            'STRANG', f'STRANG_{row["Lake"]}.csv'
        )).set_index('datetime')
        strang.index = pd.to_datetime(strang.index)
        strang_daily_mean = strang['value'].resample('D').mean()
        if isinstance(row['Open water season start'], str):
            IFstart = datetime(strang.index[0].year, 1, 1)
        else:
            IFstart = row['Open water season start']
        if isinstance(row['Open water season end'], str):
            IFend = datetime(strang.index[0].year + 1, 1, 1)
        elif row['Open water season end'].year > strang.index[0].year:
            IFend = datetime(strang.index[0].year + 1, 1, 1)
        else:
            IFend = row['Open water season end']
        IFperiod = (IFend - IFstart).total_seconds()/86400
        #icefree_period.loc[row['Lake'], 'icefree_period'] = IFperiod
        icefree_period.loc[row['Lake'], 'SWavg'] = strang_daily_mean.loc[
            np.logical_and(
                strang_daily_mean.index >= IFstart,
                strang_daily_mean.index <= IFend
            )
        ].mean()
    # Calculate maximum amount of CH4 stored in the water column
    storage = calculate_storage_and_VFAN_in_all_lakes_for_all_dates(
        profiles, volume
    )
    storage_max = pd.DataFrame(
        index=info['Lake'].values, columns=['storage', 'VFAN'],
        dtype='float64'
    )
    storage_max.index.name = 'Lake'
    for lake in info['Lake']:
        if lake in ['LJE', 'LA', 'LO', 'ST']:
            storage_max.loc[lake, 'storage'] = np.nan
            storage_max.loc[lake, 'VFAN'] = np.nan
        else:
            storage_lake = storage.loc[lake].dropna()
            if lake in ['BD3', 'BD4', 'BD6']:
                cond = storage_lake.index.map(lambda x: x.year == 2018)
                storage_lake = storage_lake[cond]
            row_max = storage_lake.iloc[np.argmax(storage_lake['storage'])]
            storage_max.loc[lake, 'storage'] = row_max['storage']
            storage_max.loc[lake, 'VFAN'] = row_max['anoxic volume fraction']

    # Merge all data together
    data_yearly_avg = pd.merge(avg_fluxes_year, avg_conc_year_uM, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, avg_conc_year_uatm, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, chem_avg, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, ph, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, absorbance, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, info, on='Lake')
    #data_yearly_avg = pd.merge(data_yearly_avg, volume_lakes, on='Lake')
    #data_yearly_avg = pd.merge(data_yearly_avg, precip, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, attenuation, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, icefree_period, on='Lake')
    data_yearly_avg = pd.merge(data_yearly_avg, storage_max, on='Lake')
    data_yearly_avg.drop(
        columns=[
            'diff+ebul_flux', 'North coordinate.1', 'East coordinate.1',
            'Open water season start', 'Open water season end',
            'SMHI_WeatherStation_ID'
        ],
        inplace=True
    )
    cols_new_names = [
        'Lake', 'deployment_length', 'k', 'k600', 'k_diff',
        'diff_flux', 'ebul_flux', 'tot_flux', 'CH4', 'pCH4',
        'TOC', 'DOC', 'TN', 'DN', 'TP', 'chla', 'chla_summer', 'pH', 'abs420',
        'Lake_LongName', 'latitude', 'longitude', 'altitude',
        'icefree_period', 'spring_ini', 'spring_fin', 'summer_ini',
        'summer_fin', 'autumn_ini', 'autumn_fin', 'winter_ini', 'winter_fin',
        'lake_perimeter', 'lake_area', 'lake_volume',
        'mean_depth', 'max_depth', 'catchment_area', 'catchment_perimeter',
        'residence_time', 'residence_time_avg', 'precipitation',
        'soildepth_mean', 'soildepth_max', 'soildepth_std',
        'elevation_mean', 'elevation_max', 'elevation_std',
        'elevationslope_deg_mean', 'elevationslope_deg_max',
        'elevationslope_deg_std', 'elevationslope_perc_mean',
        'elevationslope_perc_max', 'elevationslope_perc_std',
        'gpp_catchment',
        'soiltype_artificialfill', 'soiltype_bogpeat', 'soiltype_claytosilt',
        'soiltype_crystallinerock', 'soiltype_fenpeat',
        'soiltype_fineglacialclay', 'soiltype_glacialclay',
        'soiltype_glacialsilt', 'soiltype_glaciofluvialgravel',
        'soiltype_glaciofluvialsand', 'soiltype_glaciofluvialsediment',
        'soiltype_gravellytill', 'soiltype_gyttja', 'soiltype_gyttjaclay',
        'soiltype_peat', 'soiltype_phanerozoicdolerite',
        'soiltype_postglacialclay', 'soiltype_postglacialcoarseclay',
        'soiltype_postglacialfineclay', 'soiltype_postglacialfinesand',
        'soiltype_postglacialsand', 'soiltype_postglacialsilt',
        'soiltype_rock', 'soiltype_sandytill', 'soiltype_shingle',
        'soiltype_till', 'soiltype_unclassified', 'soiltype_water',
        'soiltype_wavewashedgravel', 'soiltype_youngfluvialsediment',
        'soiltype_youngfluvialsedimentclaytosilt',
        'landuse_buildings', 'landuse_farmland',
        'landuse_forest_notonwetland_clearcut',
        'landuse_forest_notonwetland_deciduous',
        'landuse_forest_notonwetland_evergreen',
        'landuse_forest_notonwetland_mixed',
        'landuse_forest_onwetland_clearcut',
        'landuse_forest_onwetland_deciduous',
        'landuse_forest_onwetland_evergreen',
        'landuse_forest_onwetland_mixed',
        'landuse_openwetland', 'landuse_openfield_withoutvegetation',
        'landuse_openfield_withvegetation', 'landuse_water',
        'bedrock_amphibolite', 'bedrock_basaltandesite',
        'bedrock_daciterhyolite', 'bedrock_dolerite', 'bedrock_gabbrodioritoid',
        'bedrock_granite', 'bedrock_granodioritegranite',
        'bedrock_granodioriticgraniticgneiss', 'bedrock_rhyolite',
        'bedrock_syenitoidgranite', 'bedrock_tonalitegranodiorite',
        'bedrock_ultrabasicintrusiverock', 'bedrock_wacke',
        'air_temp_smhi', 'wind_speed_smhi', 'precipitation_smhi',
        'air_temp_mesan', 'wind_speed_mesan','precipitation_mesan',
        'landuse_forest', 'landuse_openfield',
        'attenuation', 'SWavg', 'storage', 'VFAN'
    ]
    data_yearly_avg.columns = cols_new_names
    data_yearly_avg_units = [
        '', 'day', 'm d-1', 'm d-1', 'm d-1',
        'mmol m-2 d-1', 'mmol m-2 d-1', 'mmol m-2 d-1', 'µM', 'µatm',
        'mg L-1', 'mg L-1', 'mg L-1', 'mg L-1',
        'µg L-1', 'µg L-1', 'µg L-1', '-', 'cm-1',
        '', 'decimal degree', 'decimal degree', 'm',
        'day', '', '', '', '', '', '', '', '',
        'm', 'm2', 'm3', 'm', 'm', 'm2', 'm',
        'day', 'day', 'mm year-1',
        'm', 'm', 'm', 'm', 'm', 'm', 'deg', 'deg', 'deg',
        '%', '%', '%', 'g C m-2 year-1',
        '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
        '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
        '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
        '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
        '-', '-',
        'oC', 'm s-1', 'mm year-1', 'oC', 'm s-1', 'mm year-1', '-', '-',
        'm-1', 'W m-2', 'mol', '-'
    ]
    data_yearly_avg_units = pd.Series(
        index=data_yearly_avg.columns, data=data_yearly_avg_units
    )

    return data_yearly_avg, data_yearly_avg_units


def combine_all_weather_data(lake):
    """
    Combine weather station, anemometer and MESAN data available for a lake.

    Parameters
    ----------
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake of interest.
    """

    meteo_match = {
        'BD3': 'BD6', 'BD4': 'BD6', 'BD6': 'BD6', 'SGA': 'SGA', 'GUN': 'SGA',
        'GRI': 'SGA', 'LJE': 'NBJ', 'LJR': 'NBJ', 'NAS': 'NBJ', 'NBJ': 'NBJ',
        'DAM': 'NOR', 'NOR': 'NOR', 'GRA': 'NOR', 'LAM': 'GYS', 'GYS': 'GYS',
        'KLI': 'GYS'
    }
    anemo_match = {
        'BD3': 'BD3', 'BD4': 'BD4', 'BD6': 'BD6', 'SGA': 'SGA', 'GUN': 'GUN',
        'GRI': 'GRI', 'LJE': 'LJR', 'LJR': 'LJR', 'NAS': 'NBJ', 'NBJ': 'NBJ',
        'DAM': 'DAM', 'NOR': 'GRA', 'GRA': 'GRA', 'LAM': 'LAM', 'GYS': 'GYS',
        'KLI': 'KLI', 'PAR': 'PAR', 'VEN': 'PAR', 'SOD': 'PAR'
    }
    mesan_conv = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExp', 'LJR': 'LjusvattentjarnRef',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }

    entire_periods = {
        'BD3': (datetime(2018, 6, 4, 16, 17), datetime(2018, 9, 16, 22, 33)),
        'BD4': (datetime(2018, 6, 4, 16, 17), datetime(2018, 9, 16, 22, 33)),
        'BD6': (datetime(2018, 6, 4, 16, 17), datetime(2018, 9, 16, 22, 33)),
        'VEN': (datetime(2018, 5, 7, 0, 0), datetime(2018, 12, 6, 21, 0)),
        'PAR': (datetime(2018, 5, 7, 0, 0), datetime(2018, 12, 6, 21, 0)),
        'SOD': (datetime(2018, 5, 7, 0, 0), datetime(2018, 12, 6, 21, 0)),
        'SGA': (datetime(2019, 3, 20, 19, 48), datetime(2019, 12, 3, 0, 55)),
        'GUN': (datetime(2019, 3, 19, 15, 52), datetime(2019, 12, 11, 15, 12)),
        'GRI': (datetime(2019, 3, 20, 19, 48), datetime(2019, 12, 11, 17, 1)),
        'LJE': (datetime(2019, 5, 24, 13, 18), datetime(2019, 10, 27, 16, 21)),
        'LJR': (datetime(2019, 5, 24, 13, 18), datetime(2019, 10, 27, 16, 21)),
        'NBJ': (datetime(2019, 6, 4, 20, 10), datetime(2019, 10, 27, 17, 56)),
        'NAS': (datetime(2019, 6, 4, 20, 10), datetime(2019, 10, 27, 17, 56)),
        'DAM': (datetime(2020, 4, 4, 14, 20), datetime(2020, 11, 7, 10, 14)),
        'NOR': (datetime(2020, 4, 4, 14, 20), datetime(2020, 11, 6, 16, 11)),
        'GRA': (datetime(2020, 4, 4, 14, 20), datetime(2020, 11, 6, 16, 11)),
        'LAM': (datetime(2020, 4, 23, 16, 9), datetime(2020, 11, 21, 12, 39)),
        'GYS': (datetime(2020, 4, 23, 13, 10), datetime(2020, 11, 21, 12, 39)),
        'KLI': (datetime(2020, 4, 21, 20, 57), datetime(2020, 11, 21, 12, 39))
    }
    gaps_meteo = {
        'BD3': [
            (datetime(2018, 6, 5, 22, 26), datetime(2018, 6, 25, 17, 19)),
            (datetime(2018, 8, 17, 1, 33), datetime(2018, 9, 12, 15, 10))
        ],
        'BD4': [
            (datetime(2018, 6, 5, 22, 26), datetime(2018, 6, 25, 17, 19)),
            (datetime(2018, 8, 17, 1, 33), datetime(2018, 9, 12, 15, 10))
        ],
        'BD6': [
            (datetime(2018, 6, 5, 22, 26), datetime(2018, 6, 25, 17, 19)),
            (datetime(2018, 8, 17, 1, 33), datetime(2018, 9, 12, 15, 10))
        ],
        'VEN': [
            (datetime(2018, 6, 15, 20, 7), datetime(2018, 9, 15, 17, 49)),
            (datetime(2018, 10, 17, 15, 24), datetime(2018, 10, 25, 12, 8))
        ],
        'PAR': [
            (datetime(2018, 6, 15, 20, 7), datetime(2018, 9, 15, 17, 49)),
            (datetime(2018, 10, 17, 15, 24), datetime(2018, 10, 25, 12, 8))
        ],
        'SOD': [
            (datetime(2018, 6, 15, 20, 7), datetime(2018, 9, 15, 17, 49)),
            (datetime(2018, 10, 17, 15, 24), datetime(2018, 10, 25, 12, 8))
        ],
        'SGA': [
            (datetime(2019, 3, 21, 21, 35), datetime(2019, 3, 22, 14, 56)),
            (datetime(2019, 7, 3, 19, 50), datetime(2019, 7, 13, 21, 8)),
            (datetime(2019, 9, 18, 15, 25), datetime(2019, 10, 7, 10, 21)),
            (datetime(2019, 10, 15, 1, 39), datetime(2019, 11, 4, 10, 31)),
            (datetime(2019, 11, 11, 20, 19), datetime(2019, 11, 27, 10, 36))
        ],
        'GUN': [
            (datetime(2000, 1, 1, 0, 0), datetime(2019, 3, 20, 19, 48)),
            (datetime(2019, 3, 21, 21, 35), datetime(2019, 3, 22, 14, 56)),
            (datetime(2019, 7, 3, 19, 50), datetime(2019, 7, 13, 21, 8)),
            (datetime(2019, 9, 18, 15, 25), datetime(2019, 10, 7, 10, 21)),
            (datetime(2019, 10, 15, 1, 39), datetime(2019, 11, 4, 10, 31)),
            (datetime(2019, 11, 11, 20, 19), datetime(2019, 11, 27, 10, 36)),
            (datetime(2019, 12, 3, 0, 55), datetime(2030, 1, 1, 0, 0))
        ],
        'GRI': [
            (datetime(2019, 3, 21, 21, 35), datetime(2019, 3, 22, 14, 56)),
            (datetime(2019, 7, 3, 19, 50), datetime(2019, 7, 13, 21, 8)),
            (datetime(2019, 9, 18, 15, 25), datetime(2019, 10, 7, 10, 21)),
            (datetime(2019, 10, 15, 1, 39), datetime(2019, 11, 4, 10, 31)),
            (datetime(2019, 11, 11, 20, 19), datetime(2019, 11, 27, 10, 36)),
            (datetime(2019, 12, 3, 0, 55), datetime(2030, 1, 1, 0, 0))
        ],
        'LJE': [(datetime(2019, 10, 15, 17, 53), datetime(2030, 1, 1, 0, 0))],
        'LJR': [(datetime(2019, 10, 15, 17, 53), datetime(2030, 1, 1, 0, 0))],
        'NBJ': [
            (datetime(2000, 1, 1, 0, 0), datetime(2019, 7, 3, 19, 11)),
            (datetime(2019, 9, 11, 2, 34), datetime(2019, 9, 26, 10, 34)),
            (datetime(2019, 10, 4, 22, 11), datetime(2019, 10, 5, 9, 35)),
            (datetime(2019, 10, 6, 6, 14), datetime(2019, 10, 6, 9, 10)),
            (datetime(2019, 10, 6, 18, 45), datetime(2019, 10, 7, 10, 32)),
            (datetime(2019, 10, 7, 18, 4), datetime(2019, 10, 9, 10, 3)),
            (datetime(2019, 10, 9, 16, 25), datetime(2019, 10, 14, 11, 10)),
            (datetime(2019, 10, 14, 22, 24), datetime(2019, 10, 15, 10, 16)),
            (datetime(2019, 10, 15, 16, 1), datetime(2019, 10, 22, 11, 45))
        ],
        'NAS': [
            (datetime(2000, 1, 1, 0, 0), datetime(2019, 7, 3, 19, 11)),
            (datetime(2019, 9, 11, 2, 34), datetime(2019, 9, 26, 10, 34)),
            (datetime(2019, 10, 4, 22, 11), datetime(2019, 10, 5, 9, 35)),
            (datetime(2019, 10, 6, 6, 14), datetime(2019, 10, 6, 9, 10)),
            (datetime(2019, 10, 6, 18, 45), datetime(2019, 10, 7, 10, 32)),
            (datetime(2019, 10, 7, 18, 4), datetime(2019, 10, 9, 10, 3)),
            (datetime(2019, 10, 9, 16, 25), datetime(2019, 10, 14, 11, 10)),
            (datetime(2019, 10, 14, 22, 24), datetime(2019, 10, 15, 10, 16)),
            (datetime(2019, 10, 15, 16, 1), datetime(2019, 10, 22, 11, 45))
        ],
        'DAM': [
            (datetime(2020, 4, 15, 19, 22), datetime(2020, 5, 12, 14, 50)),
            (datetime(2020, 9, 16, 2, 44), datetime(2020, 11, 2, 15, 59)),
            (datetime(2020, 11, 4, 23, 28), datetime(2020, 11, 5, 12, 26)),
            (datetime(2020, 11, 5, 23, 47), datetime(2030, 1, 1, 0, 0))
        ],
        'NOR': [
            (datetime(2020, 4, 15, 19, 22), datetime(2020, 5, 12, 14, 40)),
            (datetime(2020, 9, 16, 2, 44), datetime(2020, 11, 2, 15, 59)),
            (datetime(2020, 11, 4, 23, 28), datetime(2020, 11, 5, 12, 26)),
            (datetime(2020, 11, 5, 23, 47), datetime(2030, 1, 1, 0, 0))
        ],
        'GRA': [
            (datetime(2020, 4, 15, 19, 22), datetime(2020, 5, 12, 14, 40)),
            (datetime(2020, 9, 16, 2, 44), datetime(2020, 11, 2, 15, 59)),
            (datetime(2020, 11, 4, 23, 28), datetime(2020, 11, 5, 12, 26)),
            (datetime(2020, 11, 5, 23, 47), datetime(2030, 1, 1, 0, 0))
        ],
        'LAM': [
            (datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 26, 19, 38)),
            (datetime(2020, 10, 3, 22, 8), datetime(2020, 10, 12, 12, 2)),
            (datetime(2020, 10, 21, 14, 14), datetime(2020, 11, 16, 13, 21)),
            (datetime(2020, 11, 21, 12, 39), datetime(2030, 1, 1, 0, 0))
        ],
        'GYS': [
            (datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 26, 19, 38)),
            (datetime(2020, 10, 3, 22, 8), datetime(2020, 10, 12, 12, 2)),
            (datetime(2020, 10, 21, 14, 14), datetime(2020, 11, 16, 13, 21)),
        ],
        'KLI': [
            (datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 26, 19, 38)),
            (datetime(2020, 10, 3, 22, 8), datetime(2020, 10, 12, 12, 2)),
            (datetime(2020, 10, 21, 14, 14), datetime(2020, 11, 16, 13, 21)),
        ]
    }
    gaps_anemo = {
        'BD3': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 7, 17, 15, 52)),
            (datetime(2018, 9, 12, 17, 47), datetime(2030, 1, 1, 0, 0))
        ],
        'BD4': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 7, 4, 18, 45)),
            (datetime(2018, 9, 13, 11, 55), datetime(2030, 1, 1, 0, 0))
        ],
        'BD6': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 7, 4, 23, 42)),
            (datetime(2018, 9, 12, 17, 37), datetime(2030, 1, 1, 0, 0))
        ],
        'VEN': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 9, 27, 19, 56)),
            (datetime(2018, 12, 6, 17, 59), datetime(2030, 1, 1, 0, 0))
        ],
        'PAR': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 9, 27, 19, 56)),
            (datetime(2018, 12, 6, 17, 59), datetime(2030, 1, 1, 0, 0))
        ],
        'SOD': [
            (datetime(2000, 1, 1, 0, 0), datetime(2018, 9, 27, 19, 56)),
            (datetime(2018, 12, 6, 17, 59), datetime(2030, 1, 1, 0, 0))
        ],
        'SGA': [
            (datetime(2000, 1, 1, 0, 0), datetime(2019, 3, 22, 13, 30)),
            (datetime(2019, 5, 27, 11, 51), datetime(2030, 1, 1, 0, 0))
        ],
        'GUN': [(datetime(2019, 5, 17, 14, 32), datetime(2019, 5, 27, 16, 32))],
        'GRI': [
            (datetime(2000, 1, 1, 0, 0), datetime(2019, 3, 21, 13, 44)),
            (datetime(2019, 11, 13, 2, 58), datetime(2019, 11, 28, 11, 17))
        ],
        'LJE': [(datetime(2000, 1, 1, 0, 0), datetime(2019, 9, 2, 19, 31))],
        'LJR': [(datetime(2000, 1, 1, 0, 0), datetime(2019, 9, 2, 19, 31))],
        'NBJ': [
            (datetime(2019, 10, 1, 14, 46), datetime(2019, 10, 1, 18, 28)),
            (datetime(2019, 10, 26, 20, 57), datetime(2030, 1, 1, 0, 0))
        ],
        'NAS': [
            (datetime(2019, 10, 1, 14, 46), datetime(2019, 10, 1, 18, 28)),
            (datetime(2019, 10, 26, 20, 57), datetime(2030, 1, 1, 0, 0))
        ],
        'DAM': [
            (datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 12, 11, 0)),
            (datetime(2020, 5, 14, 11, 54), datetime(2020, 6, 8, 13, 48))
        ],
        'NOR': [(datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 21, 9, 6))],
        'GRA': [(datetime(2000, 1, 1, 0, 0), datetime(2020, 5, 21, 9, 6))],
        'LAM': [
            (datetime(2020, 6, 6, 2, 47), datetime(2020, 6, 23, 10, 11)),
            (datetime(2020, 11, 20, 18, 29), datetime(2030, 1, 1, 0, 0))
        ],
        'GYS': [(datetime(2020, 11, 20, 11, 10), datetime(2030, 1, 1, 0, 0))],
        'KLI': [(datetime(2020, 11, 20, 14, 2), datetime(2030, 1, 1, 0, 0))]
    }

    # Import anemometer data
    anemo = import_merged_anemometer_data()[anemo_match[lake]]
    cos_dir = np.cos(anemo['Wind Direction, ø']*np.pi/180)
    sin_dir = np.sin(anemo['Wind Direction, ø']*np.pi/180)
    anemo['u_anemo'] = anemo['Wind Speed, m/s']*cos_dir
    anemo['v_anemo'] = anemo['Wind Speed, m/s']*sin_dir
    anemo = anemo.set_index('Date Time, GMT+02:00').resample('T').mean()
    anemo = anemo.interpolate('time')
    anemo['dir_anemo'] = \
            (np.arctan2(anemo['v_anemo'], anemo['u_anemo'])*180/np.pi)%360
    for gap in gaps_anemo[lake]:
        cond = np.logical_and(anemo.index > gap[0], anemo.index < gap[1])
        anemo.loc[cond, :] = np.nan

    # Import weather station data
    if lake in ['VEN', 'PAR', 'SOD']:
        # Using the HOBO weather station that was deployed in Parsen.
        # Logging frequency was 5 minutes in the last months, so it is
        # interpolated to a one-minute frequency.
        # The rain should not be interpolated, as it is measured as a quantity,
        # not an intensity.
        meteo = import_hobo_weather_station_data('PAR')
        meteo.set_index('Date Time, GMT+02:00', inplace=True)
        cos_dir = np.cos(meteo['Wind Direction, ø']*np.pi/180)
        sin_dir = np.sin(meteo['Wind Direction, ø']*np.pi/180)
        meteo['u_meteo'] = meteo['Wind Speed, m/s']*cos_dir
        meteo['v_meteo'] = meteo['Wind Speed, m/s']*sin_dir
        meteo = meteo.resample('T').mean()
        meteo_sum = meteo.resample('T').sum()
        rain = meteo_sum['Rain, mm']
        meteo = meteo.interpolate('time')
        meteo['Rain, mm'] = rain
        meteo['dir_meteo'] = \
                (np.arctan2(meteo['v_meteo'], meteo['u_meteo'])*180/np.pi)%360
        meteo.rename(
            columns={
                'Temp, °C': 'temperature',
                'Pressure, mbar': 'pressure',
                'Wind Speed, m/s': 'wind_speed',
                'Gust Speed, m/s': 'gust_speed',
                'Wind Direction, ø': 'wind_dir',
                'Rain, mm': 'precipitation_intensity',
                'PAR, µmol/m²/s': 'par'
            }, inplace=True
        )
        meteo.drop(columns=['#'], inplace=True)
        # Convert PAR to total solar irradiance
        #   - 0.22 is the conversion factor from µmol/m²/s to W/m². It is
        #     calculated as the average between two different estimates
        #     based on the Planck-Einstein relation (E = h*c/lambda). One
        #     estimate uses lambda = 550 nm and the other estimate integrates
        #     the Planck-Einstein relation between 400 nm and 700 nm and
        #     divides the result by the range (300 nm). The values obtained
        #     are 0.2175 and 0.2231, respectively.
        #   - 0.43 is the conversion factor from PAR to total solar irradiance
        #     It is calculated as explained in the documentation of the function
        #     'metlake_utils.solar_irradiance_in_range'.
        meteo['par'] = 0.22*meteo['par']
        meteo['solar_radiation'] = meteo['par']/0.43
        # Add a column for relative humidity for completeness
        meteo['humidity'] = np.nan
        for gap in gaps_meteo[lake]:
            cond = np.logical_and(meteo.index > gap[0], meteo.index < gap[1])
            meteo.loc[cond, :] = np.nan
        data = pd.merge(meteo, anemo, on='Date Time, GMT+02:00', how='outer')
    elif lake in ['LJE', 'LJR']:
        # Using the Umea University weather station located near LJT and adding
        # total radiation from the GMX531 weather station located near NBJ.
        # The logging frequency of the UmU weather station was 5 minutes,
        # so it is interpolated to a one-minute frequency.
        # The rain should not be interpolated as it is measured as a quantity,
        # not an intensity.
        meteo = import_hobo_weather_station_data('LJT')
        meteo.set_index('Date Time, GMT+02:00', inplace=True)
        cos_dir = np.cos(meteo['Wind Direction, ø']*np.pi/180)
        sin_dir = np.sin(meteo['Wind Direction, ø']*np.pi/180)
        meteo['u_meteo'] = meteo['Wind Speed, m/s']*cos_dir
        meteo['v_meteo'] = meteo['Wind Speed, m/s']*sin_dir
        meteo = meteo.resample('T').mean()
        meteo_sum = meteo.resample('T').sum()
        rain = meteo_sum['Rain, mm']
        meteo = meteo.interpolate('time')
        meteo['Rain, mm'] = rain
        meteo['dir_meteo'] = \
                (np.arctan2(meteo['v_meteo'], meteo['u_meteo'])*180/np.pi)%360
        meteo.rename(
            columns={
                'Temp, °C': 'temperature',
                'Pressure, mbar': 'pressure',
                'Wind Speed, m/s': 'wind_speed',
                'Gust Speed, m/s': 'gust_speed',
                'Wind Direction, ø': 'wind_dir',
                'Rain, mm': 'precipitation_intensity',
                'PAR, µmol/m²/s': 'par',
                'RH, %': 'humidity'
            }, inplace=True
        )
        meteo.drop(columns=['#'], inplace=True)
        meteo['par'] = 0.22*meteo['par']
        sol_rad = import_gmx531_data()[meteo_match[lake]]
        sol_rad = sol_rad[['Datetime', 'solar_radiation']].set_index('Datetime')
        sol_rad.index.name = 'Date Time, GMT+02:00'
        for gap in gaps_meteo[lake]:
            cond = np.logical_and(meteo.index > gap[0], meteo.index < gap[1])
            meteo.loc[cond, :] = np.nan
        for gap in gaps_meteo[meteo_match[lake]]:
            cond = np.logical_and(
                sol_rad.index > gap[0], sol_rad.index < gap[1]
            )
            sol_rad.loc[cond, 'solar_radiation'] = np.nan
        data = pd.merge(meteo, anemo, on='Date Time, GMT+02:00', how='outer')
        data = pd.merge(data, sol_rad, on='Date Time, GMT+02:00', how='outer')
    else:
        meteo = import_gmx531_data()[meteo_match[lake]]
        meteo = meteo[[
            'Datetime', 'temperature', 'pressure', 'wind_speed', 'gust_speed',
            'wind_dir', 'precipitation_intensity', 'humidity', 'solar_radiation'
        ]]
        cos_dir = np.cos(meteo['wind_dir']*np.pi/180)
        sin_dir = np.sin(meteo['wind_dir']*np.pi/180)
        meteo['u_meteo'] = meteo['wind_speed']*cos_dir
        meteo['v_meteo'] = meteo['wind_speed']*sin_dir
        meteo['dir_meteo'] = \
                (np.arctan2(meteo['v_meteo'], meteo['u_meteo'])*180/np.pi)%360
        meteo['par'] = np.nan
        for gap in gaps_meteo[lake]:
            cond = np.logical_and(
                meteo['Datetime'] > gap[0], meteo['Datetime'] < gap[1]
            )
            meteo.loc[cond, meteo.columns.drop('Datetime')] = np.nan
        meteo = meteo.set_index('Datetime')
        meteo.index.name = 'Date Time, GMT+02:00'
        data = pd.merge(meteo, anemo, on='Date Time, GMT+02:00', how='outer')

    # Use MESAN data to calculate incoming longwave radiation
    mesan = import_mesan_data(lake=mesan_conv[lake])
    lw_clear, lw_tot = calculate_LW_radiation(
        mesan['Temperature'], mesan['Relative humidity'],
        mesan['tcc']*mesan['c_sigfr']*100, mesan['cb_sig_b']
    )
    lw_tot = lw_tot.resample('T').mean().interpolate('time')
    lw_tot.index = lw_tot.index + timedelta(hours=2)
    lw_tot.index.name = 'Date Time, GMT+02:00'
    lw_tot.name = 'lw_radiation'
    data = pd.merge(data, lw_tot, on='Date Time, GMT+02:00', how='outer')

    # Fill gaps in wind speed, wind gust and wind direction when anemometer
    # data is missing but weather station data is available
    cond_gaps_anemo = np.zeros(data.shape[0], bool)
    for gap in gaps_anemo[lake]:
        cond = np.logical_and(data.index > gap[0], data.index < gap[1])
        cond_gaps_anemo[cond] = True
    data[['u', 'v', 'dir']] = data[['u_anemo', 'v_anemo', 'dir_anemo']]
    data.loc[cond_gaps_anemo, ['u', 'v', 'dir']] = \
            data.loc[cond_gaps_anemo, ['u_meteo', 'v_meteo', 'dir_meteo']]

    # Sort table according to time in index
    data.sort_index(inplace=True)

    return data


def create_final_data_table_for_empirical_models():
    """
    Create a data table with yearly averaged data to be used in empirical
    models. The final data table includes data from three lakes near Uppsala
    that were sampled in 2008.
    """

    # Set paths to some files
    path_chambers = ('~/OneDrive/VM/Metlake/Data/'
                     'METLAKE_ManualFluxChambers_DBsheet_'
                     '2018-2020_final_withDBlakes.xlsx')
    path_info = ('~/OneDrive/VM/Metlake/Data/'
                 'METLAKE_InfoLakes_withDBlakes.xlsx')

    # Load data files containing information about lakes, chambers, and MESAN
    chambers = import_chambers_data(path_chambers)
    info, area, volume = import_info_lakes(
        path_info, str_to_nan=True, only_bay_BD6=True, add_area_2008=True
    )
    mesan = import_mesan_data()

    # Modify the columns of the data table containing lake bathymetry data
    area = area.droplevel(1, axis=1)

    # Create two new data tables. The first one combines data from several
    # sources. The second one contains water concentration data.
    fc = combine_all_data_at_chamber_deployment_scale(chambers)
    water_conc = create_water_concentration_table(chambers)

    # Modify the data table containing water concentration data
    water_conc['Lake'] = water_conc['Lake'].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}
    )
    create_deployment_groups_indices(
        water_conc, 'Lake', 'Datetime', 'Datetime', 'Group', 1
    )

    # Create a new data table containing estimated daily average surface water
    # temperatures for each lake
    T_lakes = calculate_average_daily_water_temperature_lakes(mesan)

    # Create new data tables containing flux data averaged per deployment
    # and per year
    fc_avg = calculate_average_flux_per_deployment_group(
        fc, area, False, 'group', 'lake', 'dt_ini', 'dt_fin',
        'depth_ini', 'depth_fin', 'CH4_diff_flux', 'CH4_ebul_flux',
        'CH4_tot_flux', 'T_wat_mean', 'k', 'k600', 'k600_min', 'k_diff'
    )
    fc_avg_year = \
            calculate_average_flux_or_concentration_per_lake_and_year_using_temperature(
                fc_avg, T_lakes, ['diff_flux', 'ebul_flux', 'tot_flux']
            )

    # Create new data tables containing surface water concentration data
    # averaged per deployment and per year
    water_conc_avg = \
            calculate_average_surface_concentration_per_deployment_group(
                water_conc, area, False, 'Group', 'Lake', 'Datetime',
                'Depth_[m]', 'Twat_[oC]', 'CH4_[uM]'
            )
    water_conc_avg_year = \
            calculate_average_flux_or_concentration_per_lake_and_year_using_temperature(
                water_conc_avg, T_lakes, ['CH4_[uM]']
            )

    # Create a data table containing yearly averaged data from several sources
    all_variables_avg_year, units = combine_all_yearly_average_data(
        chambers, water_conc
    )

    # Modify some of the data tables
    water_conc_avg_year.rename({'CH4_[uM]': 'CH4'}, axis=1, inplace=True)
    all_variables_avg_year.rename(
        {'k': 'k_old', 'k600': 'k600_old', 'k_diff': 'k_diff_old',
         'diff_flux': 'diff_flux_old', 'ebul_flux': 'ebul_flux_old',
         'tot_flux': 'tot_flux_old', 'CH4': 'CH4_old', 'pCH4': 'pCH4_old'},
        axis=1, inplace=True
    )
    units.rename(
        {'k': 'k_old', 'k600': 'k600_old', 'k_diff': 'k_diff_old',
         'diff_flux': 'diff_flux_old', 'ebul_flux': 'ebul_flux_old',
         'tot_flux': 'tot_flux_old', 'CH4': 'CH4_old', 'pCH4': 'pCH4_old'},
        inplace=True
    )

    # Create a data table containing all relevant yearly averaged data
    d = all_variables_avg_year.set_index('Lake')\
            .merge(fc_avg_year, left_index=True, right_index=True)\
            .merge(water_conc_avg_year, left_index=True, right_index=True)
    d_units = units.copy()

    # Create new variables
    d['drainage_ratio'] = d['catchment_area']/d['lake_area']
    d['depth_ratio'] = d['mean_depth']/d['max_depth']
    d_units.drop('Lake', inplace=True)
    d_units['CH4'] = d_units['CH4_old']
    d_units['diff_flux'] = d_units['diff_flux_old']
    d_units['ebul_flux'] = d_units['ebul_flux_old']
    d_units['tot_flux'] = d_units['tot_flux_old']
    d_units['depth_ratio'] = '-'
    d_units['drainage_ratio'] = '-'

    # Create a dictionary containing all data tables created here
    results = {
        'chambers': chambers, 'info': info, 'area': area, 'volume': volume,
        'fc': fc, 'water_conc': water_conc, 'mesan': mesan, 'T_lakes': T_lakes,
        'fc_avg': fc_avg, 'fc_avg_year': fc_avg_year,
        'water_conc_avg': water_conc_avg,
        'water_conc_avg_year': water_conc_avg_year,
        'all_variables_avg_year': all_variables_avg_year, 'units': units,
        'data_final': d, 'units_data_final': d_units
    }

    return results


def curve_fit_with_diagnostics(f, xdata, ydata, CI=0.95):
    """
    Fit a function on some data using the function 'scipy.optimize.curve_fit'
    and return the results with additional diagnostics information.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data.
    CI : float
        Confidence interval to be returned for the parameters of the model.
    """

    nobs = xdata.shape[-1]
    npar = len(signature(f).parameters) - 1
    df = nobs - npar
    popt, pcov = curve_fit(f, xdata, ydata)
    ypred = f(xdata, *popt)
    shapiro_stat, shapiro_p = stats.shapiro(ydata - ypred)
    SSres = np.sum(np.square(ydata - ypred))
    SStot = np.sum(np.square(ydata - ydata.mean()))
    r2 = 1.0 - SSres/SStot
    r2_adj = 1.0 - (nobs - 1)/(nobs - npar)*(1.0 - r2)
    rmse = np.sqrt(np.mean(np.square(ypred - ydata)))
    rmse_rel = rmse/np.abs(np.mean(ydata))
    sigma = np.sqrt(np.diag(pcov))
    tvalues = popt/sigma
    pvalues = 2.0*(1.0 - stats.t(df).cdf(np.abs(tvalues)))
    lower = popt - stats.t(df).interval(CI)[1]*sigma
    upper = popt + stats.t(df).interval(CI)[1]*sigma
    results = pd.Series({
        'popt': popt, 'pcov': pcov, 'N': nobs, 'R2': r2, 'R2_adj': r2_adj,
        'RMSE': rmse, 'RMSErel': rmse_rel,
        'sigma': sigma, 'tvalues': tvalues, 'pvalues': pvalues,
        'lower': lower, 'upper': upper,
        'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p
    })

    return results


def test_curve_fit_with_diagnostics(f, N, Nmin, Nmax, e, *params):
    """
    Test the 'curve_fit_with_diagnostics' function.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).
    N : int
        Number of observations.
    Nmin : float
        Minimum value in the observations
    Nmax : float
        Maximum value in the observations
    e : float
        Amplitude of the normally distributed error term.
    *params : float
        True values of the parameters of 'f'.
    """

    fig, ax = plt.subplots()

    x = np.linspace(Nmin, Nmax, N)
    y = f(x, *params) + e*np.random.randn(N)
    ax.scatter(x, y, c='k')
    res = curve_fit_with_diagnostics(f, x, y)
    ax.plot(x, f(x, *params), c='k', label='True')
    ax.plot(x, f(x, *res['popt']), c='r', label='Model')
    ax.legend()

    return fig, ax, res


def fit_model_on_data_using_curve_fit(
    f, data, y_var, X_var, N_min=5, CI=0.95, f_shuffle=itertools.permutations
):
    """
    Fit a model on some data using the function 'curve_fit_with_diagnostics'.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).
    data : pd.DataFrame
        Table containing all relevant data. Only numeric columns will be used.
    y_var : string
        Column in 'data' containing the dependent variable.
    X_var : int, list, string
        If an integer is given, indicate the number of variables from 'data'
        to use in the model. The model is fitted on all combinations
        of variables that are possible.
        If a list is given, indicate the columns in 'data' to be
        used as independent variables.
        If a string is given, indicate the column in 'data' to be
        used as independent variable.
    N_min : int
        Minimum number of observations to have in each subgroup of data
        for performing the calculations when X_var is a string.
    CI : float
        Confidence interval to be returned for the parameters of the model.
    f_shuffle : {itertools.permutations, itertools.combinations},
                default: itertools.permutations
        Used only if 'X_var' is an integer. Determine if the order is
        accounted for (permutations) or not (combinations) when
        shuffling the columns.
    """

    if isinstance(X_var, int):
        cols = [
            col for col in data.columns
            if data.dtypes[col] in ['float64', 'int64'] and col != y_var
        ]
        results = []
        n_iter = len(list(f_shuffle(cols, X_var)))
        for variables in tqdm(f_shuffle(cols, X_var), total=n_iter):
            d = data[list(variables) + [y_var]].dropna()
            if d.shape[0] < N_min:
                continue
            if len(variables) == 1:
                X = d[variables[0]].values
            else:
                X = d[list(variables)].values.T
            y = d[y_var].values
            res = curve_fit_with_diagnostics(f, X, y, CI)
            res = pd.concat([pd.Series({'variables': variables}), res])
            results.append(res)
        results = pd.concat(results, axis=1).T
        return results
    elif isinstance(X_var, list):
        data = data[X_var + [y_var]].dropna()
    elif isinstance(X_var, str):
        data = data[[X_var] + [y_var]].dropna()

    X = data[X_var].values.T
    y = data[y_var].values
    results = curve_fit_with_diagnostics(f, X, y, CI)

    return results


def fit_linear_model_on_data_using_statsmodels(
    data, endog_var, n_exog_var=2, mode='+', N_min=5, p_thld_model=1.0,
    p_thld_params=1.0, ignore_p_intercept=False,
    f_shuffle=itertools.permutations
):
    """
    Calculate linear regression for all possible combinations of variables
    in a data table.

    Parameters
    ----------
    data : pd.DataFrame
        Table containing all relevant data. Only numeric columns will be used.
    endog_var : str
        Column in 'data' to use as endogenous (dependent) variable.
    n_exog_var : int, default: 2
        Number of exogenous (independent) variables to use. Only columns with
        dtype 'float64' and 'int64' will be considered.
    mode : {'+', '*'}, default: '+'
        If using '+', no interaction between variables will be considered.
        If using '*', interactions between variables will be considered.
    N_min : int
        Minimum number of observations to have in each subgroup of data
        for performing the calculations.
    p_thld_model : float, default: 1.0
        Only return models whose 'global' p-value is below this threshold.
    p_thld_params : float, default: 1.0
        Only return models where the p-value of all parameters is below
        this threshold.
    ignore_p_intercept : bool, default: False
        If True, the p-value of the intercept is not considered in
        the comparison with 'p_thld_params'.
    f_shuffle : {itertools.permutations, itertools.combinations},
                default: itertools.permutations
        Determine if the order is accounted for (permutations)
        or not (combinations) when shuffling the columns.
    """

    cols = [
        col for col in data.columns
        if data.dtypes[col] in ['float64', 'int64'] and col != endog_var
    ]

    cols_res = [f'var{n}' for n in range(1, n_exog_var + 1)] + \
            ['params', 'N', 'R2', 'p-value', 'p-values', 'model']

    results = []
    n_iter = len(list(f_shuffle(cols, n_exog_var)))
    for variables in tqdm(f_shuffle(cols, n_exog_var), total=n_iter):
        if data[list(variables)].dropna().shape[0] < N_min:
            continue
        model = sm.formula.ols(f'{endog_var} ~ {mode.join(variables)}', data)
        res = model.fit()
        if ignore_p_intercept:
            pvals_params = res.pvalues.drop('Intercept')
        else:
            pvals_params = res.pvalues
        if res.f_pvalue < p_thld_model and all(pvals_params < p_thld_params):
            params = '/'.join([f'{p:.3f}' for p in res.params.values])
            p_values = '/'.join([f'{p:.3f}' for p in res.pvalues.values])
            d = list(variables) + [
                params, res.nobs, res.rsquared_adj, res.f_pvalue, p_values, res
            ]
            new_row = pd.Series(index=cols_res, data=d)
            results.append(new_row)

    results = pd.concat(results, axis=1).T

    return results


def models_from_literature(data):
    """
    Formulas and calculated values for literature models.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    """

    formulas = [
        # Deemer and Holgerson, 2021
        ('np.log(16*tot_flux + 1) ~ latitude + np.log(lake_area*1e-6) '
         '+ np.log(chla)'),

        # DelSontro, Beaulieu and Downing, 2018
        'np.log10(12*diff_flux + 1) ~ np.log10(lake_area*1e-6)*np.log10(chla)',
        'np.log10(12*ebul_flux + 1) ~ np.log10(chla)',
        'np.log10(12*tot_flux + 1) ~ np.log10(chla)',

        # DelSontro et al, 2016
        #'np.log10(diff_flux) ~ Tsed',
        #'np.log10(ebul_flux) ~ Tsed',
        #'np.log10(ebul_flux) ~ Tsed*np.log10(TP)',
        #'np.log10(diff_flux) ~ Tsed*np.log10(TP)',

        # Holgerson and Raymond, 2016
        'np.log(CH4) ~ np.log(lake_area*1e-4) + latitude',

        # Rasilo, Prairie and Del Giorgio, 2015
        #'np.log10(pCH4) ~ np.log10(lake_area*1e-6) + Tw',
        #'np.log10(12*diff_flux) ~ np.log10(lake_area*1e-6) + Tw',
        #'np.log10(pCH4) ~ np.log10(lake_area*1e-6) + Tw + np.log10(TN*1e3)',

        # Sepulveda-Jauregui et al, 2015
        #'np.log10(16*ebul_flux) ~ np.log10(lake_area*1e-6) - 1',
        #'np.log10(16*diff_flux) ~ np.log10(TP) - 1',
        'np.log10(16*tot_flux) ~ np.log10(lake_area*1e-6)',
        'np.log10(16*tot_flux) ~ np.log10(TP)',
        'np.log10(16*tot_flux) ~ np.log10(TN)',

        # Wik et al, 2014
        'ebul_flux_season ~ icefree_period',
        'ebul_flux_season ~ SWavg',
        #'ebul_flux_season ~ DSTmax',
        #'ebul_flux_season ~ SSTmax',
        #'SST_transf ~ np.log(16*ebul_flux)',

        # Kankaala et al, 2013
        'np.log10(icefree_period*tot_flux*1e-3) ~ np.log10(lake_area*1e-6)',

        # Juutinen et al, 2009
        'diff_flux_peryear ~ max_depth + lake_area_km2 + TN_ugL',
        'CH4 ~ max_depth + lake_area_km2',

        # Bastviken et al, 2004
        'np.log10(CH4) ~ np.log10(lake_area)',
        'np.log10(CH4) ~ np.arcsin(np.sqrt(VFAN)) + np.log10(DOC)',
        ('np.log10(12e-3*icefree_period*ebul_flux*lake_area) '
         '~ np.log10(lake_area)'),
        ('np.log10(12e-3*icefree_period*ebul_flux*lake_area) '
         '~ np.log10(lake_area) + np.log10(TP/31)'),
        'np.log10(12e-3*icefree_period*ebul_flux) ~ np.log10(TP/31)',
        ('np.log10(12e-3*icefree_period*ebul_flux) '
         '~ np.log10(TP/31) + np.log10(CH4)'),
        ('np.log10(12e-3*icefree_period*diff_flux*lake_area) '
         '~ np.log10(lake_area)'),
        #'np.log10(12e-3*icefree_period*diff_flux) ~ np.log10(SPM)',
        #'np.log10(SPL) ~ np.log10(lake_area)',
        #'np.log10(SPL) ~ np.log10(TP/31) + np.log10(DOC)',
        #'np.log10(SPM) ~ np.log10(CH4) + np.log10(DOC)',
        #'np.log10(SPM) ~ np.log10(CH4) + np.log10(TP/31)',
        #'np.log10(SPM) ~ np.arcsin(np.sqrt(VFAN))'
    ]

    variables = [
        # Deemer and Holgerson, 2021
        [np.log(16*data['tot_flux'] + 1), data['latitude'],
         np.log(1e-6*data['lake_area']), np.log(data['chla'])],

        # DelSontro, Beaulieu and Downing, 2018
        [np.log10(12*data['diff_flux'] + 1),
         np.log10(1e-6*data['lake_area']), np.log10(data['chla']),
         np.log10(1e-6*data['lake_area'])*np.log10(data['chla'])],
        [np.log10(12*data['ebul_flux'] + 1), np.log10(data['chla'])],
        [np.log10(12*data['tot_flux'] + 1), np.log10(data['chla'])],

        # Holgerson and Raymond, 2016
        [np.log(data['CH4']), np.log(1e-4*data['lake_area']), data['latitude']],

        # Sepulveda-Jauregui et al, 2015
        [np.log10(16*data['tot_flux']), np.log10(1e-6*data['lake_area'])],
        [np.log10(16*data['tot_flux']), np.log10(data['TP'])],
        [np.log10(16*data['tot_flux']), np.log10(data['TN'])],

        # Wik et al, 2014
        [16*data['ebul_flux']*data['icefree_period'], data['icefree_period']],
        [16*data['ebul_flux']*data['icefree_period'], data['SWavg']],

        # Kankaala et al, 2013
        [np.log10(1e-3*data['icefree_period']*data['tot_flux']),
         np.log10(1e-6*data['lake_area'])],

        # Juutinen et al, 2009
        [data['diff_flux']*data['icefree_period'], data['max_depth'],
         1e-6*data['lake_area'], 1e3*data['TN']],
        [data['CH4'], data['max_depth'], 1e-6*data['lake_area']],

        # Bastviken et al, 2004
        [np.log10(data['CH4']), np.log10(data['lake_area'])],
        [np.log10(data['CH4']), np.arcsin(np.sqrt(data['VFAN'])),
         np.log10(data['DOC'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']*data['lake_area']),
         np.log10(data['lake_area'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']*data['lake_area']),
         np.log10(data['lake_area']), np.log10(1/31*data['TP'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']),
         np.log10(1/31*data['TP'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']),
         np.log10(1/31*data['TP']), np.log10(data['CH4'])],
        [np.log10(12e-3*data['icefree_period']*data['diff_flux']*data['lake_area']),
         np.log10(data['lake_area'])],
        #[np.log10(12e-3*data['icefree_period']*data['diff_flux']),
        # np.log10(12*data['storage']/data['lake_area'])],
        #[np.log10(12*data['storage']), np.log10(data['lake_area'])],
        #[np.log10(12*data['storage']), np.log10(1/31*data['TP']),
        # np.log10(data['DOC'])],
        #[np.log10(12*data['storage']/data['lake_area']), np.log10(data['CH4']),
        # np.log10(data['DOC'])],
        #[np.log10(12*data['storage']/data['lake_area']), np.log10(data['CH4']),
        # np.log10(1/31*data['TP'])],
        #[np.log10(12*data['storage']/data['lake_area']),
        # np.arcsin(np.sqrt(data['VFAN']))]
    ]

    obs_vs_pred_lit = [
        # Deemer and Holgerson, 2021
        [np.log(16*data['tot_flux'] + 1),
         -1.54 - 0.03*data['latitude'] - 0.28*np.log(1e-6*data['lake_area']) \
         + 0.43*np.log(data['chla'])],

        # DelSontro, Beaulieu and Downing, 2018
        [np.log10(12*data['diff_flux'] + 1),
         0.705 - 0.167*np.log10(1e-6*data['lake_area']) \
         + 0.530*np.log10(data['chla']) \
         + 0.098*np.log10(1e-6*data['lake_area'])*np.log10(data['chla'])],
        [np.log10(12*data['ebul_flux'] + 1),
         0.758 + 0.752*np.log10(data['chla'])],
        [np.log10(12*data['tot_flux'] + 1),
         0.940 + 0.778*np.log10(data['chla'])],

        # Holgerson and Raymond, 2016
        [np.log(data['CH4']),
         4.25 - 0.278*np.log(1e-4*data['lake_area']) - 0.080*data['latitude']],

        # Sepulveda-Jauregui et al, 2015
        [np.log10(16*data['tot_flux']),
         0.43 - 0.37*np.log10(1e-6*data['lake_area'])],
        [np.log10(16*data['tot_flux']), 0.42 + 0.55*np.log10(data['TP'])],
        [np.log10(16*data['tot_flux']), 0.98 - 0.61*np.log10(data['TN'])],

        # Wik et al, 2014
        [16*data['ebul_flux']*data['icefree_period'],
         -1871.2 + 22.64*data['icefree_period']],
        [16*data['ebul_flux']*data['icefree_period'],
         -2878 + 26.83*data['SWavg']],

        # Kankaala et al, 2013
        [np.log10(1e-3*data['icefree_period']*data['tot_flux']),
         -1.596 - 0.401*np.log10(1e-6*data['lake_area'])],

        # Juutinen et al, 2009
        [data['diff_flux']*data['icefree_period'],
         0.558 - 0.360*data['max_depth'] - 0.179*1e-6*data['lake_area'] \
         + 0.488*1e3*data['TN']],
        [data['CH4'],
         0.322 - 0.050*data['max_depth'] - 0.043*1e-6*data['lake_area']],

        # Bastviken et al, 2004
        [np.log10(data['CH4']), 0.781 - 0.227*np.log10(data['lake_area'])],
        [np.log10(data['CH4']),
         0.228 + 1.209*np.arcsin(np.sqrt(data['VFAN'])) \
         - 1.042*np.log10(data['DOC'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']*data['lake_area']),
         1.190 + 0.841*np.log10(data['lake_area'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']*data['lake_area']),
         0.838 + 0.934*np.log10(data['lake_area']) \
         + 0.881*np.log10(1/31*data['TP'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']),
         0.523 + 0.950*np.log10(1/31*data['TP'])],
        [np.log10(12e-3*data['icefree_period']*data['ebul_flux']),
         0.601 + 0.821*np.log10(1/31*data['TP']) + 1.169*np.log10(data['CH4'])],
        [np.log10(12e-3*data['icefree_period']*data['diff_flux']*data['lake_area']),
         0.234 + 0.927*np.log10(data['lake_area'])]
    ]

    return formulas, variables, obs_vs_pred_lit


def test_models_from_literature(data, mode='curve_fit', CI=0.95):
    """
    Apply a range of models found in the literature to METLAKE data.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    mode : {'curve_fit', 'sklearn', 'statsmodels'}
        If mode is 'curve_fit', use the function 'curve_fit_with_diagnostics'.
        If mode is 'sklearn', use 'sklearn.linear_model.LinearRegression'.
        If mode is 'statsmodels', use 'statsmodels.api.formula.ols'.
    CI : float
        Confidence interval to be returned for the parameters of the model.
        Used only if mode is 'curve_fit'.
    """

    data = data.copy()

    if mode == 'statsmodels':
        #data.loc[data['storage'] == 0.0, 'storage'] = np.nan
        data['diff_flux_peryear'] = data['diff_flux']*data['icefree_period']
        data['lake_area_km2'] = data['lake_area']*1e-6
        data['TN_ugL'] = data['TN']*1e3
        data['ebul_flux_season'] = 16*data['ebul_flux']*data['icefree_period']
        #data['SPL'] = 12*data['storage']
        #data['SPM'] = 12*data['storage']/data['lake_area']
        #data['SST_transf'] = 1/(data['SST'] + 273.15)

    formulas, variables, obs_vs_pred_lit = models_from_literature(data)

    models = []
    summary = []
    for n, d in enumerate(variables):
        if n in [1, 4, 11, 12, 13, 14, 19]:
            drop_lakes = ['LJE']
        elif n in [0, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18]:
            drop_lakes = ['LJE', 'SOD']
        d = pd.concat(d, axis=1) \
                .drop(drop_lakes) \
                .replace([-np.inf, np.inf], np.nan) \
                .dropna()
        l = pd.concat(obs_vs_pred_lit[n], axis=1) \
                .drop(drop_lakes) \
                .replace([-np.inf, np.inf], np.nan) \
                .dropna()
        yobs_orig = l.iloc[:, 0].values
        ypred_orig = l.iloc[:, 1].values
        rmse_orig = np.sqrt(np.nanmean(np.square(ypred_orig - yobs_orig)))
        rmserel_orig = rmse_orig/np.abs(np.nanmean(yobs_orig))
        SSres_orig = np.nansum(np.square(ypred_orig - yobs_orig))
        SStot_orig = np.nansum(np.square(yobs_orig - np.nanmean(yobs_orig)))
        r2_orig = 1.0 - SSres_orig/SStot_orig
        res_orig = pd.Series({
            'RMSE_orig': rmse_orig, 'RMSErel_orig': rmserel_orig,
            'R2_orig': r2_orig
        })
        if mode == 'curve_fit':
            if d.shape[1] - 1 == 1:
                f = lambda x, a, b: a + b*x[0]
            elif d.shape[1] - 1 == 2:
                f = lambda x, a, b, c: a + b*x[0] + c*x[1]
            elif d.shape[1] - 1 == 3:
                f = lambda x, a, b, c, d: a + b*x[0] + c*x[1] + d*x[2]
            elif d.shape[1] - 1 == 4:
                f = lambda x, a, b, c, d, e: a + b*x[0] + c*x[1] + d*x[2] + e*x[3]
            X = d.iloc[:, 1:].values.T
            y = d.iloc[:, 0].values
            res = curve_fit_with_diagnostics(f, X, y, CI)
            res['pcov'] = [list(pcov) for pcov in res['pcov']]
            summary.append(pd.concat([
                pd.Series({'formula': formulas[n]}), res, res_orig
            ]))
        elif mode == 'sklearn':
            X = d.iloc[:, 1:].values
            y = d.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            models.append(model)
            summary.append(pd.concat([
                pd.Series({'N': len(y), 'R2': r2_score(y, model.predict(X))}),
                res_orig
            ]))
        elif mode == 'statsmodels':
            formula = formulas[n]
            try:
                lm = sm.formula.ols(formula, data).fit()
            except patsy.PatsyError:
                models.append(formula)
                continue
            models.append(lm)
            if 'Intercept' in lm.params.index:
                formula = f'{lm.params["Intercept"]:.4f}'
                for p in lm.params.drop('Intercept').index:
                    if lm.params[p] < 0:
                        formula += f' - {-lm.params[p]:.4f}*{p}'
                    else:
                        formula += f' + {lm.params[p]:.4f}*{p}'
                formula = lm.model.formula.split('~')[0] + '=  ' + formula
                p_values = '/'.join([f'{p:.3f}' for p in lm.pvalues.values])
                summary.append(pd.concat([
                    pd.Series({
                        'formula': formula, 'N': lm.nobs,
                        'R2': lm.rsquared, 'R2_adj': lm.rsquared_adj,
                        'p_value': lm.f_pvalue, 'p_values': p_values
                    }), res_orig
                ]))
    summary = pd.concat(summary, axis=1).T
    #summary.index = range(1, summary.shape[0] + 1)

    return models, summary


def save_data_tables_yearly_average(data, units):
    """
    Save yearly average data to CSV in the order used in the manuscript.

    Parameters
    ----------
    data : pd.DataFrame
        Data table returned by the function 'combine_all_yearly_average_data'.
    units : pd.Series
        Unit table returned by the function 'combine_all_yearly_average_data'.
    """

    data = data.copy()

    data.rename(index={'PAR': 'PRS'}, inplace=True)

    index = [
        'BD3', 'BD4', 'BD6', 'NAS', 'NBJ', 'LJR', 'NOR', 'GRA', 'DAM',
        'LA', 'ST', 'LO', 'VEN', 'SOD', 'PRS', 'GUN', 'GRI', 'SGA',
        'LAM', 'KLI', 'GYS'
    ]
    columns1 = [
        'latitude', 'longitude', 'altitude', 'catchment_area', 'drainage_ratio',
        'landuse_openfield', 'landuse_forest', 'landuse_farmland',
        'landuse_openwetland', 'landuse_water', 'elevationslope_deg_mean',
        'air_temp_mesan', 'wind_speed_mesan', 'precipitation_mesan',
        'SWavg', 'gpp_catchment'
    ]
    columns2 = [
        'lake_perimeter', 'lake_area', 'lake_volume',
        'mean_depth', 'max_depth', 'depth_ratio', 'residence_time',
        'icefree_period', 'storage', 'VFAN',
        'TOC', 'DOC', 'TN', 'TP', 'chla_summer', 'pH', 'abs420'
    ]
    columns3 = [
        'CH4', 'diff_flux', 'ebul_flux', 'tot_flux'
    ]
    precision = {
        'latitude': 4, 'longitude': 4, 'altitude': 0, 'catchment_area': -3,
        'drainage_ratio': 0, 'landuse_openfield': 2, 'landuse_forest': 2,
        'landuse_farmland': 2, 'landuse_openwetland': 2, 'landuse_water': 2,
        'elevationslope_deg_mean': 1, 'air_temp_mesan': 1,
        'wind_speed_mesan': 1, 'precipitation_mesan': 0, 'SWavg': 0,
        'gpp_catchment': 0,
        'lake_perimeter': -1, 'lake_area': -2, 'lake_volume': -3,
        'mean_depth': 1, 'max_depth': 1, 'depth_ratio': 1,
        'residence_time': -1, 'icefree_period': 0, 'storage': -2, 'VFAN': 2,
        'TOC': 1, 'DOC': 1, 'TN': 2, 'TP': 1, 'chla_summer': 1, 'pH': 1,
        'abs420': 3,
        'CH4': 3, 'diff_flux': 3, 'ebul_flux': 3, 'tot_flux': 3
    }

    # OLD VERSION
    #index = [
    #    'BD3', 'BD4', 'BD6', 'NAS', 'NBJ', 'LJR', 'LJE', 'NOR', 'GRA', 'DAM',
    #    'VEN', 'SOD', 'PAR', 'GUN', 'GRI', 'SGA', 'LAM', 'KLI', 'GYS'
    #]
    #columns1 = [
    #    'latitude', 'longitude', 'altitude', 'lake_area', 'lake_volume',
    #    'max_depth', 'residence_time', 'catchment_area', 'icefree_period',
    #    'precipitation', 'SWavg', 'gpp_catchment'
    #]
    #columns2 = [
    #    'tot_flux', 'diff_flux', 'ebul_flux', 'CH4', 'pCH4', 'k_diff',
    #    'TC', 'DOC', 'TN', 'TP', 'chla', 'attenuation', 'storage', 'VFAN'
    #]
    #precision = {
    #    'latitude': 4, 'longitude': 4, 'altitude': 0, 'lake_area': -2,
    #    'lake_volume': -3, 'max_depth': 1, 'residence_time': -1,
    #    'catchment_area': -3, 'icefree_period': 0, 'precipitation': 0,
    #    'SWavg': 0, 'gpp_catchment': 0, 'tot_flux': 3, 'diff_flux': 3,
    #    'ebul_flux': 3, 'CH4': 3, 'pCH4': 0, 'k_diff': 2, 'TC': 1, 'DOC': 1,
    #    'TN': 2, 'TP': 1, 'chla': 1, 'attenuation': 2, 'storage': -2, 'VFAN': 2
    #}

    for col, n in precision.items():
        data[col] = np.round(data[col], n)
        if n <= 0:
            data[col] = data[col].astype(pd.Int64Dtype())

    for n, cols in enumerate([columns1, columns2, columns3]):
        #d = data.set_index('Lake').loc[index, cols]
        d = data.loc[index, cols]
        d.columns = pd.MultiIndex.from_arrays([d.columns, units[cols].values])
        date_stamp = datetime.now().strftime('%Y%m%d')
        d.to_csv(os.path.join(
            f'/home/jonathan/OneDrive/VM/Metlake',
            f'EmpiricalModels_Table{n + 1}_{date_stamp}.csv'
        ))


def save_tables_empirical_models(
    data, levels_models=['linear'],
    levels_lakes_removed=[('LJE',), ('LJE', 'SOD')], levels_N_lin=[1, 2, 3],
    levels_variables=['diff_flux', 'ebul_flux', 'tot_flux', 'CH4']
):
    """
    Save tables containing the results of applying empirical models
    to whole-lake, whole-year data.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    levels_models : list
        Types of models to use.
    levels_lakes_removed : list
        Lakes to remove for each category of run.
    levels_N_lin : list
        Number of parameters to use in the linear models.
    levels_variables : list
        Variables to use as dependent variables.
    """

    warnings.filterwarnings('ignore')

    cols = [
        'latitude', 'altitude', 'catchment_area', 'drainage_ratio',
        'landuse_openfield', 'landuse_forest', 'landuse_farmland',
        'landuse_openwetland', 'landuse_water', 'elevationslope_deg_mean',
        'air_temp_mesan', 'wind_speed_mesan', 'precipitation_mesan',
        'SWavg', 'gpp_catchment',
        'CH4', 'diff_flux', 'ebul_flux', 'tot_flux',
        'lake_perimeter', 'lake_area', 'lake_volume',
        'mean_depth', 'max_depth', 'depth_ratio', 'residence_time',
        'storage', 'VFAN',
        'TOC', 'TN', 'TP', 'chla_summer', 'pH', 'abs420'
    ]
    data = data[cols]

    levels = [
        levels_models, levels_lakes_removed, levels_N_lin, levels_variables
    ]
    index = pd.MultiIndex(levels=levels, codes=[[], [], [], []])
    results_curve_fit = pd.Series(index=index, dtype='object')
    results_statsmodels = pd.Series(index=index, dtype='object')

    f_lin = {}
    f_lin[1] = lambda x, a, b      : a + b*x
    f_lin[2] = lambda x, a, b, c   : a + b*x[0] + c*x[1]
    f_lin[3] = lambda x, a, b, c, d: a + b*x[0] + c*x[1] + d*x[2]

    for lakes_removed in levels[1]:
        d = data.copy()
        for lake in lakes_removed:
            d = d.drop(lake)
        for n_var in levels[2]:
            for y_var in levels[3]:
                settings = ('linear', lakes_removed, n_var, y_var)
                print(settings)
                res_curve_fit = fit_model_on_data_using_curve_fit(
                    f_lin[n_var], d, y_var, n_var,
                    f_shuffle=itertools.combinations
                )
                res_statsmodels = fit_linear_model_on_data_using_statsmodels(
                    d, y_var, n_var, f_shuffle=itertools.combinations
                )
                results_curve_fit[settings] = res_curve_fit
                results_statsmodels[settings] = res_statsmodels
                # Save data to file
                res_curve_fit = res_curve_fit.sort_values('R2')
                res_curve_fit['pcov'] = [
                    [list(p) for p in pcov] for pcov in res_curve_fit['pcov']
                ]
                date_stamp = datetime.now().strftime('%Y%m%d')
                res_curve_fit.to_csv(os.path.join(
                    f'/home/jonathan/OneDrive/VM/Metlake/'
                    f'EmpiricalModels_Results_linear_'
                    f'-{"-".join(lakes_removed)}_{n_var}var_{y_var}_'
                    f'{date_stamp}.csv'
                ))

    return results_curve_fit, results_statsmodels


def pick_random_deployment_group_and_compare_fluxes(all_fluxes, lake_fluxes):
    """
    Pick a random deployment group from the individual fluxes table and
    the total lake fluxes table.

    Parameters
    ----------
    all_fluxes : pandas.DataFrame
        Table containing individual chambers measurements (imported using
        the function 'import_chambers_data' and including a column added
        with the function 'create_deployment_groups_indices').
    lake_fluxes : pandas.DataFrame
        Table containing the whole-lake fluxes calculated using the function
        'calculate_average_flux_per_deployment_group'.
    """

    n_max = np.nanmax(all_fluxes[('General', 'Event', '')])
    n = np.round(np.random.rand()*n_max)
    all_fluxes_group = all_fluxes.loc[
        all_fluxes[('General', 'Event', '')] == n,
        [
            ('General', 'Event', ''),
            ('General', 'Lake', 'Unnamed: 0_level_2'),
            ('General', 'Chamber ID', 'Unnamed: 1_level_2'),
            ('Initial', 'Depth', 'm'), ('Final', 'Depth', 'm'),
            ('Flux calculation', 'Mean KH', 'M/atm'),
            ('Flux calculation', 'k', 'm d-1'),
            ('Initial', 'pCH4', 'µatm'),
            ('Flux calculation', 'Mean pCH4aq', 'µatm'),
            ('Flux calculation', 'CH4 flux non-linear', 'mmol m-2 d-1'),
            ('Flux calculation', 'CH4 flux linear', 'mmol m-2 d-1')
        ]
    ]
    lake_fluxes_group = lake_fluxes.loc[lake_fluxes.index == n]

    return all_fluxes_group, lake_fluxes_group


def check_time_coherence_in_chambers_data(chambers, mode):
    """
    Verify that there are no issues with chambers data when it comes
    to their ordering in time.

    Parameters
    ----------
    chambers : dict
        Dictionary of data table returned by the function
        'import_chambers_data'.
    mode : {1, 2}
        1: Check that final water sample values match next initial water
           sample values or that there is no later deployment within
           the same sampling campaign. Does that only for data from lakes
           'LA', 'LO', and 'ST'.
        2: Check that the initial time of deployment of a chamber is always
           later than the final time of the previous deployment of the same
           chamber.
    """

    col_lake = ('General', 'Lake', 'Unnamed: 0_level_2')
    col_chamber = ('General', 'Chamber ID', 'Unnamed: 1_level_2')
    col_dt_ini = ('General', 'Initial sampling', 'Date and time')
    col_dt_fin = ('General', 'Final sampling', 'Date and time')
    col_CH4_HS_ini = ('Initial', 'CH4 in water vial headspace', 'ppm')
    col_CH4_HS_fin = ('Final', 'CH4 in water vial headspace', 'ppm')

    if mode == 1:
        data = chambers['data_in']
        check = []
        for lake, d_lake in data.groupby(col_lake):
            for chamber, d_chamber in d_lake.groupby(col_chamber):
                dt_ini = d_chamber[col_dt_ini].iloc[1:].values
                dt_fin = d_chamber[col_dt_fin].iloc[:-1].values
                dt_diff = dt_ini - dt_fin
                cond = np.logical_or(
                    np.isnan(dt_diff), dt_diff >= np.timedelta64(0)
                )
                check.append(pd.Series({
                    'Lake': lake, 'Chamber': chamber, 'Coherence OK': all(cond)
                }))
        check = pd.concat(check, axis=1).T
        return check
    elif mode == 2:
        data = chambers['raw_data_in'].copy()
        f_lake = lambda x: x in ['LA', 'LO', 'ST']
        data = data.loc[data[col_lake].apply(f_lake)].copy()
        check = {}
        for ind, row in data.iterrows():
            if not np.isnan(row[col_CH4_HS_fin]):
                cond_lake = data[col_lake] == row[col_lake]
                cond_chamber = data[col_chamber] == row[col_chamber]
                diff_time = data[col_dt_ini] - row[col_dt_fin]
                cond_time = np.logical_and(
                    diff_time <= timedelta(hours=1),
                    diff_time >= timedelta(hours=0)
                )
                cond = np.logical_and(np.logical_and(
                    cond_lake, cond_chamber), cond_time)
                if cond.sum() == 0:
                    cond2 = np.logical_and(cond_lake, cond_chamber)
                    diff_time_2 = data.loc[cond2, col_dt_ini] - row[col_dt_fin]
                    cond3 = diff_time_2 >= timedelta(hours=0)
                    if cond3.sum() == 0:
                        check[ind] = 'New sampling period'
                    else:
                        min_diff_time_2 = min(
                            diff_time_2[diff_time_2 >= timedelta(hours=0)]
                        )
                        if min_diff_time_2 > timedelta(hours=24*7):
                            check[ind] = 'New sampling period'
                        else:
                            check[ind] = 'No value at the same time'
                elif cond.sum() > 1:
                    check[ind] = 'More than one value at the same time'
                else:
                    if data.loc[cond, col_CH4_HS_ini].values[0] == \
                       row[col_CH4_HS_fin]:
                        check[ind] = 'Equal values'
                    else:
                        check[ind] = 'Different values'
        check = pd.Series(check)
        return check


def compare_linear_regressions_curve_fit_vs_statsmodels_1run(
    results_curve_fit, results_statsmodels, rtol=1e-5
):
    """
    Verify if the regression results obtained using the functions based
    on curve_fit and statsmodels provide equivalent results.

    Parameters
    ----------
    results_curve_fit : pd.DataFrame
        Table of regression results obtained with the function
        'save_tables_empirical_models'.
    results_statsmodels : pd.DataFrame
        Table of regression results obtained with the function
        'save_tables_empirical_models'.
    rtol : float, default: 1e-5
        The relative tolerance parameter in the function 'numpy.allclose'.
    """

    comp = pd.DataFrame(
        columns=['variables', 'popt', 'N', 'R2', 'R2_adj',
                 'tvalues', 'pvalues', 'lower', 'upper'],
        dtype='bool'
    )
    for ind, res_cf in results_curve_fit.iterrows():
        res_sm = results_statsmodels.loc[ind]
        vars_sm = tuple(
            res_sm[label] for label in res_sm.index if label.startswith('var')
        )
        res_sm = results_statsmodels.loc[ind, 'model']
        comp.loc[ind, 'variables'] = res_cf['variables'] == vars_sm
        comp.loc[ind, 'popt'] = np.allclose(
            res_cf['popt'], res_sm.params.values, equal_nan=True, rtol=rtol
        )
        comp.loc[ind, 'N'] = res_cf['N'] == res_sm.nobs
        comp.loc[ind, 'R2'] = np.allclose(
            res_cf['R2'], res_sm.rsquared, equal_nan=True, rtol=rtol
        )
        comp.loc[ind, 'R2_adj'] = np.allclose(
            res_cf['R2_adj'], res_sm.rsquared_adj, equal_nan=True, rtol=rtol
        )
        comp.loc[ind, 'tvalues'] = np.allclose(
            res_cf['tvalues'], res_sm.tvalues.values, equal_nan=True, rtol=rtol
        )
        comp.loc[ind, 'pvalues'] = np.allclose(
            res_cf['pvalues'], res_sm.pvalues.values, equal_nan=True, rtol=rtol
        )
        comp.loc[ind, 'lower'] = np.allclose(
            res_cf['lower'], res_sm.conf_int()[0].values, equal_nan=True,
            rtol=rtol
        )
        comp.loc[ind, 'upper'] = np.allclose(
            res_cf['upper'], res_sm.conf_int()[1].values, equal_nan=True,
            rtol=rtol
        )

    return comp


def compare_linear_regressions_curve_fit_vs_statsmodels_all_runs(
    results_curve_fit, results_statsmodels, rtol=1e-5, cols_ignore=[]
):
    """
    Verify if the regression results obtained using the functions based
    on curve_fit and statsmodels provide equivalent results.

    Parameters
    ----------
    results_curve_fit : pd.DataFrame
        First table of regression results obtained with the function
        'fit_model_on_data_using_curve_fit'.
    results_statsmodels : pd.DataFrame
        First table of regression results obtained with the function
        'fit_linear_model_on_data_using_statsmodels'.
    rtol : float, default: 1e-5
        The relative tolerance parameter in the function 'numpy.allclose'.
    cols_ignore : list, default: []
        Columns from the comparison table to ignore.
        The variables used and the number of observations will usually always
        be the same in 'results_curve_fit' and 'results_statsmodels',
        which can results in some comparisons below to state that
        the tables are similar even if they are not.
    """

    for ind in results_curve_fit.index:
        comp = compare_linear_regressions_curve_fit_vs_statsmodels_1run(
            results_curve_fit[ind], results_statsmodels[ind], rtol
        )
        comp = comp.drop(columns=cols_ignore)
        cols_included = [repr(col) for col in comp.columns]
        if len(cols_included) > 1:
            col_info = (f'Parameters {", ".join(cols_included[:-1])} and '
                        f'{cols_included[-1]} were considered')
        elif len(cols_included) == 1:
            col_info = (f'Parameter {cols_included[0]} was considered only')
        else:
            col_info = 'No parameter was considered'
        print(ind)
        print(f'Any regression where all results are different? ({col_info})',
              np.any(np.all(np.logical_not(comp), axis=1)), end='\n\n')


def plot_boxplot_number_of_fluxes_per_lake_and_chamber(ch4_flux):
    """
    Create boxplots showing how many fluxes could be calculated.

    Parameters
    ----------
    ch4_flux : pandas.DataFrame
        Table obtained by importing the "CH4 Flux" sheet of the main Excel file.
    """

    regions = [
        ['BD03', 'BD04', 'BD06'], ['PAR', 'VEN', 'SOD'],
        ['SGA', 'GUN', 'GRI'], ['LJE', 'LJR', 'NAS', 'NBJ'],
        ['NOR', 'GRA', 'DAM'], ['LAM', 'GYS', 'KLI']
    ]

    col_lake = ('General', 'Lake', 'Unnamed: 0_level_2')
    col_chamber = ('General', 'Chamber ID', 'Unnamed: 1_level_2')
    col_nonlinear_flux = (
        'Flux calculation', 'CH4 flux non-linear', 'mmol m-2 d-1'
    )
    col_linear_flux = ('Flux calculation', 'CH4 flux linear', 'mmol m-2 d-1')

    cond_nlf = pd.notna(ch4_flux[col_nonlinear_flux])
    cond_lf = pd.notna(ch4_flux[col_linear_flux])

    for lakes in regions:
        fig, ax = plt.subplots(len(lakes), 1)
        for n, lake in enumerate(lakes):
            cond_l = ch4_flux[col_lake] == lake
            n_dep = []
            n_nlflux = []
            n_lflux = []
            for chamber in range(1, 13):
                cond_c = ch4_flux[col_chamber] == chamber
                cond_c_l = np.logical_and(cond_l, cond_c)
                cond_c_l_nlflux = np.logical_and(cond_c_l, cond_nlf)
                cond_c_l_lflux = np.logical_and(cond_c_l, cond_lf)
                n_dep.append(len(ch4_flux.loc[cond_c_l]))
                n_nlflux.append(len(ch4_flux.loc[cond_c_l_nlflux]))
                n_lflux.append(len(ch4_flux.loc[cond_c_l_lflux]))
            df = pd.DataFrame(
                {'n_dep': n_dep, 'n_nl_flux': n_nlflux, 'n_l_flux': n_lflux},
                index=range(1, 13)
            )
            df.plot.bar(ax=ax[n])
            ax[n].set_title(lake)


def plot_water_concentration_one_chamber(
    data, lake, chamber, p, ax=None, **kwargs
):
    """
    Plot a timeseries of water concentration of a gas at a chamber location.

    Parameters
    ----------
    data : pandas.DataFrame
        Table returned by the function 'create_water_concentration_table'.
    lake : str
        Lake name.
    chamber : int
        Chamber ID.
    p : {'CH4', 'DIC', 'N2O', 'CO2'}
        Parameter to plot.
    ax : None or matplotlib.axes.Axes, default: None
        Axis on which to draw the timeseries.
    **kwargs : '.Line2D' properties
        See the documentation of matplotlib.pyplot.plot for details.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    cond_lake_chamber = np.logical_and(
        data['Lake'] == lake, data['Chamber'] == chamber
    )
    subdata = data.loc[cond_lake_chamber]
    ax.plot(subdata['Datetime'], subdata[f'{p}_[uM]'], **kwargs)

    return fig, ax


def plot_water_concentration_one_lake(
    data, lake, chambers, p, ax=None, **kwargs
):
    """
    Plot timeseries of water concentration of a gas at several chamber locations
    in a lake.

    Parameters
    ----------
    data : pandas.DataFrame
        Table returned by the function 'create_water_concentration_table'.
    lake : str
        Lake name.
    chambers : list or str
        Chamber IDs. If chambers = 'all', use all available chambers for
        the given lake.
    p : {'CH4', 'DIC', 'N2O', 'CO2'}
        Parameter to plot.
    ax : None or matplotlib.axes.Axes, default: None
        Axis on which to draw the timeseries.
    **kwargs : '.Line2D' properties
        See the documentation of matplotlib.pyplot.plot for details.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    ax.set_title(lake, fontsize=20)

    cond_lake = data['Lake'] == lake
    if chambers == 'all':
        chambers = list(set(data.loc[cond_lake, 'Chamber']))

    cp = sns.xkcd_palette([
        'purple', 'green', 'blue', 'pink', 'brown', 'red', 'light blue',
        'teal', 'orange', 'light green', 'magenta', 'yellow', 'lilac'
    ])
    all_chambers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, '12b']

    for n, chamber in enumerate(all_chambers):
        if chamber in chambers:
            plot_water_concentration_one_chamber(
                data, lake, chamber, p, ax, label=chamber, color=cp[n], **kwargs
            )
            chambers.remove(chamber)
    ax.legend()

    if len(chambers) > 0:
        print(f'The following chambers were not plotted in {lake}: {chambers}.')

    return fig, ax


def plot_water_concentration_all_lakes(data1, data2=None, p='CH4', **kwargs):
    """
    Draw one figure per lake showing water concentrations of a gas at all
    chambers

    Parameters
    ----------
    data1 : pandas.DataFrame
        Table returned by the function 'create_water_concentration_table'.
    data2 : pandas.DataFrame or None, default: None
        Table returned by the function 'create_water_concentration_table'
        using the value for the 'ignore_gap_filled' parameter that was not
        used for 'data1'.
        If None, only 'data1' will be displayed on the figures.
    p : {'CH4', 'DIC', 'N2O', 'CO2'}, default: 'CH4'
        Parameter to plot.
    **kwargs : '.Line2D' properties
        See the documentation of matplotlib.pyplot.plot for details.
    """

    lakes = [
        'BD03', 'BD04', 'BD06', 'PAR', 'VEN', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM'
    ]

    if data2 is not None and \
       pd.isna(data1).sum().sum() > pd.isna(data2).sum().sum():
        marker1 = 'x'
        marker2 = '.'
    else:
        marker1 = '.'
        marker2 = 'x'

    figs = []
    axs = []

    for lake in lakes:
        fig, ax = plot_water_concentration_one_lake(
            data1, lake, 'all', p, marker=marker1, **kwargs
        )
        if data2 is not None:
            fig, ax = plot_water_concentration_one_lake(
                data2, lake, 'all', p, ax, marker=marker2, **kwargs
            )
        ax.tick_params(labelsize=20)
        figs.append(fig)
        axs.append(ax)

    return figs, axs


def plot_histogram_fluxes(data, col, log=True, ax=None, **kwargs):
    """
    Draw an histogram of fluxes and compare it to a normal distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing flux data.
    col : str or tuple
        Column in 'data' containing flux data.
    log : bool, default: True
        If True, the function will draw the histogram of the natural
        logarithm of the fluxes.
        If False, the function will draw the histogram of the fluxes.
    ax : None or matplotlib.axes.Axes, default: None
        Axis on which to draw the timeseries.
    **kwargs : 'matplotlib.pyplot.hist' parameters
        See the documentation of matplotlib.pyplot.hist for details.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    if log:
        d = np.log(data[col])
    else:
        d = data[col]

    mu = np.mean(d)
    sig = np.std(d)
    d_min = np.min(d)
    d_max = np.max(d)
    x = np.linspace(d_min - (d_max - d_min)/3, d_max + (d_max - d_min)/3, 1000)
    y = stats.norm.pdf(x, mu, sig)

    ax.hist(d, bins=50, density=True, **kwargs)
    ax.plot(x, y, 'k')

    return fig, ax


def plot_subplots_deployment_groups_fluxes_per_lake(fluxes):
    """
    Create a figure showing scatterplots of time vs. deployment groups fluxes
    for all lakes.

    Parameters
    ----------
    fluxes : pandas.DataFrame
         Data table returned by the function
        'calculate_average_flux_per_deployment_group'.
    """

    lakes = [
        'BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM'
    ]

    # Plot diffusive fluxes
    fig1, ax1 = plt.subplots(4, 5, figsize=(20, 13))
    fig1.suptitle('Diffusive flux [mmol m$^{-2}$ d$^{-1}$]', fontsize=30)
    fig1.delaxes(ax1[3, 4])
    for n, lake in enumerate(lakes):
        x, y = n//5, n%5
        d = fluxes.loc[fluxes['lake'] == lake]
        ax1[x, y].scatter(
            d['deployment_start'].apply(
                lambda x: datetime(2000, x.month, x.day)
            ), d['diff_flux']
        )
        ax1[x, y].text(
            0.8, 0.9, lake, fontsize=14, fontweight='bold',
            transform=ax1[x, y].transAxes
        )
        ax1[x, y].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax1[x, y].set_xlim(datetime(2000, 3, 1), datetime(2001, 1, 1))
        ax1[x, y].set_ylim(-0.05, 0.75)
        ax1[x, y].grid(linewidth=0.3)

    # Plot diffusive fluxes
    fig2, ax2 = plt.subplots(4, 5, figsize=(20, 13))
    fig2.suptitle('Total flux [mmol m$^{-2}$ d$^{-1}$]', fontsize=30)
    fig2.delaxes(ax2[3, 4])
    for n, lake in enumerate(lakes):
        x, y = n//5, n%5
        d = fluxes.loc[fluxes['lake'] == lake]
        ax2[x, y].scatter(
            d['deployment_start'].apply(
                lambda x: datetime(2000, x.month, x.day)
            ), d['tot_flux']
        )
        ax2[x, y].text(
            0.8, 0.9, lake, fontsize=14, fontweight='bold',
            transform=ax2[x, y].transAxes
        )
        ax2[x, y].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax2[x, y].set_xlim(datetime(2000, 3, 1), datetime(2001, 1, 1))
        if lake == 'SOD':
            ax2[x, y].set_ylim(-0.1, 12.1)
        elif lake == 'PAR':
            ax2[x, y].set_ylim(-0.1, 3.1)
        else:
            ax2[x, y].set_ylim(-0.05, 2.1)
        ax2[x, y].grid(linewidth=0.3)

    return fig1, ax1, fig2, ax2


def plot_param1_vs_param2_by_chamber_using_LinearRegression(
    fc, param_groups, values_groups, type_groups, r2_thld=0.5,
    x_param='T_wat_mean', y_param='CH4_diff_flux', x_func=None, y_func=np.log,
    x_func_opposite=None, y_func_opposite=np.exp, unique_min_max_y=True,
    folder_savefig=None, filenames_savefig=None
):
    """
    Plot any two parameters against each other for all chambers individually or
    for groups of chambers.

    Each figure corresponds to one lake and each pannel corresponds to
    one chamber or one group of chambers. Each pannel also includes the best fit
    of a linear regression between the two parameters. Arbitrary functions can
    be applied to the data prior to calculating the linear regression.
    If diffusive flux is the dependant variable, only data from chambers where
    no ebullitive flux was detected are used.
    If ebullitive flux is the independant variable, chambers where
    the ebullitive flux is zero are discarded.

    Parameters
    ----------
    fc : pandas.DataFrame
        Data table returned by the function
        'combine_all_data_at_chamber_deployment_scale'.
    param_groups : str
        Label of the column to use for selecting groups of chambers.
    values_groups : list
        List of values to use as groups or groups boundaries when selecting
        chambers whose data is displayed in each pannel.
    type_groups : {'list', 'boundaries'}
        If 'list' is used, the values in 'values_groups' are used as values
        that have to be matched when creating groups of chambers.
        If 'boundaries' is used, the values in 'values_groups' are used as
        boundaries for the groups of chambers when creating them.
    r2_thld : float, default: 0.5
        R2 value marking the threshold above which regression lines are
        displayed in black instead of red.
    x_param : str, default: 'T_wat_mean'
        Column in table 'fc' to use as the independent variable.
    y_param : str, default: 'CH4_diff_flux'
        Column in table 'fc' to use as the dependent variable.
    x_func : function or None, default: None
        Function to apply to the independent variable before calculating
        the linear regression coefficients.
    y_func : function or None, default: numpy.log
        Function to apply to the dependent variable before calculating
        the linear regression coefficients.
    x_func_opposite : function or None, default: None
        Opposite function to 'x_func'.
    y_func_opposite : function or None, default: numpy.exp
        Opposite function to 'y_func'.
    unique_min_max_y : bool, default: True
        If True, use the same values for the limits of the y axis in all
        pannels for a given lake.
    folder_savefig : str or None, default: None
        If None, figures are not saved. If a string is passed, it must
        indicate the folder where to save the figures.
    filenames_savefig : str or None, default: None
        Common part of the saved figures' filenames.
    """

    figs = []
    axs = []
    if type_groups == 'list':
        n_pannels = len(values_groups)
        max_len_cat = max([len(repr(c)) for c in values_groups])
    elif type_groups == 'boundaries':
        n_pannels = len(values_groups) - 1
        max_len_cat = 0
        for ind_g, g in enumerate(values_groups[:-1]):
            cat = repr(f'[{g:.2f}, {values_groups[ind_g + 1]:.2f})')
            if len(cat) > max_len_cat:
                max_len_cat = len(cat)
    n_rows = int(np.floor(np.sqrt(n_pannels)))
    n_columns = int(n_pannels//n_rows + 1*(n_pannels%n_rows != 0))
    for lake, d_lake in fc.groupby('lake'):
        print(f'\n{lake}')
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(14, 10))
        fig.suptitle(lake)
        min_x_val = np.nanmin(d_lake[x_param])
        max_x_val = np.nanmax(d_lake[x_param])
        min_xlim = min_x_val - 0.05*(max_x_val - min_x_val)
        max_xlim = max_x_val + 0.05*(max_x_val - min_x_val)
        if unique_min_max_y:
            min_y_val = np.nanmin(d_lake[y_param])
            max_y_val = np.nanmax(d_lake[y_param])
            min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
            max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
        vals = np.linspace(min_x_val, max_x_val, 100)
        for ind_g, g in enumerate(values_groups):
            # Select data from a group of chambers
            if type_groups == 'list':
                if isinstance(g, list):
                    d_chamber = d_lake[
                        d_lake[param_groups].apply(lambda x: x in g)
                    ]
                else:
                    d_chamber = d_lake[d_lake[param_groups] == g]
            elif type_groups == 'boundaries':
                if ind_g == len(values_groups) - 1:
                    continue
                cond = np.logical_and(
                    d_lake[param_groups] >= g,
                    d_lake[param_groups] < values_groups[ind_g + 1]
                )
                d_chamber = d_lake[cond]
                g = f'[{g:.2f}, {values_groups[ind_g + 1]:.2f})'
            if not unique_min_max_y:
                min_y_val = np.nanmin(d_chamber[y_param])
                max_y_val = np.nanmax(d_chamber[y_param])
                min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
                max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
            # Declare the current axis
            if n_pannels == 1:
                curax = ax
            else:
                n, m = ind_g//n_columns, ind_g%n_columns
                curax = ax[n, m]
            # Remove tick labels on both axes
            #curax.set_xticklabels('')
            #curax.set_yticklabels('')
            # Select data depending on the parameters of interest
            if y_param == 'CH4_diff_flux':
                d = d_chamber[[x_param, y_param, 'k_diff']].dropna()
            elif y_param == 'CH4_ebul_flux':
                d = d_chamber[[x_param, y_param]].dropna()
                d = d[d[y_param] > 0.]
            else:
                d = d_chamber[[x_param, y_param]].dropna()
            # Display data points
            curax.scatter(d[x_param], d[y_param])
            # Skip regression calculation if less than three points are
            # available
            if d.shape[0] < 3:
                continue
            # Apply transformations to data if requested
            if x_func is None:
                X = d[x_param].values.reshape(-1, 1)
            else:
                X = x_func(d[x_param].values.reshape(-1, 1))
            if y_func is None:
                y = d[y_param].values.reshape(-1, 1)
            else:
                y = y_func(d[y_param].values.reshape(-1, 1))
            # Calculate linear regression and R2 score
            lm = LinearRegression().fit(X, y)
            r2 = lm.score(X, y)
            # Calculate and add a trendline in the pannels. Also calculate
            # the average of observed values and the average of predicted
            # values.
            if x_func_opposite is None:
                x_vals = vals
            else:
                x_vals = x_func_opposite(vals)
            if y_func_opposite is None:
                y_vals = lm.intercept_[0] + lm.coef_[0][0]*vals
                y_model = lm.intercept_[0] + lm.coef_[0][0]*X
            else:
                y_vals = y_func_opposite(lm.intercept_[0] + lm.coef_[0][0]*vals)
                y_model = y_func_opposite(lm.intercept_[0] + lm.coef_[0][0]*X)
            if r2 >= r2_thld:
                c = 'k'
                print(f'{repr(g):{max_len_cat}}  R2 = {r2:.3f}')
            else:
                c = 'r'
                print(f'{repr(g):{max_len_cat}}  R2 = {r2:.3f}  LOW')
            curax.plot(x_vals, y_vals, c=c)
            # Add quantitative information as text in the pannel
            text = (f'{g}\nn = {d.shape[0]}\nR$^2$ = {r2:.3f}'
                    f'\nintercept = {lm.intercept_[0]:.3f}'
                    f'\nslope = {lm.coef_[0][0]:.3f}'
                    f'\n{repr(y_func_opposite)}(intercept) = '
                    f'{y_func_opposite(lm.intercept_[0]):.3f}'
                    f'\nmean measurements = {np.mean(d[y_param]):.3f}'
                    f'\nmean predictions = {np.mean(y_model):.3f}'
                   )
            curax.text(
                0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                verticalalignment='top', transform=curax.transAxes
            )
            # Set limits of the axes
            curax.set_xlim(min_xlim, max_xlim)
            curax.set_ylim(min_ylim, max_ylim)
        figs.append(fig)
        axs.append(ax)
        if folder_savefig is not None:
            fig.savefig(os.path.join(
                folder_savefig, f'{lake}_{filenames_savefig}.jpg'
            ))

    return figs, axs


def plot_param1_vs_param2_by_chamber_using_curve_fit(
    fc, param_groups, values_groups, type_groups, f, p0, f_disp, r2_thld=0.5,
    x_param='T_wat_mean', y_param='CH4_diff_flux', unique_min_max_y=True,
    folder_savefig=None, filenames_savefig=None
):
    """
    Plot any two parameters against each other for all chambers individually or
    for groups of chambers.

    Each figure corresponds to one lake and each pannel corresponds to
    one chamber or one group of chambers. Each pannel also includes the best fit
    of an arbitrary regression between the two parameters.
    If diffusive flux is the dependant variable, only data from chambers where
    no ebullitive flux was detected are used.
    If ebullitive flux is the independant variable, chambers where
    the ebullitive flux is zero are discarded.

    Parameters
    ----------
    fc : pandas.DataFrame
        Data table returned by the function
        'combine_all_data_at_chamber_deployment_scale'.
    param_groups : str
        Label of the column to use for selecting groups of chambers.
    values_groups : list
        List of values to use as groups or groups boundaries when selecting
        chambers whose data is displayed in each pannel.
    type_groups : {'list', 'boundaries'}
        If 'list' is used, the values in 'values_groups' are used as values
        that have to be matched when creating groups of chambers.
        If 'boundaries' is used, the values in 'values_groups' are used as
        boundaries for the groups of chambers when creating them.
    f : callable
        Model function to be used in 'scipy.curve_fit'.
    p0 : list
        Initial guess for the parameters in function 'f'.
    f_disp : str
        String of the generic function fitted on the data.
    r2_thld : float, default: 0.5
        R2 value marking the threshold above which regression lines are
        displayed in black instead of red.
    x_param : str, default: 'T_wat_mean'
        Column in table 'fc' to use as the independent variable.
    y_param : str, default: 'CH4_diff_flux'
        Column in table 'fc' to use as the dependent variable.
    unique_min_max_y : bool, default: True
        If True, use the same values for the limits of the y axis in all
        pannels for a given lake.
    folder_savefig : str or None, default: None
        If None, figures are not saved. If a string is passed, it must
        indicate the folder where to save the figures.
    filenames_savefig : str or None, default: None
        Common part of the saved figures' filenames.
    """

    figs = []
    axs = []
    if type_groups == 'list':
        n_pannels = len(values_groups)
        max_len_cat = max([len(repr(c)) for c in values_groups])
    elif type_groups == 'boundaries':
        n_pannels = len(values_groups) - 1
        max_len_cat = 0
        for ind_g, g in enumerate(values_groups[:-1]):
            cat = repr(f'[{g:.2f}, {values_groups[ind_g + 1]:.2f})')
            if len(cat) > max_len_cat:
                max_len_cat = len(cat)
    n_rows = int(np.floor(np.sqrt(n_pannels)))
    n_columns = int(n_pannels//n_rows + 1*(n_pannels%n_rows != 0))
    for lake, d_lake in fc.groupby('lake'):
        print(f'\n{lake}')
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(14, 10))
        fig.suptitle(lake)
        min_x_val = np.nanmin(d_lake[x_param])
        max_x_val = np.nanmax(d_lake[x_param])
        min_xlim = min_x_val - 0.05*(max_x_val - min_x_val)
        max_xlim = max_x_val + 0.05*(max_x_val - min_x_val)
        if unique_min_max_y:
            min_y_val = np.nanmin(d_lake[y_param])
            max_y_val = np.nanmax(d_lake[y_param])
            min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
            max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
        vals = np.linspace(min_x_val, max_x_val, 100)
        for ind_g, g in enumerate(values_groups):
            # Select data from a group of chambers
            if type_groups == 'list':
                if isinstance(g, list):
                    d_chamber = d_lake[
                        d_lake[param_groups].apply(lambda x: x in g)
                    ]
                else:
                    d_chamber = d_lake[d_lake[param_groups] == g]
            elif type_groups == 'boundaries':
                if ind_g == len(values_groups) - 1:
                    continue
                cond = np.logical_and(
                    d_lake[param_groups] >= g,
                    d_lake[param_groups] < values_groups[ind_g + 1]
                )
                d_chamber = d_lake[cond]
                g = f'[{g:.2f}, {values_groups[ind_g + 1]:.2f})'
            if not unique_min_max_y:
                min_y_val = np.nanmin(d_chamber[y_param])
                max_y_val = np.nanmax(d_chamber[y_param])
                min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
                max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
            # Declare the current axis
            if n_pannels == 1:
                curax = ax
            else:
                n, m = ind_g//n_columns, ind_g%n_columns
                curax = ax[n, m]
            # Remove tick labels on both axes
            #curax.set_xticklabels('')
            #curax.set_yticklabels('')
            # Select data depending on the parameters of interest
            if y_param == 'CH4_diff_flux':
                d = d_chamber[[x_param, y_param, 'k_diff']].dropna()
            elif y_param == 'CH4_ebul_flux':
                d = d_chamber[[x_param, y_param]].dropna()
                d = d[d[y_param] > 0.]
            else:
                d = d_chamber[[x_param, y_param]].dropna()
            # Display data points
            curax.scatter(d[x_param], d[y_param])
            # Skip regression calculation if less than three points are
            # available
            if d.shape[0] < 3:
                continue
            # Calculate regression and R2 score
            try:
                popt, pcov = curve_fit(f, d[x_param], d[y_param], p0)
            except RuntimeError:
                text = f'{g}\nn = {d.shape[0]}'
                text += f'\nmean measurements = {np.mean(d[y_param]):.3f}'
                curax.text(
                    0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                    verticalalignment='top', transform=curax.transAxes
                )
                curax.set_xlim(min_xlim, max_xlim)
                curax.set_ylim(min_ylim, max_ylim)
                continue
            y_model = f(d[x_param], *popt)
            r2 = r2_score(d[y_param], y_model)
            # Calculate and add a trendline in the pannels. Also calculate
            # the average of observed values and the average of predicted
            # values.
            if r2 >= r2_thld:
                c = 'k'
                print(f'{repr(g):{max_len_cat}}  R2 = {r2:.3f}')
            else:
                c = 'r'
                print(f'{repr(g):{max_len_cat}}  R2 = {r2:.3f}  LOW')
            curax.plot(vals, f(vals, *popt), c=c)
            # Add quantitative information as text in the pannel
            text = f'{g}\nn = {d.shape[0]}\nR$^2$ = {r2:.3f}\n{f_disp}'
            for n, p in enumerate(popt):
                text += f'\np{n} = {p:.3e}'
            text += f'\nmean measurements = {np.mean(d[y_param]):.3f}'
            text += f'\nmean predictions = {np.mean(y_model):.3f}'
            curax.text(
                0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                verticalalignment='top', transform=curax.transAxes
            )
            # Set limits of the axes
            curax.set_xlim(min_xlim, max_xlim)
            curax.set_ylim(min_ylim, max_ylim)
        figs.append(fig)
        axs.append(ax)
        if folder_savefig is not None:
            fig.savefig(os.path.join(
                folder_savefig, f'{lake}_{filenames_savefig}.jpg'
            ))

    return figs, axs


def plot_param1_vs_param2_averaged_by_depth_zones_using_curve_fit(
    fc_avg, f, p0, f_disp, r2_thld=0.5, x_param='T', y_param='diff_flux',
    unique_min_max_y=True, folder_savefig=None, filenames_savefig=None
):
    """
    Plot any two parameters against each other from a data table containing
    values averaged by depth zones.

    Each figure corresponds to one lake and each pannel corresponds to
    one depth zone. Each pannel also includes the best fit of the regression
    of an arbitrary function between the two parameters.

    Parameters
    ----------
    fc_avg : pandas.DataFrame
        Data table returned by the function
        'calculate_average_flux_per_deployment_group'.
    f : callable
        Model function to be used in 'scipy.curve_fit'.
    p0 : list
        Initial guess for the parameters in function 'f'.
    f_disp : str
        String of the generic function fitted on the data.
    r2_thld : float, default: 0.5
        R2 value marking the threshold above which regression lines are
        displayed in black instead of red.
    x_param : str, default: 'T'
        Column in table 'fc_avg' to use as the independent variable.
    y_param : str, default: 'diff_flux'
        Column in table 'fc_avg' to use as the dependent variable.
    unique_min_max_y : bool, default: True
        If True, use the same values for the limits of the y axis in all
        pannels for a given lake.
    folder_savefig : str or None, default: None
        If None, figures are not saved. If a string is passed, it must
        indicate the folder where to save the figures.
    filenames_savefig : str or None, default: None
        Common part of the saved figures' filenames.
    """

    figs = []
    axs = []
    n_rows = 2
    n_columns = 2
    x_cols = [f'{x_param}_zone{n}' for n in range(1, 5)]
    y_cols = [f'{y_param}_zone{n}' for n in range(1, 5)]
    for lake, d_lake in fc_avg.groupby('lake'):
        print(f'\n{lake}')
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(14, 10))
        fig.suptitle(lake)
        min_x_val = np.nanmin(d_lake[x_cols])
        max_x_val = np.nanmax(d_lake[x_cols])
        min_xlim = min_x_val - 0.05*(max_x_val - min_x_val)
        max_xlim = max_x_val + 0.05*(max_x_val - min_x_val)
        if unique_min_max_y:
            min_y_val = np.nanmin(d_lake[y_cols])
            max_y_val = np.nanmax(d_lake[y_cols])
            min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
            max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
        vals = np.linspace(min_x_val, max_x_val, 100)
        for n_zone in range(4):
            n, m = n_zone//n_columns, n_zone%n_columns
            curax = ax[n, m]
            nz = n_zone + 1
            d = d_lake[[x_cols[n_zone], y_cols[n_zone]]].dropna()
            curax.scatter(d[x_cols[n_zone]], d[y_cols[n_zone]], c='r')
            if y_param.endswith('flux') and lake == 'PAR':
                # Previously (before adding three lakes from Uppsala),
                # dropped rows 282 and 283.
                # d = d.drop([282, 283])
                d = d.iloc[np.r_[0:18, 20:24]]
            elif y_param == 'ebul_flux' and lake == 'GRA' and nz == 1:
                # Previously (before adding three lakes from Uppsala),
                # dropped rows 93 and 94.
                # d = d.drop([93, 94])
                d = d.iloc[np.r_[0:5, 7:15]]
            elif y_param == 'ebul_flux' and lake == 'GRI' and nz == 4:
                # This is not an optimal solution because the resulting
                # regression line increases steeply at high
                # temperatures. This effect is reduced by removing row 112.
                # Previously (before adding three lakes from Uppsala),
                # dropped rows 112 and 114.
                # d = d.drop([112, 114])
                d = d.iloc[np.r_[0:8, 9:10, 11:17]]
            elif y_param == 'ebul_flux' and lake == 'KLI' and nz == 1:
                # This is not an optimal solution because the resulting
                # regression line underestimates the average flux
                # obtained when the very high flux value at the
                # highest temperature is considered.
                # Previously (before adding three lakes from Uppsala),
                # dropped row 166.
                # d = d.drop(166)
                d = d.iloc[np.r_[0:8, 9:16]]
            elif y_param == 'tot_flux' and lake == 'KLI' and nz == 1:
                # This is not an optimal solution because the resulting
                # regression line underestimates the average flux
                # obtained when the very high flux value at the
                # highest temperature is considered. Dropping row 167
                # instead leads to a regression that increases fast
                # at high temperatures. Row 166 is removed here for
                # consistence with the ebullition case.
                # Previously (before adding three lakes from Uppsala),
                # dropped row 166.
                # d = d.drop(166)
                d = d.iloc[np.r_[0:8, 9:16]]
            elif y_param == 'CH4_[uM]' and lake == 'PAR':
                # Previously (before adding three lakes from Uppsala),
                # dropped rows 423, 424, and 425.
                # d = d.drop([423, 424, 425])
                d = d.iloc[np.r_[0:23, 26:32]]
            x_vals = d[x_cols[n_zone]]
            y_vals = d[y_cols[n_zone]]
            curax.scatter(x_vals, y_vals)
            if not unique_min_max_y:
                min_y_val = np.nanmin(y_vals)
                max_y_val = np.nanmax(y_vals)
                min_ylim = min_y_val - 0.05*(max_y_val - min_y_val)
                max_ylim = max_y_val + 0.05*(max_y_val - min_y_val)
            # Skip regression calculation if less than three points are
            # available but display the average value with a dotted line
            # if there is at least one value.
            if d.shape[0] == 0:
                continue
            elif d.shape[0] < 3:
                curax.hlines(
                    d.iloc[:, 1].mean(), min_xlim, max_xlim, 'r', 'dotted'
                )
                text = f'zone{n_zone + 1}\nn = {d.shape[0]}'
                text += f'\nmean measurements = {np.mean(y_vals):.3f}'
                text += '\nless than 3 values available, mean value is used'
                curax.text(
                    0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                    verticalalignment='top', transform=curax.transAxes
                )
                curax.set_xlim(min_xlim, max_xlim)
                curax.set_ylim(min_ylim, max_ylim)
                continue
            # Calculate regression and R2 score
            try:
                popt, pcov = curve_fit(f, x_vals, y_vals, p0)
            except RuntimeError:
                curax.hlines(
                    d.iloc[:, 1].mean(), min_xlim, max_xlim, 'r', 'dashed'
                )
                text = f'zone{n_zone + 1}\nn = {d.shape[0]}'
                text += f'\nmean measurements = {np.mean(y_vals):.3f}'
                text += '\ncurve_fit did not converge, mean value is used'
                curax.text(
                    0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                    verticalalignment='top', transform=curax.transAxes
                )
                curax.set_xlim(min_xlim, max_xlim)
                curax.set_ylim(min_ylim, max_ylim)
                continue
            y_model = f(x_vals, *popt)
            r2 = r2_score(y_vals, y_model)
            # Calculate and add a trendline in the pannels. Also calculate
            # the average of observed values and the average of predicted
            # values.
            if r2 >= r2_thld:
                c = 'k'
                print(f'zone {n_zone + 1}  R2 = {r2:.3f}')
            else:
                c = 'r'
                print(f'zone {n_zone + 1}  R2 = {r2:.3f}  LOW')
            # Add quantitative information as text in the pannel
            text = f'zone{n_zone + 1}\nn = {d.shape[0]}\nR$^2$ = {r2:.3f}'
            text += f'\n{f_disp}'
            for n, p in enumerate(popt):
                text += f'\np{n} = {p:.3e}'
            text += f'\nmean measurements = {np.mean(y_vals):.3f}'
            text += f'\nmean predictions = {np.mean(y_model):.3f}'
            # Use mean value instead of regression if the exponent is negative
            if popt[1] < 0:
                curax.hlines(
                    d.iloc[:, 1].mean(), min_xlim, max_xlim, 'r', 'dashed'
                )
                text += '\nnegative exponent, mean value is used'
            else:
                curax.plot(vals, f(vals, *popt), c=c)
            curax.text(
                0.01, 0.99, text, fontsize=8, horizontalalignment='left',
                verticalalignment='top', transform=curax.transAxes
            )
            # Set limits of the axes
            curax.set_xlim(min_xlim, max_xlim)
            curax.set_ylim(min_ylim, max_ylim)
        figs.append(fig)
        axs.append(ax)
        if folder_savefig is not None:
            fig.savefig(os.path.join(
                folder_savefig, f'{lake}_{filenames_savefig}.jpg'
            ))

    return figs, axs


def plot_timeseries_water_chemistry(
    CN=None, P=None, Chla=None, water_chem_clean=None
):
    """
    Plot timeseries of water chemistry variables for each lake and depth.

    Parameters
    ----------
    CN : pandas.DataFrame or None, default: None
        Data table containing carbon and nitrogen concentration data returned
        by the function 'import_water_chemistry_data' when using mode='raw'.
    P : pandas.DataFrame or None, default: None
        Data table containing total phosphorus concentration data returned
        by the function 'import_water_chemistry_data' when using mode='raw'.
    Chla : pandas.DataFrame or None, default: None
        Data table containing chlorophyll a concentration data returned
        by the function 'import_water_chemistry_data' when using mode='raw'.
    water_chem_clean : pandas.DataFrame or None, default: None
        Data table containing averaged values of duplicate or triplicate
        measurements of water chemistry data. This data table can either
        be obtained using the function 'import_water_chemistry_data' when
        using mode='cleaned' or can be imported from the 'SummaryFromPython'
        sheet of the file 'METLAKE_WaterChemistryWithSummary_2018-2020.xlsx'.
    """

    save_folder = os.path.join(
        '..', 'EmpiricalModels_manuscript',
        'figures_timeseries_waterchemistry'
    )

    if CN is None or P is None or Chla is None:
        CN_new, P_new, Chla_new = import_water_chemistry_data(mode='raw')
        if CN is None:
            CN = CN_new
        if P is None:
            P = P_new
        if Chla is None:
            Chla = Chla_new
    if water_chem_clean is None:
        # The version of the data table found in the Excel file has been
        # corrected for double dates compared to the table returned by
        # the function 'import_water_chemistry_data' using mode='cleaned'.
        # Depths (in meters) at which samples were taken were also added
        # in the Excel file. In addition, values were also corrected after
        # visual inspection of all values and TP data were replaced with
        # values from trace elements analysis for the lakes sampled in 2020.
        water_chem_clean = pd.read_excel(
            '../Data/METLAKE_WaterChemistryWithSummary_2018-2020.xlsx',
            sheet_name='SummaryFromPython', index_col=[0, 1, 2]
        )
        #water_chem_clean = import_water_chemistry_data(mode='cleaned')

    water_chem_cleaned_manually, water_chem_avg = \
            import_water_chemistry_data_cleaned_manually()

    TOC = CN[np.logical_and(np.logical_and(
        CN['Anal.'].apply(lambda x: x in ['TC', 'TOC']),
        CN['Reference_run']),
        np.logical_not(CN['Filtered'])
    )]
    DOC = CN[np.logical_and(np.logical_and(
        CN['Anal.'].apply(lambda x: x in ['TC', 'TOC']),
        CN['Reference_run']),
        CN['Filtered']
    )]
    TN = CN[np.logical_and(np.logical_and(
        CN['Anal.'].apply(lambda x: x == 'TN'),
        CN['Reference_run']),
        np.logical_not(CN['Filtered'])
    )]
    DN = CN[np.logical_and(np.logical_and(
        CN['Anal.'].apply(lambda x: x == 'TN'),
        CN['Reference_run']),
        CN['Filtered']
    )]

    data = [
        [TOC, 'Depth_category', 'Date_sampling', 'TOC_final_mg/L', 'TOC'],
        [DOC, 'Depth_category', 'Date_sampling', 'TOC_final_mg/L', 'DOC'],
        [TN, 'Depth_category', 'Date_sampling', 'TN_final_mg/L', 'TN'],
        [DN, 'Depth_category', 'Date_sampling', 'TN_final_mg/L', 'DN'],
        [P, 'Depth_category', 'Date_sampling', 'TP_ug/L', 'TP'],
        [Chla, 'Depth category', 'Date sampling',
         'Chlorophyll A concentration [ug/L]', 'Chla']
    ]

    colors = {'surface': 'white', 'middle': 'white', 'bottom': 'white'}
    edgecolors = {'surface': 'black', 'middle': 'black', 'bottom': 'black'}
    markers = {'surface': 'o', 'middle': 'D', 'bottom': 'v'}
    depth_categories = ['surface', 'middle', 'bottom']

    for df, depth_label, date_label, param_label, param in data:
        if param in ['TOC', 'DOC', 'TN', 'DN']:
            p = f'{param}_mg/L'
        elif param in ['TP', 'Chla']:
            p = f'{param}_ug/L'
        for lake, d_lake in df.groupby('Lake'):
            if lake in ['NOR', 'GRA', 'DAM', 'LAM', 'KLI', 'GYS'] \
               and param == 'TP':
                color_dots = 'grey'
            else:
                color_dots = 'black'
            depths = set(d_lake[depth_label]).intersection(depth_categories)
            n_depths = len(depths)
            fig, ax = plt.subplots(2, n_depths)
            fig.set_figheight(8)
            fig.set_figwidth(n_depths*4)
            fig.suptitle(f'{param} - {lake}')
            date_min = min(d_lake[date_label])
            date_max = max(d_lake[date_label])
            year_min = date_min.year
            year_max = date_max.year
            month_min = date_min.month
            month_max = date_max.month
            if year_min == year_max:
                n_months = date_max.month - date_min.month + 2
                interval = (n_months - 1)//5 + 1
                months = range(month_min, month_max + 2, interval)
                xticks = [datetime(year_min, month, 1) for month in months]
            else:
                n_months = date_max.month + 12 - date_min.month + 2
                interval = (n_months - 1)//5 + 1
                months = list(range(month_min, month_max + 14, interval))
                ind_new_year = sum(np.array(months) <= 12)
                months = [m%12 if m != 12 else m for m in months]
                months1 = months[:ind_new_year]
                months2 = months[ind_new_year:]
                xticks = [datetime(year_min, month, 1) for month in months1] +\
                        [datetime(year_max, month, 1) for month in months2]
            x_lim_min = date_min - timedelta(days=15)
            x_lim_max = date_max + timedelta(days=15)
            for n, depth in enumerate(list(sorted(depths, reverse=True))):
                d_depth_avg = water_chem_cleaned_manually.loc[
                    (lake, slice(None), depth), p
                ]
                d_depth_clean = water_chem_clean.loc[
                    (lake, slice(None), depth), p
                ]
                d_depth_all = d_lake[d_lake[depth_label] == depth]
                for ind_row in [0, 1]:
                    ax[ind_row, n].hlines(
                        water_chem_avg.loc[(lake, depth), p],
                        x_lim_min, x_lim_max, colors=['green']
                    )
                    if param == 'Chla':
                        ax[ind_row, n].hlines(
                            water_chem_avg.loc[(lake, depth),
                                               'Chla_summer_ug/L'],
                            x_lim_min, x_lim_max, colors=['orange']
                        )
                    ax[ind_row, n].scatter(
                        d_depth_avg.index.get_level_values(1), d_depth_avg,
                        s=150, marker='o', color='white', edgecolor='green'
                    )
                    ax[ind_row, n].scatter(
                        d_depth_clean.index.get_level_values(1), d_depth_clean,
                        s=100, marker='_', color='blue'
                    )
                    ax[ind_row, n].scatter(
                        d_depth_all[date_label], d_depth_all[param_label],
                        s=30, marker='.', color=color_dots
                    )
                    ax[ind_row, n].grid(lw=0.5)
                    ax[ind_row, n].set_xlim(x_lim_min, x_lim_max)
                    ax[ind_row, n].set_xticks(xticks)
                ax[0, n].set_title(depth)
                ax[0, n].set_xticklabels([])
                ax[1, n].set_ylim(0, 1.05*max(d_lake[param_label]))
                ax[1, n].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
            fig.tight_layout()
            fig.savefig(os.path.join(save_folder, f'{param}_{lake}.jpg'))


def plot_measured_vs_predicted_values(
    data, xcols, ycol, drop_lakes=['LJE'], title_xcols='', figfmt=None
):
    """
    Create figures for supplementary material showing measured vs predicted
    CH4-related values.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    xcols : list
        Columns of table 'data' to use as independent variables.
    ycol : str
        Column of table 'data' to use as dependent variable.
    drop_lakes : list, default: ['LJE']
        Lakes in table 'data' to ignore.
    title_xcols : str, default: ''
        Text listing independent variables to use as part of the title
        for the figure.
    figfmt : {None, 'png', 'tif'}, default: None
        Format to use when saving the figure.
        If None is passed, no figure is saved.
    """

    if len(xcols) == 1:
        xcol = xcols[0]
        d = data[[xcol, ycol]].drop(drop_lakes).dropna()
        xdata = d[xcol].values
        ydata = d[ycol].values
        f = lambda x, a, b: a + b*x
    elif len(xcols) == 2:
        d = data[xcols + [ycol]].drop(drop_lakes).dropna()
        xdata = d[xcols].values.T
        ydata = d[ycol].values
        f = lambda x, a, b, c: a + b*x[0] + c*x[1]

    if ycol == 'CH4':
        axis_label = 'average CH$_4$ concentration [µM]'
        title_label = 'CH$_4$ concentration'
    elif ycol == 'diff_flux':
        axis_label = 'average CH$_4$ diffusive flux [mmol m$^{-2}$ d$^{-1}$]'
        title_label = 'CH$_4$ diffusive flux'
    elif ycol == 'ebul_flux':
        axis_label = 'average CH$_4$ ebulitive flux [mmol m$^{-2}$ d$^{-1}$]'
        title_label = 'CH$_4$ ebullitive flux'
    elif ycol == 'tot_flux':
        axis_label = 'average CH$_4$ total flux [mmol m$^{-2}$ d$^{-1}$]'
        title_label = 'CH$_4$ total flux'
    else:
        axis_label = ''

    reg = curve_fit_with_diagnostics(f, xdata, ydata)
    ypred = f(xdata, *reg['popt'])

    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_axes([0.15, 0.10, 0.8, 0.74])
    ax.scatter(ydata, ypred, c='k')
    ax.axline((0, 0), slope=1, c='k', linestyle='--')
    ax.set_xlim(min(ydata) - 0.05*(max(ydata) - min(ydata)),
                max(ydata) + 0.05*(max(ydata) - min(ydata)))
    ax.set_ylim(min(ypred) - 0.05*(max(ypred) - min(ypred)),
                max(ypred) + 0.05*(max(ypred) - min(ypred)))
    ax.set_xlabel(f'Measured {axis_label}', fontsize=14)
    ax.set_ylabel(f'Predicted {axis_label}', fontsize=14)
    if isinstance(title_xcols, str) and title_xcols != '':
        ax.set_title(
            (f'{title_label}\npredicted from\n{title_xcols}\n'
             f'adjusted R$^2$={reg["R2_adj"]:.2f}'),
            fontsize=14, fontweight='bold'
        )
    if figfmt is not None:
        fig.savefig(os.path.join(
            '/home/jonathan/OneDrive/VM/Metlake/EmpiricalModels_manuscript',
            f'{ycol}_predicted_from_{"+".join(xcols)}.{figfmt}'
        ))

    return reg, fig, ax


def plot_measured_vs_predicted_values_all_figures_for_manuscript(data, figfmt):
    """
    Create figures for the supplementary material of the manuscript.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    figfmt : {None, 'png', 'tif'}, default: None
        Format to use when saving the figure.
        If None is passed, no figure is saved.
    """

    params = [
        ['CH4', ['landuse_forest', 'wind_speed_mesan']],
        ['CH4', ['landuse_openfield', 'wind_speed_mesan']],
        ['CH4', ['lake_area']],
        ['diff_flux', ['CH4']],
        ['diff_flux', ['landuse_forest', 'wind_speed_mesan']],
        ['diff_flux', ['landuse_openfield', 'wind_speed_mesan']],
        ['diff_flux', ['lake_area']],
        ['ebul_flux', ['TP']],
        ['ebul_flux', ['chla_summer']],
        ['ebul_flux', ['lake_area']],
        ['tot_flux', ['TP']],
        ['tot_flux', ['chla_summer']],
        ['tot_flux', ['lake_area']]
    ]

    titles_xcols = [
        'forest coverage and wind speed',
        'open field coverage and wind speed',
        'lake area',
        'CH4 concentration',
        'forest coverage and wind speed',
        'open field coverage and wind speed',
        'lake area',
        'total phosphorus concentration',
        'chlorophyll a concentration in summer',
        'lake area',
        'total phosphorus concentration',
        'chlorophyll a concentration in summer',
        'lake area'
    ]

    for n, [ycol, xcols] in enumerate(params):
        if ycol == 'CH4' or ycol == 'diff_flux':
            drop_lakes = ['LJE']
        elif ycol == 'ebul_flux' or ycol == 'tot_flux':
            drop_lakes = ['LJE', 'SOD']
        reg, fig, ax = plot_measured_vs_predicted_values(
            data, xcols, ycol, drop_lakes, titles_xcols[n], figfmt
        )
        print(f'independent variable(s): {" + ".join(xcols)}', sep='\n')
        print(f'dependent variable: {ycol}', sep='\n')
        print(f'fitted parameters: {[f"{p:.3e}" for p in reg["popt"]]}', sep='\n')
        print(f'lower: {[f"{p:.3e}" for p in reg["lower"]]}', sep='\n')
        print(f'upper: {[f"{p:.3e}" for p in reg["upper"]]}', sep='\n')
        print(f'p-values: {[f"{p:.3e}" for p in reg["pvalues"]]}', sep='\n')
        print(f'N: {reg["N"]}', sep='\n')
        print(f'adj. R2: {reg["R2_adj"]:.2f}', sep='\n')
        print(f'RMSE: {reg["RMSE"]:.3e}', sep='\n')
        print(f'RMSE norm.: {reg["RMSErel"]:.3e}', sep='\n', end='\n\n')


def plot_measured_vs_predicted_values_literature_models_for_manuscript(
    data, figfmt
):
    """
    Create figures for the supplementary material of the manuscript.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with key 'data_final' in the dictionary returned by the function
        'create_final_data_table_for_empirical_models'.
    figfmt : {None, 'png', 'tif'}, default: None
        Format to use when saving the figure.
        If None is passed, no figure is saved.
    """

    formulas, variables, obs_vs_pred_lit = models_from_literature(data)

    for n in range(len(variables)):
        if n in [4, 12, 13, 14]:
            drop_lakes = ['LJE']
            title_label = 'CH$_4$ concentration'
            if n == 12:
                ax_lab = 'CH$_{4,aq}$ [µM]'
                title = 'Model 1.4'
            if n == 4:
                ax_lab = 'ln(CH$_{4,aq}$) [ln(µM)]'
                title = 'Model 1.5'
            if n == 13:
                ax_lab = 'log(CH$_{4,aq}$) [log(µM)]'
                title = 'Model 1.6'
            if n == 14:
                ax_lab = 'log(CH$_{4,aq}$) [log(µM)]'
                title = 'Model 1.7'
        elif n in [1, 11, 19]:
            drop_lakes = ['LJE']
            title_label = 'CH$_4$ diffusive flux'
            if n == 11:
                ax_lab = 'F$_{diff}$ [mg C m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 2.5'
            if n == 1:
                ax_lab = 'log(F$_{diff}$ + 1) [log(mg C m$^{-2}$ d$^{-1}$)]'
                title = 'Model 2.6'
            if n == 19:
                ax_lab = 'F$_{diff}$ [g C lake$^{-1}$ yr$^{-1}$)]'
                title = 'Model 2.7'
        elif n in [2, 8, 9, 15, 16, 17, 18]:
            drop_lakes = ['LJE', 'SOD']
            title_label = 'CH$_4$ ebullitive flux'
            if n == 8:
                ax_lab = 'F$_{ebul}$ [mg CH$_4$ m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 3.4'
            if n == 9:
                ax_lab = 'F$_{ebul}$ [mg CH$_4$ m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 3.5'
            if n == 2:
                ax_lab = 'log(F$_{ebul}$ + 1) [log(mg C m$^{-2}$ d$^{-1}$)]'
                title = 'Model 3.6'
            if n == 17:
                ax_lab = 'F$_{ebul}$ [g C m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 3.7'
            if n == 18:
                ax_lab = 'F$_{ebul}$ [g C m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 3.8'
            if n == 15:
                ax_lab = 'F$_{ebul}$ [g C lake$^{-1}$ yr$^{-1}$)]'
                title = 'Model 3.9'
            if n == 16:
                ax_lab = 'F$_{ebul}$ [g C lake$^{-1}$ yr$^{-1}$)]'
                title = 'Model 3.10'
        elif n in [0, 3, 5, 6, 7, 10]:
            drop_lakes = ['LJE', 'SOD']
            title_label = 'CH$_4$ total flux'
            if n == 3:
                ax_lab = 'log(F$_{tot}$ + 1) [log(mg C m$^{-2}$ d$^{-1}$)]'
                title = 'Model 4.4'
            if n == 6:
                ax_lab = 'log(F$_{tot}$) [log(mg CH$_4$ m$^{-2}$ d$^{-1}$)]'
                title = 'Model 4.5'
            if n == 7:
                ax_lab = 'log(F$_{tot}$) [log(mg CH$_4$ m$^{-2}$ d$^{-1}$)]'
                title = 'Model 4.6'
            if n == 5:
                ax_lab = 'log(F$_{tot}$) [log(mg CH$_4$ m$^{-2}$ d$^{-1}$)]'
                title = 'Model 4.7'
            if n == 0:
                ax_lab = 'ln(F$_{tot}$ + 1) [log(mg CH$_4$ m$^{-2}$ d$^{-1}$)]'
                title = 'Model 4.8'
            if n == 10:
                ax_lab = 'log(F$_{tot}$) [log(mol m$^{-2}$ yr$^{-1}$)]'
                title = 'Model 4.9'

        d = pd.concat(variables[n], axis=1).drop(drop_lakes).dropna()
        obs_lit = obs_vs_pred_lit[n][0].drop(drop_lakes)
        pred_lit = obs_vs_pred_lit[n][1].drop(drop_lakes)

        if len(variables[n]) == 2:
            xdata = d.iloc[:, 1:].values.T[0]
            ydata = d.iloc[:, 0].values
            f = lambda x, a, b: a + b*x
        elif len(variables[n]) > 2:
            xdata = d.iloc[:, 1:].values.T
            ydata = d.iloc[:, 0].values
            if len(variables[n]) == 3:
                f = lambda x, a, b, c: a + b*x[0] + c*x[1]
            elif len(variables[n]) == 4:
                f = lambda x, a, b, c, d: a + b*x[0] + c*x[1] + d*x[2]

        reg = curve_fit_with_diagnostics(f, xdata, ydata)
        ypred = f(xdata, *reg['popt'])

        print(title)
        print(f'fitted parameters: {[f"{p:.3e}" for p in reg["popt"]]}', sep='\n')
        print(f'lower: {[f"{p:.3e}" for p in reg["lower"]]}', sep='\n')
        print(f'upper: {[f"{p:.3e}" for p in reg["upper"]]}', sep='\n')
        print(f'p-values: {[f"{p:.3e}" for p in reg["pvalues"]]}', sep='\n')
        print(f'N: {reg["N"]}', sep='\n')
        print(f'adj. R2: {reg["R2_adj"]:.2f}', sep='\n')
        print(f'RMSE: {reg["RMSE"]:.3e}', sep='\n')
        print(f'RMSE norm.: {reg["RMSErel"]:.3e}', sep='\n', end='\n\n')

        fig = plt.figure(figsize=(6, 6.5))
        ax = fig.add_axes([0.15, 0.10, 0.8, 0.74])
        ax.scatter(obs_lit, pred_lit,
                   color='w', edgecolor='k', label="original coef.")
        ax.scatter(ydata, ypred, c='k', label="re-fitted coef.")
        ax.axline((0, 0), slope=1, c='k', linestyle='--')
        ydata_min = min(min(ydata), min(obs_vs_pred_lit[n][0]))
        ydata_max = max(max(ydata), max(obs_vs_pred_lit[n][0]))
        ypred_min = min(min(ypred), min(obs_vs_pred_lit[n][1]))
        ypred_max = max(max(ypred), max(obs_vs_pred_lit[n][1]))
        ax.set_xlim(ydata_min - 0.05*(ydata_max - ydata_min),
                    ydata_max + 0.05*(ydata_max - ydata_min))
        ax.set_ylim(ypred_min - 0.05*(ypred_max - ypred_min),
                    ypred_max + 0.05*(ypred_max - ypred_min))
        ax.set_xlabel(f'Measured {ax_lab}', fontsize=14)
        ax.set_ylabel(f'Predicted {ax_lab}', fontsize=14)
        ax.legend(fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if figfmt is not None:
            fig.savefig(os.path.join(
                '/home/jonathan/OneDrive/VM/Metlake/EmpiricalModels_manuscript',
                f'{title.replace(" ", "_").replace(".", "-")}.{figfmt}'
            ))


def plot_air_and_surface_thermistors_temperature_all_lakes(mesan, freq):
    """
    Create one figure per lake where the surface thermistor data from all
    thermistor chains are shown, as well as the average of all chains.
    Figures also contain MESAN air temperature data.

    Parameters
    ----------
    mesan : dict
        Dictionary of pandas.DataFrame tables obtained using the function
        'import_mesan_data'.
    freq : str
        Frequency formatting string used in the method 'resample' of
        DataFrames from pandas (e.g., '10D').
    """

    p = '../Data/ThermistorChains/'
    save_folder = '../EmpiricalModels_manuscript/figures_temp_air_water'
    lakes = [
        'BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM'
    ]
    conv_lake = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExp', 'LJR': 'LjusvattentjarnRef',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }
    for lake in lakes:
        if lake in ['BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD']:
            year = 2018
        elif lake in ['SGA', 'GRI', 'GUN', 'LJE', 'LJR', 'NAS', 'NBJ']:
            year = 2019
        elif lake in ['NOR', 'GRA', 'DAM', 'LAM', 'KLI', 'GYS']:
            year = 2020
        fig = plt.figure(constrained_layout=True, figsize=(15, 5))
        s = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax1 = fig.add_subplot(s[0])
        ax2 = fig.add_subplot(s[1])
        fig.suptitle(f'{lake}\ntime resolution: {freq}', fontsize=16)
        d_surf_all = []
        for n in range(1, 4):
            f = f'Thermistors_{lake}_T{n}_raw.mat'
            if f in os.listdir(p):
                d = import_thermistor_chain_data(
                    os.path.join(p, f), mode='dataframe', year=year
                )
                if min(d.columns) < 0.1:
                    d_surf = d[min(d.columns)]
                else:
                    continue
                if lake == 'GUN' and n == 2:
                    datetime_failure = datetime(2019, 9, 12, 6, 0)
                    d_surf[d_surf.index >= datetime_failure] = np.nan
                d_surf.name = f'T{n}'
                d_surf_all.append(d_surf)
        d_surf_all = pd.concat(d_surf_all, axis=1)
        mesan_lake = mesan[conv_lake[lake]]
        mesan_lake = mesan_lake.loc[
            np.logical_and(
                mesan_lake.index >= d_surf_all.index[0],
                mesan_lake.index <= d_surf_all.index[-1]
            ), 'Temperature'
        ]
        d_surf_all = d_surf_all.resample(freq).mean()
        mesan_lake = mesan_lake.resample(freq).mean()
        d_air_wat = pd.concat([mesan_lake, d_surf_all.mean(axis=1)], axis=1)
        d_air_wat = d_air_wat.dropna()
        X = d_air_wat.iloc[:, 0].values.reshape(-1, 1)
        y = d_air_wat.iloc[:, 1].values.reshape(-1, 1)
        lm = LinearRegression().fit(X, y)
        lm_slope = lm.coef_[0][0]
        lm_intercept = lm.intercept_[0]
        r2 = lm.score(X, y)
        text_ax2 = (f'slope: {lm_slope:.3f}\nintercept: {lm_intercept:.3f}'
                    f'\nR$^2$: {r2:.3f}')
        for col in d_surf_all.columns:
            ax1.plot(d_surf_all[col], label=col)
        ax1.plot(d_surf_all.mean(axis=1), 'k', linewidth=0.5, label='mean T1-3')
        ax1.plot(mesan_lake, 'r', label='MESAN')
        ax1.set_ylabel('Temperature [oC]', fontsize=13)
        ax1.tick_params(labelsize=10)
        ax1.legend()
        ax2.scatter(mesan_lake, d_surf_all.mean(axis=1), c='k')
        ax2.axline((10, 10), slope=1, color='g', linestyle='--',
                   label='1:1 line')
        ax2.axline((0, lm_intercept), slope=lm_slope, color='r', label='OLS')
        ax2.axline((-2, lm_intercept), slope=lm_slope, color='r',
                   linewidth=0.6, label='OLS +- 2')
        ax2.axline((2, lm_intercept), slope=lm_slope, color='r', linewidth=0.6)
        ax2.text(0.7, 0.05, text_ax2, transform=ax2.transAxes)
        ax2.set_xlabel('Air temperature (MESAN) [oC]', fontsize=13)
        ax2.set_ylabel('Average surface water temperature [oC]', fontsize=13)
        ax2.tick_params(labelsize=10)
        ax2.legend(loc=2)
        fig.savefig(os.path.join(save_folder, f'{lake}_{freq}.jpg'))


def plot_air_and_surface_water_temperature_all_lakes(T_lakes):
    """
    """

    save_folder = '../EmpiricalModels_manuscript/figures_temp_air_water'

    for lake, d_lake in T_lakes.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        dt_meas = d_lake['T_water_meas'].dropna().index
        dt_meas_min = (dt_meas.min() - datetime(1970, 1, 1))\
                .total_seconds()/86400
        dt_meas_max = (dt_meas.max() - datetime(1970, 1, 1))\
                .total_seconds()/86400
        ax.set_title(lake, fontsize=20)
        ax.plot(d_lake['T_air_MESAN'], color='limegreen', linestyle='-')
        ax.plot(d_lake['T_water_meas'], color='cyan', linestyle='-')
        ax.plot(d_lake['T_water_pred'], color='dodgerblue', linestyle='--')
        ax.plot(d_lake['T_water_pred_rolling_avg'], color='blue',
                linestyle='--')
        ax.plot(d_lake['T_water'], color='red', linestyle=':')
        ax.axline((dt_meas_min, 10), (dt_meas_min, 15), color='grey',
                  linestyle=':')
        ax.axline((dt_meas_max, 10), (dt_meas_max, 15), color='grey',
                  linestyle=':')
        ax.set_ylabel('Temperature [oC]', fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend([
            'T$_{air}$ MESAN', 'T$_{wat}$ measured', 'T$_{wat}$ predicted',
            'T$_{wat}$ predicted rolling average', 'T$_{wat}$ combined'
        ])
        fig.savefig(os.path.join(save_folder, f'{lake}_comparison.jpg'))


def plot_comparison_gmx_vs_mesan(gmx, mesan, gmx_col, mesan_col, offset):
    """
    Create a scatterplot comparing GMX531 weather stations data with MESAN
    data for a given parameter. Add the equation of the best fit of a linear
    regression on the figure.

    GMX531 weather station data is resampled at an hour frequency to match
    MESAN data frequency.

    Parameters
    ----------
    gmx : pandas.DataFrame
        Table containing GMX531 weather station data with index containing
        datetime.
    mesan : pandas.DataFrame
        Table containing MESAN data with index containing datetime.
    gmx_col : str
        Column from the 'gmx' table to compare with MESAN data.
    gmx_col : str
        Column from the 'mesan' table to compare with GMX531 data.
    offset : float
        Offset from the main regression line where to draw parallel lines.
    """

    concat_df = pd.concat([gmx.resample('H').mean(), mesan], axis=1)

    data = concat_df[[gmx_col, mesan_col]].dropna()
    data_gmx = data[gmx_col].values.reshape(-1, 1)
    data_mesan = data[mesan_col].values
    reg = LinearRegression().fit(data_gmx, data_mesan)
    r2 = reg.score(data_gmx, data_mesan)
    if reg.intercept_ >= 0:
        text = (
            f'y = {reg.coef_[0]:.4f}*x + {reg.intercept_:.4f} '
            f'(R$^2$ = {r2:.4f})'
        )
    else:
        text = (
            f'y = {reg.coef_[0]:.4f}*x - {-reg.intercept_:.4f} '
            f'(R$^2$ = {r2:.4f})'
        )
    x_min, x_max = min(data_gmx), max(data_gmx)
    y_min, y_max = min(data_mesan), max(data_mesan)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(concat_df[gmx_col], concat_df[mesan_col])
    ax.axline((0, reg.intercept_), slope=reg.coef_[0], c='k')
    ax.axline(
        (0, reg.intercept_ + offset), slope=reg.coef_[0], c=(0.5, 0.5, 0.5)
    )
    ax.axline(
        (0, reg.intercept_ - offset), slope=reg.coef_[0], c=(0.5, 0.5, 0.5)
    )
    ax.set_title(text, fontsize=24)
    ax.set_xlabel(f'"{gmx_col}" in GMX531 data', fontsize=24)
    ax.set_ylabel(f'"{mesan_col}" in MESAN data', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_xlim(x_min - (x_max - x_min)/20, x_max + (x_max - x_min)/20)
    ax.set_ylim(y_min - (y_max - y_min)/20, y_max + (y_max - y_min)/20)
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

    return fig, ax


def plot_comparison_measured_vs_calculated_LWradiation_using_SMHI_data(
    smhi_station_T, smhi_station_LW
):
    """
    Compare longwave solar radiation measured at an SMHI station with values
    calculated from air temperature, relative humidity, cloud cover and
    altitude of cloud base using the methods described in Idso (1981) and
    Kimball (1982).

    Parameters
    ----------
    smhi_station_T : int
        ID of the SMHI station providing temperature measurements.
    smhi_station_LW : int
        ID of the SMHI station providing longwave radiation measurements.
    Available values for SMHI station IDs are as follows:
       LOCATION         smhi_station_T    smhi_station_LW
       Vaxjo                  64510             64565
       Visby                  78400             78645
       Norrkoping             86340             86655
       Svenska Hogarna        99280             99275
       Umea                  140480            140615
       Tarfala               178970            178985
       Kiruna                180940            180025

    References
    ----------
    Idso, S. B. (1981), A set of equations for full spectrum and 8- to 14-µm
    and 10.5- to 12.5-µm thermal radiation from cloudless skies, Water Resour.
    Res., 17(2), 295-304, doi:10.1029/WR017i002p00295.
    Kimball, B. A., Idso, S. B., and Aase, J. K. (1982), A model of thermal
    radiation from partly cloudy and overcast skies, Water Resour. Res., 18(4),
    931-936, doi:10.1029/WR018i004p00931.
    """

    # Load SMHI data
    print('Load SMHI temperature data...')
    T = SMHI_data.load_station_data(1, smhi_station_T, 'corrected-archive')
    T.rename(columns={'Value': 'T'}, inplace=True)
    print('Load SMHI relative humidity data...')
    RH = SMHI_data.load_station_data(6, smhi_station_T, 'corrected-archive')
    RH.rename(columns={'Value': 'RH'}, inplace=True)
    print('Load SMHI cloud cover data...')
    cl = SMHI_data.load_station_data(16, smhi_station_T, 'corrected-archive')
    cl.rename(columns={'Value': 'clc'}, inplace=True)
    print('Load SMHI cloud base data...')
    clb = SMHI_data.load_station_data(36, smhi_station_T, 'corrected-archive')
    clb.rename(columns={'Value': 'clb'}, inplace=True)
    print('Load SMHI longwave radiation data...')
    LW = SMHI_data.load_station_data(24, smhi_station_LW, 'corrected-archive')
    LW.rename(columns={'Value': 'LW'}, inplace=True)
    # Combine SMHI data
    data = T
    for d in [RH, cl, clb, LW]:
        data = pd.merge(data, d, on='Datetime', how='outer')
    data = data[['Datetime', 'T', 'RH', 'clc', 'clb', 'LW']]
    lw_clear, lw_total = calculate_LW_radiation(
        data['T'], data['RH'], data['clc'], data['clb']
    )
    data['LW_clear_[Wm-2]'] = lw_clear
    data['LW_total_[Wm-2]'] = lw_total

    data.rename(
        columns={
            'T': 'T_[oC]', 'RH': 'RH_[%]', 'clc': 'cloudcover_[-]',
            'clb': 'cloudbase_[m]', 'LW': 'LW_[Wm-2]',
        }, inplace=True
    )

    lw_c = data[['LW_[Wm-2]', 'LW_clear_[Wm-2]']].dropna()
    lw_c_meas = lw_c['LW_[Wm-2]'].values.reshape(-1, 1)
    lw_c_calc = lw_c['LW_clear_[Wm-2]']
    lw_t = data[['LW_[Wm-2]', 'LW_total_[Wm-2]']].dropna()
    lw_t_meas = lw_t['LW_[Wm-2]'].values.reshape(-1, 1)
    lw_t_calc = lw_t['LW_total_[Wm-2]']
    reg_clear = LinearRegression().fit(lw_c_meas, lw_c_calc)
    reg_total = LinearRegression().fit(lw_t_meas, lw_t_calc)
    r2_clear = reg_clear.score(lw_c_meas, lw_c_calc)
    r2_total = reg_total.score(lw_t_meas, lw_t_calc)

    plot_inputs = [
        ('clear sky', lw_c_meas, lw_c_calc, reg_clear, r2_clear),
        ('cloudy sky', lw_t_meas, lw_t_calc, reg_total, r2_total)
    ]

    for title, lw_meas, lw_calc, reg, r2 in plot_inputs:
        if reg.intercept_ >= 0:
            text = (
                f'y = {reg.coef_[0]:.4f}*x + {reg.intercept_:.4f} '
                f'(R$^2$ = {r2:.4f})'
            )
        else:
            text = (
                f'y = {reg.coef_[0]:.4f}*x - {-reg.intercept_:.4f} '
                f'(R$^2$ = {r2:.4f})'
            )
        x_min, x_max = min(lw_meas), max(lw_meas)
        y_min, y_max = min(lw_calc), max(lw_calc)

        # Create scatterplot to compare measured and calculated values
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(lw_meas, lw_calc)
        ax.axline((0, reg.intercept_), slope=reg.coef_[0], c='k')
        ax.axline(
            (0, reg.intercept_ + 50), slope=reg.coef_[0], c=(0.5, 0.5, 0.5)
        )
        ax.axline(
            (0, reg.intercept_ - 50), slope=reg.coef_[0], c=(0.5, 0.5, 0.5)
        )
        fig.suptitle(title, fontsize=28)
        ax.set_title(text, fontsize=24)
        ax.set_xlabel(
            f'Measured longwave radiation [W m$^{{-2}}$]', fontsize=24
        )
        ax.set_ylabel(
            f'Calculated longwave radiation [W m$^{{-2}}$]', fontsize=24
        )
        ax.tick_params(labelsize=20)
        ax.set_xlim(x_min - (x_max - x_min)/20, x_max + (x_max - x_min)/20)
        ax.set_ylim(y_min - (y_max - y_min)/20, y_max + (y_max - y_min)/20)
        ax.grid()
        ax.set_aspect('equal', adjustable='box')

        # Create histogram of difference between calculated and measured values
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist((lw_calc.values - lw_meas.T).T, bins=100)
        fig.suptitle(title)
        ax.set_title('Difference between calculated and measured values')
        ax.set_ylabel('W m$^{-2}$', fontsize=24)

    return data


def plot_scatterplot_final_table_yearly_data(d):
    """
    Create figures showing correlation between yearly data.

    Parameters
    ----------
    d : pd.DataFrame
        Data table returned by the function 'combine_final_table'.
    """

    catchment_vars = [
        'latitude', 'altitude', 'catchment_area', 'drainage_ratio',
        'landuse_openfield', 'landuse_forest', 'landuse_farmland',
        'landuse_openwetland', 'landuse_water', 'elevationslope_deg_mean',
        'air_temp_mesan', 'wind_speed_mesan', 'precipitation_mesan',
        'SWavg', 'gpp_catchment'
    ]
    lake_vars = [
        'lake_perimeter', 'lake_area', 'lake_volume',
        'mean_depth', 'max_depth', 'depth_ratio', 'residence_time',
        'storage', 'VFAN', 'TOC', 'TN', 'TP', 'chla_summer'
    ]
    ch4_vars = ['CH4', 'diff_flux', 'ebul_flux', 'tot_flux']

    pass
