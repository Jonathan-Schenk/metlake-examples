import os
import itertools
import subprocess
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import metlake_data
from metlake_utils import calculate_LW_radiation
from metlake_utils import relative_to_specific_humidity


def create_meteorological_forcing_data_table(
    lake, source='metlake', keep_only_longest_continuous_period=True,
    coef_ws=1.0, coef_swrad=1.0
):
    """
    Build a table containing meteorological forcing data to use in the LAKE
    model.

    Meteorological forcing data used in the LAKE model must contain:
    air temperature [K], specific humidity [kg/kg], barometric pressure [Pa],
    x-component of wind speed [m/s], y-component of wind speed [m/s],
    net shortwave radiation [W/m2], net longwave radiation [W/m2], and
    precipitation [m/s].

    Parameters
    ----------
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake for which the meteorological forcing file will be
        created.
    source : {'metlake', 'mesan'}
        Source of data.
        - 'metlake': Data obtained with the function 'combine_all_weather_data'
            in module 'metlake_data'.
        - 'mesan': Use MESAN and STRANG data only.
    keep_only_longest_continuous_period : bool, default: False
        Used only if source='metlake'.
        True: Select the longest period with continuous data available.
        False: Use the entire (discontinuous) data set.
    coef_ws : float, default: 1.0
        Multiplication coefficient to use with wind speed components.
    coef_swrad : float, default: 1.0
        Multiplication coefficient to use with shortwave solar radiation.
    """

    conv_lake = {
        'BD3': 'BD03', 'BD4': 'BD04', 'BD6': 'BD06',
        'PAR': 'Parsen', 'VEN': 'Venasjon', 'SOD': 'SodraTeden',
        'SGA': 'StoraGalten', 'GUN': 'Gundlebosjon', 'GRI': 'Grinnsjon',
        'LJE': 'LjusvattentjarnExp', 'LJR': 'LjusvattentjarnRef',
        'NAS': 'Nastjarn', 'NBJ': 'NedreBjorntjarn',
        'DAM': 'Dammsjon', 'NOR': 'Norrtjarn', 'GRA': 'Grastjarn',
        'LAM': 'Lammen', 'GYS': 'Gyslattasjon', 'KLI': 'Klintsjon'
    }

    if source == 'metlake':
        data = metlake_data.combine_all_weather_data(lake)
        data = data[[
            'temperature', 'humidity', 'pressure', 'u', 'v',
            'solar_radiation', 'lw_radiation', 'precipitation_intensity'
        ]].copy()
        if keep_only_longest_continuous_period:
            data = data.dropna()
            jumps = (data.index[1:] - data.index[:-1]) != timedelta(minutes=1)
            jumps = np.append(False, jumps).cumsum()
            data = data[jumps == np.bincount(jumps).argmax()]
        data['humidity'] = relative_to_specific_humidity(
            data['humidity'], data['pressure'], data['temperature']
        )
        data['temperature'] = data['temperature'] + 273.15
        data['pressure'] = data['pressure']*100
        data['precipitation_intensity'] = data['precipitation_intensity']/3.6e6
        data['u'] = coef_ws*data['u']
        data['v'] = coef_ws*data['v']
        data['solar_radiation'] = coef_swrad*data['solar_radiation']
    elif source == 'mesan':
        data = metlake_data.import_and_combine_mesan_and_strang_data(lake)
        data.rename(
            columns={'t': 'temperature', 'prec1h': 'precipitation_intensity'},
            inplace=True
        )
        data['humidity'] = relative_to_specific_humidity(
            data['Relative humidity'], data['Barometric pressure'],
            data['Temperature']
        )
        data['pressure'] = data['Barometric pressure']*100
        data['precipitation_intensity'] = data['precipitation_intensity']/3.6e6
        data = data[[
            'temperature', 'humidity', 'pressure', 'u', 'v',
            'solar_radiation', 'lw_radiation', 'precipitation_intensity'
        ]]
        data['u'] = coef_ws*data['u']
        data['v'] = coef_ws*data['v']
        data['solar_radiation'] = coef_swrad*data['solar_radiation']

    return data


def create_meteorological_forcing_data_file(data, filename):
    """
    Create a meteorological forcing data file to use in the LAKE model.

    Parameters
    ----------
    data : pd.DataFrame
        Data table returned by the function
        'create_meteorological_forcing_data_table'.
    filename : str
        Full or relative path to the new file to create, including file name.
    """

    with open(filename, 'w') as f:
        for col in data.columns:
            f.write(f'{col:>30}')
        f.write('\n')
        for ind, row in data.iterrows():
            for v in row:
                f.write(f'{v:>30.12f}')
            f.write('\n')


def linear_regressions_metlake_vs_mesan(lakes=None):
    """
    Calculate linear regression coefficients between weather data collected
    in the METLAKE project and MESAN/STRANG data.

    Parameters
    ----------
    lakes : list or None, default: None
        List of lakes to process.
        Possibilities are:
        'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN', 'LJE',
        'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM', 'KLI'.
    """

    if lakes is None:
        lakes = [
            'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'
        ]

    params = [
        'temperature', 'humidity', 'pressure', 'u', 'v', 'solar_radiation',
        'lw_radiation', 'precipitation_intensity', 'wind_speed', 'wind_dir'
    ]

    data = {}
    for lake in lakes:
        print(lake)
        met = create_meteorological_forcing_data_table(lake, 'metlake', False)
        mes = create_meteorological_forcing_data_table(lake, 'mesan')
        met['wind_speed'] = np.sqrt(met['u']**2 + met['v']**2)
        met['wind_dir'] = (np.arctan2(met['v'], met['u'])*180/np.pi)%360
        met_sum = met.resample('H').sum()
        met = met.resample('H').mean()
        met['precipitation_intensity'] = met_sum['precipitation_intensity']
        met.index.name = 'datetime'
        met.rename(columns=lambda x: x + '_metlake', inplace=True)
        mes.index = mes.index + timedelta(hours=2)
        mes['wind_speed'] = np.sqrt(mes['u']**2 + mes['v']**2)
        mes['wind_dir'] = (np.arctan2(mes['v'], mes['u'])*180/np.pi)%360
        mes.rename(columns=lambda x: x + '_mesan', inplace=True)
        data[lake] = pd.merge(met, mes, on='datetime', how='inner')

    slope = pd.DataFrame(index=lakes, columns=params, dtype='float64')
    intercept = pd.DataFrame(index=lakes, columns=params, dtype='float64')
    r2 = pd.DataFrame(index=lakes, columns=params, dtype='float64')
    for lake in lakes:
        for p in params:
            d = data[lake][[f'{p}_metlake', f'{p}_mesan']].dropna()
            if d.shape[0] == 0:
                slope.loc[lake, p] = np.nan
                intercept.loc[lake, p] = np.nan
                r2.loc[lake, p] = np.nan
                continue
            X = d[f'{p}_metlake'].values.reshape(-1, 1)
            y = d[f'{p}_mesan'].values
            reg = LinearRegression().fit(X, y)
            score = reg.score(X, y)
            slope.loc[lake, p] = reg.coef_[0]
            intercept.loc[lake, p] = reg.intercept_
            r2.loc[lake, p] = score

    return data, slope, intercept, r2


def compare_wind_speed_mesan_vs_metlake(
    mesan, metlake, lake, t_start, t_end, make_fig
):
    """
    Create a scatterplot and calculate a linear regression of MESAN vs.
    METLAKE wind speeds.

    Parameters
    ----------
    mesan : pd.DataFrame
        Table containing MESAN data. Format should be similar to format of
        tables returned by the function 'metlake_data.import_mesan_data'.
    metlake : pd.DataFrame
        Table containing METLAKE weather data. Format should be similar to
        format of tables returned by the function
        'metlake_data.combine_all_weather_data'.
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake
    t_start : datetime.datetime
        Beginning of the period to use for comparison.
    t_end : datetime.datetime
        End of the period to use for comparison.
    make_fig : bool
        True: Create two figures comparing MESAN and METLAKE wind data.
        False: Do not create any figure.
    """

    ws_met = metlake.loc[
        np.logical_and(metlake.index >= t_start, metlake.index <= t_end),
        ['u', 'v']
    ]
    ws_met = np.sqrt(np.square(ws_met['u']) + np.square(ws_met['v']))
    ws_met.index = ws_met.index - timedelta(hours=2)
    ws_met = ws_met.resample('H').mean()
    ws_mes = mesan.loc[
        np.logical_and(mesan.index >=t_start, mesan.index <= t_end),
        'Wind speed'
    ]
    ws = pd.DataFrame(columns=['mesan', 'metlake'], index=ws_mes.index)
    ws['mesan'] = ws_mes
    ws['metlake'] = ws_met
    ws.dropna(inplace=True)
    reg = LinearRegression().fit(
        ws['mesan'].values.reshape(-1, 1), ws['metlake'].values
    )
    r2 = reg.score(ws['mesan'].values.reshape(-1, 1), ws['metlake'].values)
    reg_no_intercept = LinearRegression(fit_intercept=False).fit(
        ws['mesan'].values.reshape(-1, 1), ws['metlake'].values
    )
    r2_no_intercept = reg_no_intercept.score(
        ws['mesan'].values.reshape(-1, 1), ws['metlake'].values
    )
    if make_fig:
        fig1, ax1 = plt.subplots()
        ax1.plot(ws.index, ws['mesan'], alpha=0.7, label='mesan')
        ax1.plot(ws.index, ws['metlake'], alpha=0.7, label='metlake')
        ax1.set_ylabel('Wind speed (m/s)')
        ax1.set_title(lake)
        ax1.legend()
        fig2, ax2 = plt.subplots()
        fig2.suptitle(lake)
        ax2.set_title((
            f'slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}, '
            f'R$^2$={r2:.3f}'
        ))
        ws.plot('mesan', 'metlake', 'scatter', ax=ax2)
        ax2.axline((0, reg.intercept_), slope=reg.coef_, c='k')
        ax2.grid()
        return ws, reg, r2, reg_no_intercept, r2_no_intercept,\
                fig1, ax1, fig2, ax2
    else:
        return ws, reg, r2, reg_no_intercept, r2_no_intercept


def create_parameter_dictionaries(
    lake, dataname, tinteg, dt, height_T_q=2.0, height_u=10.0, interval=1.0,
    fetch=1000.0, cellipt=2.0, lakeform=1, Ts0=15.0, Tb0=5.0, Tbb0=5.0, Tm=10.0,
    h_ML0=1.5, Sals0=1e-3, Salb0=30e-3, soiltype=5, soil_depth=1.0,
    backdiff0=-1.0, VmaxCH4aeroboxid=-1.0, khsCH4=-1.0, khsO2=-1.0,
    r0methprod=-1.0, accum_begin=2000010100, accum_end=2100010100, **kwargs
):
    """
    Create two dictionaries containing parameters to run the LAKE model.

    Parameters
    ----------
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake for which the parameter dictionaries will be created.
    dataname : str
        Project name (used for meteo, setup and driver file names).
    tinteg : int
        Duration of the integration, days.
    dt : int
        Time step of integration, s.
    height_T_q : float, default: 2.0
        Height at which temperature and humidity are given, m.
    height_u : float, default: 10.0
        Height at which wind speed is given, m.
    interval : float, default: 1.0
        Time step of meteorological data, hour.
    fetch : float, default: 1000.0
        Average wind fetch, m.
    cellipt : float, default: 2.0
        The ratio of x dimension to y dimension of the lake.
    lakeform : int, default: 1
        Form of the lake's horizontal cross-section. 1 - ellipse, 2 - rectangle.
    Ts0 : float, default: 15.0
        Initial temperature of the top mixed layer, oC.
        Not used if init_T=3, which is the case by default.
    Tb0 : float, default: 5.0
        Initial bottom temperature, oC.
        Not used if init_T=3, which is the case by default.
    Tbb0 : float, default: 5.0
        Initial temperature at the bottom of the soil layer, oC.
        Not used if init_T=3, which is the case by default.
    Tm : float, default: 10.0
        Initial mean temperature of the water column, oC.
        Not used if init_T=3, which is the case by default.
    h_ML0 : float, default: 1.5
        Initial mixed layer depth, m.
        Not used if init_T=3, which is the case by default.
    Sals0 : float, default: 1e-3
        Initial salinity in the mixed layer, kg/kg.
        Not used if init_T=3, which is the case by default.
    Salb0 : float, default: 30e-3
        Initial bottom salinity, kg/kg.
        Not used if init_T=3, which is the case by default.
    soiltype : int, default: 5
        Soil type. 1 - sand, 2 - loamy sand, 3 - sandy loam, 4 - loam,
        5 - silt loam, 6 - sandy clay loam, 7 - clay loam,
        8 - silty clay loam, 9 - sandy clay, 10 - silty clay, 11 - clay.
    soil_depth : float, default: 1.0
        Depth of the soil layer.
    backdiff0 : float, default: -1.0
        A multiplier in background diffusivity expression by Hondzo and Stefan
        (1993), relevant if backdiff=1; if backdiff0<0., then a reference
        value from original paper is used, which is 8.17e-4.
    VmaxCH4aeroboxid : float, default: -1.0
        Maximal methane aerobic oxidation rate in Monod equation throughout
        water column; if VmaxCH4aeroboxid<0., then a reference value is used,
        which is 1.e-2/86400 = 1.1574e-7 mol m-3 s-1.
    khsCH4 : float, default: -1.0
        Methane half-saturation constant in methane aerobic oxidation rate
        in Monod equation throughout water column; if khsCH4<0., then
        a reference value is used, which is 0.6/16 = 0.0375 mol m-3.
    khsO2 : float, default: -1.0
        Oxygen half-saturation constant in methane aerobic oxidation rate
        in Monod equation throughout water column; if khsO2<0., then
        a reference value is used, which is 0.672/32 = 0.021 mol m-3.
    r0methprod : float, default: -1.0
        A constant multiplier in the expression for methane production
        in sediments; if r0methprod<0., then a reference value is used, which
        is 6.e-8 mol m-3 s-1.
    accum_begin : int, default: 2000010100
        Date and time for beginning accumulating fluxes.
    accum_end : int, default: 2100010100
        Date and time for finishing accumulating fluxes.
    **kwargs
        Pairs of (parameter: value) to use in the driver or setup files.
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
    date_profile = {
        'BD3': datetime(2018, 6, 29), 'BD4': datetime(2018, 6, 30),
        'BD6': datetime(2018, 7, 1), 'PAR': datetime(2018, 5, 23),
        'VEN': datetime(2018, 5, 8), 'SOD': datetime(2018, 5, 9),
        'SGA': datetime(2019, 3, 22), 'GUN': datetime(2019, 3, 25),
        'GRI': datetime(2019, 3, 23),
        'LJE': datetime(2019, 6, 6), 'LJR': datetime(2019, 6, 6),
        'NAS': datetime(2019, 6, 7), 'NBJ': datetime(2019, 6, 7),
        'DAM': datetime(2020, 4, 7), 'NOR': datetime(2020, 4, 6),
        'GRA': datetime(2020, 4, 7), 'LAM': datetime(2020, 4, 23),
        'GYS': datetime(2020, 4, 22), 'KLI': datetime(2020, 4, 21)
    }
    lake_length = {
        'BD3': 580, 'BD4': 220, 'BD6': 510, 'PAR': 770, 'VEN': 2150,
        'SOD': 1800, 'SGA': 1000, 'GUN': 1200, 'GRI': 720, 'LJE': 130,
        'LJR': 180, 'NAS': 150, 'NBJ': 280, 'DAM': 470, 'NOR': 250,
        'GRA': 580, 'LAM': 2000, 'GYS': 1200, 'KLI': 820
    }
    #backdiff = {
    #    'BD3': 1, 'BD4': 0, 'BD6': 1, 'PAR': 1, 'VEN': 1, 'SOD': 1,
    #    'SGA': 1, 'GUN': 1, 'GRI': 1, 'LJE': 0, 'LJR': 0, 'NAS': 0, 'NBJ': 0,
    #    'DAM': 1, 'NOR': 0, 'GRA': 1, 'LAM': 1, 'GYS': 1, 'KLI': 1
    #}

    params_driver = {
        # Information on the file with atmospheric data
        'dataname': None, 'forc_format': 0, 'npoints': 1, 'select_call': 0,
        'lakinterac': 1, 'form': 0, 'height_T_q': None, 'height_u': None,
        'interval': None, 'rad': 1, 'N_header_lines': 1, 'N_coloumns': 8,
        'N_Year': -1, 'N_Month': -1, 'N_Day': -1, 'N_Hour': -1, 'N_Uspeed': 4,
        'N_Vspeed': 5, 'N_Temp': 1, 'N_Hum': 2, 'N_Pres': 3, 'N_SWdown': 6,
        'N_LWdown': 7, 'N_Precip': 8, 'N_SensFlux': -1, 'N_LatentFlux': -1,
        'N_Ustar': -1, 'N_surfrad': -1, 'N_NetRad': -1, 'N_cloud': -1,
        'N_SurfTemp': -1,
        # Time integration parameters
        'year0': None, 'month0': None, 'day0': None, 'hour0': None,
        'tinteg': None, 'spinup_times': 0, 'spinup_period': 0,
        'control_point': 0, 'cp_period': 0, 'dt': None, 'call_Flake': 0,
        # Physical parameters
        'extwat': None, 'extice': 1e7, 'alphax': 0.0, 'alphay': 0.0,
        'a_veg': 1.0, 'c_veg': 1e-3, 'h_veg': 0.0, 'kor': -999.0,
        'phi': None, 'lam': None, 'fetch': None,
        # Lake parameters
        'area_lake': None, 'cellipt': None, 'lakeform': None,
        'trib_inflow': -9999.0, 'morphometry': None, 'effl_outflow': None,
        # Initial conditions
        'l10': 0.0, 'h10': None, 'select_h10': 0, 'ls10': 0.0, 'hs10': 0.0,
        'Ts0': None, 'Tb0': None, 'Tbb0': None, 'Tm': None, 'h_ML0': None,
        'Sals0': None, 'Salb0': None, 'us0': 1e-3, 'vs0': 1e-3, 'init_T': 3,
        # NetCDF output parameter
        'nstep_ncout': -1,
        # FLAKE model output parameter
        'nstep_out_Flake': 3,
        # Postprocessing options
        'moving_average_window': -1, 'mean_cycle_period': -1
    }
    params_setup = {
        # General controls
        'path': "''", 'runmode': 1, 'omp': 0,
        # Spatial resolution of the model
        'nstep_keps': 1, 'M': None, 'ns': 10, 'nsoilcols': 5, 'Mice': 10,
        'd_surf': 1e-2, 'd_bot': 1e-2,
        # Controls for physics of the model
        'varalb': 1, 'PBLpar': 3, 'waveenh': 0, 'momflxpart': 1, 'c_d': -999,
        'kwe': 100.0, 'relwind': 0, 'eos': 5, 'lindens': 0, 'nmeltpoint': 1,
        'Turbpar': 2, 'stabfunc': 2, 'kepsbc': 1, 'soiltype': None,
        'soil_depth': None, 'soilswitch': 1, 'saltice': 0, 'tricemethhydr': 0.0,
        'skin': 0, 'massflux': 0, 'ifrad': 1, 'ifbubble': 1, 'carbon_model': 2,
        'sedim': 0, 'salsoil': 0, 'dyn_pgrad': 0, 'pgrad': 0.0, 'botfric': 1,
        'horvisc': 0.0, 'backdiff': None, 'backdiff0': None, 'nManning': 5e-2,
        'zero_model': 0, 'thermokarst_meth_prod': 0.0, 'soil_meth_prod': 1.0,
        'VmaxCH4aeroboxid': None, 'khsCH4': None, 'khsO2': None,
        'r0methprod': None, 'outflpar': 0, 'sensflux0': 100.0,
        'momflux0': 1e-15, 'soilbotflx': 0.0, 'cuette': 0, 'deadvol': 0.0,
        # Initial conditions
        'T_profile': None, 'T_soilprofile': None,
        # Tributaries and effluents
        'tribheat': 0, 'N_tribin': 0, 'N_triblev': 0, 'iefflloc': 1,
        'fileinflow': "''", 'fileoutflow': "''", 'dttribupdate': 0.25,
        # Data assimilation controls
        'assim': 0, 'error_cov': 0,
        # Output controls
        'turb_out': 0, 'monthly': 0, 'daily': 0, 'hourly': 0,
        'everystep': 0, 'time_series': 1, 'zserout': -999.0, 'dt_out': 4.0,
        'nscreen': int(3600/dt), 'scale_output': 0, 'accum_begin': None,
        'accum_end': None, 'ngrid_out': None, 'ngridice_out': 0,
        'ngridsoil_out': None, 'rtemp': None
    }

    # Get lake bathymetry and area and calculate fetch and cellipt
    info_lakes, area_lakes, volume_lakes = metlake_data.import_info_lakes()
    info = info_lakes.loc[info_lakes['Lake'] == lake].squeeze()
    area_lakes.columns = area_lakes.columns.droplevel(1)
    area_lakes['depth'] = area_lakes.index
    area = area_lakes[['depth', lake]].dropna()
    area = area[area[lake] > 0]
    area = area.iloc[::int(np.ceil(len(area)/100))]
    depth_max = info['LakeDepth_Max_[m]']
    lake_width = info['LakeArea_[m2]']/(np.pi*lake_length[lake]/2)*2
    fetch = (lake_length[lake] + lake_width)/2
    cellipt = lake_length[lake]/lake_width
    # Create an effluent with bottom half a meter below the water surface
    effl = pd.DataFrame([0.0, 0.0, depth_max - 0.5]).T
    # Get depth profiles data and extract light absorbance coefficient and
    # inital profile of temperature, salinity, CH4aq, CO2aq (calculated as
    # 0.3*DIC, which is the case for pH=6.79), dissolved oxygen and phosphorus.
    dp = metlake_data.import_depth_profiles_data()
    absorbance = metlake_data.calculate_absorbance_from_profiles(dp)
    absorbance_value = absorbance.groupby('Lake').median().loc[
        conv_lake[lake], 'Absorbance_m-1'
    ]
    col_depth = ('General', 'Depth', 'm')
    col_T = ('LI-COR and HACH probes', 'WaterTemperature', 'oC')
    col_S = ('Aquaread', 'SAL', 'PSU')
    col_CH4 = ('CH4, DIC, N2O aq conc', 'CH4aq', 'µM')
    col_CO2 = ('CH4, DIC, N2O aq conc', 'CO2aq', 'µM')
    col_DO = ('LI-COR and HACH probes', 'DissolvedOxygen', 'mg/L')
    dp_lake = dp[np.logical_and(
        dp[('General', 'Lake', 'Unnamed: 1_level_2')] == conv_lake[lake],
        dp[('General', 'Date', 'Unnamed: 2_level_2')] == date_profile[lake]
    )].copy()
    if all(np.isnan(dp_lake[col_T])):
        col_T = ('Aquaread', 'Temp', 'C')
    if all(np.isnan(dp_lake[col_DO])):
        col_DO = ('Aquaread', 'DO', 'mg/L')
    cols_keep = [col_depth, col_T, col_S, col_CH4, col_CO2, col_DO]
    dp_lake = dp_lake[cols_keep].copy()
    dp_lake.columns = ['depth', 'T', 'S', 'CH4', 'CO2', 'DO']
    dp_lake['S'] = dp_lake['S']/1e3
    dp_lake['CH4'] = dp_lake['CH4']/1e3
    dp_lake['CO2'] = 0.3*dp_lake['CO2']/1e3
    dp_lake['DO'] = dp_lake['DO']/32
    dp_lake['P'] = 0.0
    if all(np.isnan(dp_lake['S'])):
        dp_lake['S'] = 0.0
    if lake in ['PAR', 'VEN', 'SOD']:
        dp_lake_interp = dp_lake.set_index('depth').interpolate('index')
        dp_lake[['T', 'DO']] = dp_lake_interp[['T', 'DO']].values
    dp_lake.dropna(inplace=True)
    # Create simple initial temperature profile in soil
    T_soilprofile = pd.DataFrame(
        [[0.0, 4.0], [soil_depth, 4.0]]
    )
    # Parameters setting at which depths outputs must be returned
    depths_out_surface = [0.0, 0.05, 0.33, 0.66, 1.0]
    if depth_max < 4:
        depths_out_column = np.arange(1.25, depth_max, 0.25)
    elif depth_max >= 4 and depth_max < 15:
        depths_out_column = np.arange(1.5, depth_max, 0.5)
    else:
        depths_out_column = np.arange(2.0, depth_max, 1.0)
    depths_out = np.concatenate([depths_out_surface, depths_out_column])
    dgrid_out = pd.DataFrame(depths_out)
    dgridsoil_out = pd.DataFrame(np.linspace(0, 0.9*soil_depth, 3))
    rtemp = pd.DataFrame([[0.0, 0.0, d] for d in depths_out])

    params_driver['dataname'] = repr(dataname)
    params_driver['height_T_q'] = height_T_q
    params_driver['height_u'] = height_u
    params_driver['interval'] = interval
    params_driver['year0'] = date_profile[lake].year
    params_driver['month0'] = date_profile[lake].month
    params_driver['day0'] = date_profile[lake].day
    params_driver['hour0'] = 12.0
    params_driver['tinteg'] = tinteg
    params_driver['dt'] = dt
    params_driver['extwat'] = absorbance_value
    params_driver['phi'] = info['North coordinate']
    params_driver['lam'] = info['East coordinate']
    params_driver['fetch'] = fetch
    params_driver['area_lake'] = info['LakeArea_[m2]']
    params_driver['cellipt'] = cellipt
    params_driver['lakeform'] = lakeform
    params_driver['morphometry'] = [len(area), area]
    params_driver['effl_outflow'] = [effl.shape[1] - 2, effl]
    params_driver['h10'] = depth_max
    params_driver['Ts0'] = Ts0
    params_driver['Tb0'] = Tb0
    params_driver['Tbb0'] = Tbb0
    params_driver['Tm'] = Tm
    params_driver['h_ML0'] = h_ML0
    params_driver['Sals0'] = Sals0
    params_driver['Salb0'] = Salb0
    params_driver.pop('select_call')
    params_driver.pop('select_h10')
    params_setup['M'] = max(20, int(depth_max/0.5))
    params_setup['soiltype'] = soiltype
    params_setup['soil_depth'] = soil_depth
    params_setup['backdiff'] = 1 if info['LakeArea_[m2]'] >= 5e4 else 0
    params_setup['backdiff0'] = backdiff0
    params_setup['VmaxCH4aeroboxid'] = VmaxCH4aeroboxid
    params_setup['khsCH4'] = khsCH4
    params_setup['khsO2'] = khsO2
    params_setup['r0methprod'] = r0methprod
    params_setup['T_profile'] = [dp_lake.shape[0], dp_lake]
    params_setup['T_soilprofile'] = [T_soilprofile.shape[0], T_soilprofile]
    params_setup['accum_begin'] = accum_begin
    params_setup['accum_end'] = accum_end
    params_setup['ngrid_out'] = [dgrid_out.shape[0], dgrid_out]
    params_setup['ngridsoil_out'] = [dgridsoil_out.shape[0], dgridsoil_out]
    params_setup['rtemp'] = [len(rtemp), rtemp]

    for p, v in kwargs.items():
        if p in params_driver.keys():
            params_driver[p] = v
        elif p in params_setup.keys():
            params_setup[p] = v

    return params_driver, params_setup


def create_parameter_file(path_to_file, params):
    """
    Create a driver or setup file to use in the LAKE model.

    Parameters
    ----------
    path_to_file : str
        Full or relative path to the driver or setup file.
    params : dict
        Dictonary where each (key, value) pair contains a parameter name
        and its corresponding value for the parameter file.
    """

    with open(path_to_file, 'w') as f:
        for p, v in params.items():
            v_table = None
            if isinstance(v, list):
                v_table = v[1]
                v = v[0]
            if isinstance(v, int):
                f.write(f'{p:25}{v:<20d}\n')
            elif isinstance(v, float):
                f.write(f'{p:25}{v:<20.10f}\n')
            else:
                f.write(f'{p:25}{v:<}\n')
            if v_table is not None:
                for ind, row in v_table.iterrows():
                    for n in row:
                        if isinstance(n, int):
                            f.write(f'{n:<20d}')
                        elif isinstance(n, float):
                            f.write(f'{n:<20.10f}')
                        else:
                            f.write(f'{n:<20}')
                    f.write('\n')
        f.write('end')


def compare_parameter_files(lakes=None, runs=['001', '002'], ver='2.6'):
    """
    Print lines that differ in parameter files of different runs.

    Parameters
    ----------
    lakes : list or None, default: None
        List of lakes for which files will be compared.
        Possibilities are:
        'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN', 'LJE',
        'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM', 'KLI'.
    runs : list, default: ['001', '002']
        List of two IDs representing runs to compare.
    ver : str, default: '2.6'
        Version of the LAKE model that was used to run the model.
    """

    path_files = f'/home/jonathan/OneDrive/VM/Metlake/Models/LAKE{ver}/setup'

    if lakes is None:
        lakes = [
            'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'
        ]

    for lake in lakes:
        print(f'\n\n{lake}')
        d = {}
        s = {}
        for n, run in enumerate(runs):
            driver = f'{lake}_run{run}_driver.dat'
            setup = f'{lake}_run{run}_setup.dat'
            with open(os.path.join(path_files, driver)) as driver:
                d[n] = driver.read().split('\n')
            with open(os.path.join(path_files, setup)) as setup:
                s[n] = setup.read().split('\n')
        if len(d[0]) != len(d[1]):
            print('   Not equal number of lines in driver files.')
        if len(s[0]) != len(s[1]):
            print('   Not equal number of lines in setup files.')
        for n in range(len(d[0])):
            if d[0][n] != d[1][n]:
                print(d[0][n])
                print(d[1][n])
        for n in range(len(s[0])):
            if s[0][n] != s[1][n]:
                print(s[0][n])
                print(s[1][n])


def run(
    lake, proj_name, tinteg, dt, ver='2.6', coef_ws=1.0, coef_swrad=1.0,
    **kwargs
):
    """
    Run the LAKE model for a given set of parameters.

    Parameters
    ----------
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake.
    proj_name : str
        Project name. This name will be used to create the meteorological
        forcing data file and the driver and setup files. It is also the name
        of the folder where the results of the model will be stored.
    tinteg : int
        Duration of the integration, days.
    dt : int
        Time step of integration, s.
    ver : str, default: '2.6'
        Version of the LAKE model to use.
    coef_ws : float, default: 1.0
        Multiplication coefficient to use with wind speed components.
    coef_swrad : float, default: 1.0
        Multiplication coefficient to use with shortwave solar radiation.
    **kwargs
        Pairs of (parameter: value) to use in the driver or setup files.
    """

    # Create parameter dictionaries
    params_driver, params_setup = create_parameter_dictionaries(
        lake, proj_name, tinteg, dt, **kwargs
    )

    # Create meteorological data table
    meteo = create_meteorological_forcing_data_table(
        lake, 'mesan', coef_ws=coef_ws, coef_swrad=coef_swrad
    )
    year = params_driver['year0']
    month = params_driver['month0']
    day = params_driver['day0']
    hour = params_driver['hour0']
    dt_ini = datetime(year, month, day) + timedelta(hours=hour)
    dt_fin = dt_ini + timedelta(days=params_driver['tinteg'])
    meteo = meteo[np.logical_and(meteo.index >= dt_ini, meteo.index <= dt_fin)]

    # Move to LAKE directory
    os.chdir(f'/home/jonathan/OneDrive/VM/Metlake/Models/LAKE{ver}')

    # Create meteorological forcing file
    create_meteorological_forcing_data_file(meteo, f'./data/{proj_name}.dat')

    # Create driver and setup files
    create_parameter_file(f'./setup/{proj_name}_driver.dat', params_driver)
    create_parameter_file(f'./setup/{proj_name}_setup.dat', params_setup)

    # Run the model
    subprocess.run(['./crproj', proj_name])
    subprocess.run('./lake.out')

    # Return to the original folder
    os.chdir('/home/jonathan/OneDrive/VM/Metlake/PythonScripts')


def run_calibration_set(lake, n0, tinteg, dt, params, ver, **kwargs):
    """
    Run the LAKE model multiple times with different values for some parameters.

    WARNING: Potentially very long run time and large disk space used.

    Parameters
    ----------
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake.
    n0 : int
        Runs will be named '{lake}_run{str(n).zfill(3)}' with n starting at n0.
    tinteg : int
        Duration of the integration, days.
    dt : int
        Time step of integration, s.
    params : dict
        Dictionary where each key is the name of a parameter used by the LAKE
        model and the corresponding variable is a list containing the values
        to use for this parameter.
    ver : str
        Version of the LAKE model to use.
    **kwargs
        Pairs of (parameter: value) to use in the driver or setup files.
        In contrary to 'params', the values of parameters passed to kwargs
        are identical for all runs.
    """

    params = pd.DataFrame(params)
    params = pd.DataFrame(
        itertools.product(*params.values.T), columns=params.columns
    )
    params.index += n0

    timestamp_now = datetime.now().strftime("%Y%m%d%H%M")
    params.to_csv(os.path.join(
        '/home/jonathan/OneDrive/VM/Metlake/LAKE_results/',
        f'LAKE_ParameterTableBatchRun_{lake}_{timestamp_now}'
    ))

    for n, row in params.iterrows():
        p = dict(row)
        p.update(kwargs)
        run(lake, f'{lake}_run{str(n).zfill(3)}', tinteg, dt, ver, **p)


def import_time_vs_depth_file(path_to_file):
    """
    Import a "time_series" result file containing data at different depths.

    Parameters
    ----------
    path_to_file : str
        Full or relative path to the file of interest.
    """

    data = pd.read_csv(
        path_to_file, header=None, skiprows=6, delim_whitespace=True,
        na_values=-999
    )
    depths = [set(data[n]) for n in range(5, data.shape[1], 2)]
    if any([max(vals) - min(vals) > 0.05 for vals in depths]):
        print(
            f'More than 5 cm variation for the depth of one or more levels '
            f'over time in file {path_to_file}'
        )
    depths = [np.median(list(vals)) for vals in depths]
    dt = data.apply(
        lambda x: datetime(int(x[0]), int(x[1]), int(x[2])) + \
                timedelta(hours=x[3]),
        axis=1
    )
    data = data.iloc[:, 6::2].values
    data = pd.DataFrame(data=data, index=dt, columns=depths, dtype='float64')

    return data


def import_methane_time_series_file(path_to_file):
    """
    Import a "methane_series" result file.

    Parameters
    ----------
    path_to_file : str
        Full or relative path to the file of interest.
    """

    data = pd.read_csv(
        path_to_file, header=None, skiprows=32, delim_whitespace=True,
    )
    dt = data.apply(
        lambda x: datetime(int(x[0]), int(x[1]), int(x[2])) + \
                timedelta(hours=x[3]),
        axis=1
    )
    columns = [
        'ElapsedTime_[h]', 'TalikDepth_[m]', 'CH4aq,surf_[mol*m-3]',
        'CH4aq,bott_[mol*m-3]', 'CH4soil,bott_[mol*m-3]',
        'O2aq,surf_[mol*m-3]', 'O2aq,bott_[mol*m-3]',
        'CH4prod,youngC_[mol*m-2*s-1]', 'CHprod,oldC_[mol*m-2*s-1]',
        'FCH4ebul,surf_[mol*m-2*s-1]', 'FCH4plant,bott_[mol*m-2*s-1]',
        'FCH4diff,bott_[mol*m-2*s-1]', 'FCH4turb,surf_[mol*m-2*s-1]',
        'FCH4ebul,surf_[mg*m-2*d-1]', 'FCH4plant,bott_[mg*m-2*d-1]',
        'FCH4diff,bott_[mg*m-2*d-1]', 'FCH4turb,surf_[mg*m-2*d-1]',
        'FCH4turb,MLbott_[mg*m-2*d-1]', 'FCH4sed,ML_[mg*m-2*d-1]',
        'FCH4bubble,MLbott_[mg*m-2*d-1]',
        'MOX,lake_[mg*m-2*d-1]', 'MOX,ML_[mg*m-2*d-1]',
        'FCO2turb,surf_[mol*m-2*s-1]', 'FCO2ebul,surf_[mol*m-2*s-1]',
        'FO2turb,surf_[mol*m-2*s-1]', 'FO2ebul,surf_[mol*m-2*s-1]',
        'FCH4,outlet_[mol*m-2*s-1]'
    ]
    info_columns = {}
    with open(path_to_file, 'r') as f:
        # Iterate through all header lines. Discard the four first ones
        # containing year, month, day and hour. Do not use the last one
        # immediately but keep it for the next 'for' loop.
        for n in range(32):
            line = f.readline()
            if n >= 4 and n < 31:
                info_columns[columns[n-4]] = line.split(' - ')[1][:-1]
    for n in range(1, data.shape[1]-30):
        col = f'FCH4ebul,soilcol{n}_[mg*m-2*d-1]'
        columns.append(col)
        info_columns[col] = line.split(' - ')[1][:-1]
    data = data.iloc[:, 4:].values
    data = pd.DataFrame(data=data, index=dt, columns=columns, dtype='float64')

    return data, info_columns


def import_and_homogenize_model_and_measured_T(
    file_T_model, file_T_measured, depths_interp='loggers'
):
    """
    Import predicted and measured water temperature and adjust to the same grid.

    Parameters
    ----------
    file_T_model : str
        Full or relative path to a file containing modeled temperature.
    file_T_measured : str
        Full or relative path to a thermistor chain file.
    depths_interp : {'loggers', 'model'}, default: 'loggers'
        loggers: Interpolate model results on the loggers' depths.
        model: Interpolate loggers' measurements on the model output's depths.
    """

    # Import data
    T_model = import_time_vs_depth_file(file_T_model)
    T_meas = metlake_data.import_thermistor_chain_data(
        file_T_measured, 'dataframe', T_model.index[0].year
    )
    # Discard measurements from HOBO located above the deepest RBR
    col_keep = [T_meas.columns[0]]
    for n, col in enumerate(T_meas.columns[1:]):
        if all(col > T_meas.columns[:n+1]):
            col_keep.append(col)
    T_meas = T_meas[col_keep]
    # Interpolate
    if depths_interp == 'loggers':
        f_interp = interp2d(T_model.columns, T_model.index, T_model.values)
        T_model_interp = pd.DataFrame(
            data=f_interp(
                T_meas.columns,
                (T_meas.index - datetime(1970, 1, 1, 0, 0)).total_seconds()*1e9
            ), columns=T_meas.columns, index=T_meas.index
        )
        return T_model_interp, T_meas
    elif depths_interp == 'model':
        # The function "interp2d" does not work if the input array contains NaN
        # values. Here, the measurements are first interpolated to fill gaps
        # and then interpolated on the new grid. NaN values are reintroduced
        # afterwards using a "NaN mask" that is also interpolated.
        T_meas_interp = T_meas.interpolate(
            axis=1, limit=4, limit_direction='both'
        )
        T_meas_nan = np.zeros_like(T_meas.values)
        T_meas_nan[np.isnan(T_meas.values)] = 1
        f_interp = interp2d(T_meas.columns, T_meas.index, T_meas_interp.values)
        f_nan_interp = interp2d(T_meas.columns, T_meas.index, T_meas_nan)
        T_meas_interp = pd.DataFrame(
            data=f_interp(
                T_model.columns,
                (T_model.index - datetime(1970, 1, 1, 0, 0)).total_seconds()*1e9
            ), columns=T_model.columns, index=T_model.index
        )
        T_meas_nan_interp = pd.DataFrame(
            data=f_nan_interp(
                T_model.columns,
                (T_model.index - datetime(1970, 1, 1, 0, 0)).total_seconds()*1e9
            ), columns=T_model.columns, index=T_model.index
        )
        T_meas_interp[T_meas_nan_interp > 0] = np.nan
        # Adjust the temperature data tables to keep only depths that are above
        # the deepest thermistor.
        T_meas_interp = T_meas_interp[
            T_meas_interp.columns[T_meas_interp.columns <= f_interp.x_max]
        ]
        T_model = T_model[
            T_model.columns[T_model.columns <= f_interp.x_max]
        ]
        return T_model, T_meas_interp


def calculate_scores_temperature_all_lakes(run, depths_interp='loggers'):
    """
    Calculate correlation coefficient and RMSE of the predicted temperature.

    Parameters
    ----------
    run : int or str
        Run ID.
    depths_interp : {'loggers', 'model'}, default: 'loggers'
        loggers: Interpolate model results on the loggers' depths.
        model: Interpolate loggers' measurements on the model output's depths.
    """

    path_tchains = '/home/jonathan/OneDrive/VM/Metlake/Data/ThermistorChains/'
    path_modelres = '/home/jonathan/OneDrive/VM/Metlake/LAKE_results/essential'

    chains_lakes = {
        'BD3': 2, 'BD4': 3, 'BD6': 1, 'PAR': 3, 'VEN': 2, 'SOD': 3,
        'SGA': 1, 'GUN': 3, 'GRI': 1, 'LJE': 1, 'LJR': 1, 'NAS': 1, 'NBJ': 2,
        'DAM': 2, 'NOR': 3, 'GRA': 2, 'KLI': 1, 'GYS': 3, 'LAM': 1
    }

    columns = [
        'Measurements_Mean', 'Model_Mean', 'Measurements_StandardDeviation',
        'Model_StandardDeviation', 'CorrelationCoefficient', 'RMSE',
        'RMSE_centered'
    ]
    whole_lake_stats = pd.DataFrame(
        index=sorted(chains_lakes.keys()), columns=columns, dtype='float64'
    )
    whole_lake_stats.index.name = 'Lake'
    depths_stats = {}
    T_model_all = {}
    T_meas_all = {}

    for lake, chain in chains_lakes.items():
        print(lake)
        # Path to files
        file_T_meas = os.path.join(
            path_tchains, f'Thermistors_{lake}_T{chain}_raw.mat'
        )
        if isinstance(run, str):
            folder_run = f'{lake}_run{run}'
        elif isinstance(run, int):
            folder_run = f'{lake}_run{str(run).zfill(3)}'
        if folder_run not in os.listdir(path_modelres):
            continue
        file_T_model = os.path.join(
            path_modelres, folder_run, 'water_temp  1  1.dat'
        )
        # Import data tables
        T_model, T_meas = import_and_homogenize_model_and_measured_T(
            file_T_model, file_T_meas, depths_interp
        )
        # Calculate statistics by depth and insert in dedicated table
        d_stats = pd.DataFrame(
            index=T_model.columns, columns=columns, dtype='float64'
        )
        d_stats.index.name = 'Depth_[m]'
        d_stats['Measurements_Mean'] = T_meas.mean()
        d_stats['Model_Mean'] = T_model.mean()
        d_stats['Measurements_StandardDeviation'] = T_meas.std()
        d_stats['Model_StandardDeviation'] = T_model.std()
        d_stats['CorrelationCoefficient'] = np.mean(
            (T_meas - T_meas.mean())*(T_model - T_model.mean())
        )/(T_meas.std()*T_model.std())
        d_stats['RMSE'] = \
                np.sqrt(np.square(T_model - T_meas).mean())
        d_stats['RMSE_centered'] = np.sqrt(
            np.square(
                (T_model - T_model.mean()) - (T_meas - T_meas.mean())
            ).mean()
        )
        depths_stats[lake] = d_stats
        # Calculate whole-lake statistics and insert in dedicated table
        T_model_f = T_model.values.flatten()
        T_meas_f = T_meas.values.flatten()
        T_model_f_centered = T_model_f - np.nanmean(T_model_f)
        T_meas_f_centered = T_meas_f - np.nanmean(T_meas_f)
        whole_lake_stats.loc[lake, 'Measurements_Mean'] = np.nanmean(T_meas_f)
        whole_lake_stats.loc[lake, 'Model_Mean'] = np.nanmean(T_model_f)
        whole_lake_stats.loc[lake, 'Measurements_StandardDeviation'] = \
                np.nanstd(T_meas_f)
        whole_lake_stats.loc[lake, 'Model_StandardDeviation'] = \
                np.nanstd(T_model_f)
        whole_lake_stats.loc[lake, 'CorrelationCoefficient'] = np.nanmean(
            T_meas_f_centered*T_model_f_centered
        )/(np.nanstd(T_meas_f)*np.nanstd(T_model_f))
        whole_lake_stats.loc[lake, 'RMSE'] = np.sqrt(
            np.nanmean(np.square(T_model_f - T_meas_f))
        )
        whole_lake_stats.loc[lake, 'RMSE_centered'] = np.sqrt(
            np.nanmean(np.square(T_model_f_centered - T_meas_f_centered))
        )
        # Save predicted and measured temperature tables
        T_model_all[lake] = T_model
        T_meas_all[lake] = T_meas

    return depths_stats, whole_lake_stats, T_model_all, T_meas_all


def calculate_stats_calibration_runs(
    meas, lake, params_file, main_param, mode
):
    """
    Extract best run(s) among a set of calibration runs.

    Parameters
    ----------
    meas : pd.DataFrame
        Data table containing measurements to compare with the model results.
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake of interest.
    params_file : pd.DataFrame
        Data table containing the list of values used for each calibrated
        parameter.
    main_param : {'CH4_[uM]', 'diff_flux', 'ebul_flux'}
        Indicate if concentration or flux data should be used to evaluate
        the runs.
        If 'CH4_[uM]' is used, the data table 'meas' should be the table
        'avg_conc_groups_uM' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
        If 'diff_flux' or 'ebul_flux' are used, the data table 'meas' should be
        the table 'avg_fluxes_groups' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    mode : {'average', 'individual'}
        Indicate if seasonal average data or individual measurement should
        be used to evaluate the runs.
        WARNING: 'individual' only works for main_param='CH4_[uM]'.
    """

    meas = meas[meas['Lake'] == lake].copy()
    params_file = params_file.copy()

    if main_param == 'CH4_[uM]':
        mod_param = 'CH4aq,surf_[mol*m-3]'
    elif main_param == 'diff_flux':
        mod_param = 'FCH4turb,surf_[mol*m-2*s-1]'
    elif main_param == 'ebul_flux':
        mod_param = 'FCH4ebul,surf_[mol*m-2*s-1]'

    for n in params_file.index:
        mod, mod_units = import_methane_time_series_file(
            os.path.join(
                '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake',
                'LAKE_results', 'essential', f'{lake}_run{str(n).zfill(3)}',
                'methane_series  1  1.dat'
            )
        )
        # Skip the first five days because the model takes some time
        # to equilibrate
        dt_ini = mod.index[0] + timedelta(days=5)
        dt_fin = mod.index[-1]
        if mode == 'average':
            mod = mod[mod.index >= dt_ini]
            if main_param == 'CH4_[uM]':
                meas = meas[np.logical_and(
                    meas['sampling_middle'] >= dt_ini,
                    meas['sampling_middle'] <= dt_fin
                )]
                mod_avg = mod[mod_param].mean()*1e3
            elif main_param == 'diff_flux' or main_param == 'ebul_flux':
                meas = meas[np.logical_and(
                    meas['deployment_start'] >= dt_ini,
                    meas['deployment_end'] <= dt_fin
                )]
                mod_avg = mod[mod_param].mean()*1e3*86400
            meas_avg = meas[main_param].mean()
            params_file.loc[n, 'relative error'] = (mod_avg - meas_avg)/meas_avg
        elif mode == 'individual':
            for row_n, row in meas.iterrows():
                sampling_mid = row['sampling_middle']
                if sampling_mid >= dt_ini and sampling_mid <= dt_fin:
                    time_cond = np.logical_and(
                        mod.index >= row['sampling_start'],
                        mod.index <= row['sampling_end']
                    )
                    if sum(time_cond) == 0:
                        mod_value = mod.iloc[
                            np.argmin(
                                np.abs(mod.index - row['sampling_middle'])
                            ), mod.columns.get_loc(mod_param)
                        ]
                    else:
                        mod_value = mod.loc[time_cond, mod_param].mean()
                    meas.loc[row_n, 'CH4_model'] = mod_value*1e3
                else:
                    meas.loc[row_n, 'CH4_model'] = np.nan
            d = meas[['CH4_[uM]', 'CH4_model']].dropna()
            lm = LinearRegression().fit(
                d['CH4_[uM]'].values.reshape(-1, 1),
                d['CH4_model'].values.reshape(-1, 1)
            )
            r2 = lm.score(
                d['CH4_[uM]'].values.reshape(-1, 1),
                d['CH4_model'].values.reshape(-1, 1)
            )
            params_file.loc[n, 'intercept'] = lm.intercept_
            params_file.loc[n, 'slope'] = lm.coef_[0]
            params_file.loc[n, 'r2'] = r2

    return params_file


def extract_measured_and_modelled_CH4aq(meas, lake, run):
    """
    Extract modelled CH4 concentration at the times of measurement.

    Parameters
    ----------
    meas : pd.DataFrame
        Data table containing measurements to compare with the model results.
        It should be the table 'avg_conc_groups_uM' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    lake : {'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN',
            'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM',
            'KLI'}
        Name of the lake of interest.
    run : int or str
        Run ID.
    """

    meas = meas[meas['Lake'] == lake].copy()
    mod, mod_units = import_methane_time_series_file(
        os.path.join(
            '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake',
            'LAKE_results', 'essential', f'{lake}_run{str(run).zfill(3)}',
            'methane_series  1  1.dat'
        )
    )

    # Skip the first five days because the model takes some time
    # to equilibrate
    dt_ini = mod.index[0] + timedelta(days=5)
    dt_fin = mod.index[-1]

    for row_n, row in meas.iterrows():
        sampling_mid = row['sampling_middle']
        if sampling_mid >= dt_ini and sampling_mid <= dt_fin:
            time_cond = np.logical_and(
                mod.index >= row['sampling_start'],
                mod.index <= row['sampling_end']
            )
            if sum(time_cond) == 0:
                row_ind = np.argmin(np.abs(mod.index - row['sampling_middle']))
                mod_time = mod.index[row_ind]
                mod_value = mod.iloc[
                    row_ind, mod.columns.get_loc('CH4aq,surf_[mol*m-3]')
                ]
            else:
                mod_time = mod.index[time_cond].mean()
                mod_value = mod.loc[time_cond, 'CH4aq,surf_[mol*m-3]'].mean()
            meas.loc[row_n, 'time_model'] = mod_time
            meas.loc[row_n, 'CH4_model'] = mod_value*1e3
        else:
            meas.loc[row_n, 'time_model'] = np.nan
            meas.loc[row_n, 'CH4_model'] = np.nan

    meas = meas.astype({'time_model': meas.dtypes['sampling_middle']})

    meas = meas[[
        'Lake', 'sampling_middle', 'CH4_[uM]', 'time_model', 'CH4_model'
    ]].dropna()

    return meas


def calculate_RMSE_measured_vs_modelled_CH4aq(meas, run):
    """
    Calculate the RMSE of CH4aq predicted by the model for all lakes.

    Parameters
    ----------
    meas : pd.DataFrame
        Data table containing measurements to compare with the model results.
        It should be the table 'avg_conc_groups_uM' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    run : int or str
        Run ID.
    """

    lakes = [
        'BD3', 'BD4', 'VEN', 'PAR', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'LAM', 'GYS', 'KLI'
    ]

    result = pd.DataFrame(
        index=lakes, columns=['RMSE', 'CH4_avg'], dtype='float64'
    )

    for lake in lakes:
        df = extract_measured_and_modelled_CH4aq(meas, lake, run)
        result.loc[lake, 'RMSE'] = np.sqrt(np.mean(np.square(
            df['CH4_[uM]'] - df['CH4_model']
        )))
        result.loc[lake, 'CH4_avg'] = df['CH4_[uM]'].mean()

    return result


def plot_model_vs_measured_temperature(file_T_model, file_T_measured):
    """
    Plot time series of predicted and measured water temperature.

    Parameters
    ----------
    file_T_model : str
        Full or relative path to a file containing modeled temperature.
    file_T_measured : str
        Full or relative path to a thermistor chain file.
    """

    # Load data
    T_model = import_time_vs_depth_file(file_T_model)
    T_measured, time_measured, depths_measured, info_measured = \
            metlake_data.import_thermistor_chain_data(file_T_measured)
    # Convert doy to regular datetime and to UTC
    time_measured_dt = np.array([
        datetime(T_model.index[0].year, 1, 1) + timedelta(hours=24*(d-1)-2)
        for d in time_measured.squeeze()
    ])
    # Extract model values that are closest to measurements
    depths_model = [
        T_model.columns[np.argmin(np.abs(T_model.columns - d))]
        for d in depths_measured
    ]
    depths_all = [d for vec in [depths_measured, depths_model] for d in vec]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(time_measured_dt, T_measured, '-')
    ax.set_prop_cycle(None)
    ax.plot(T_model[depths_model], '--')
    ax.legend(depths_all, title='Depth [m]')
    ax.set_ylabel('Water temperature [oC]', fontsize=24)
    ax.set_title(
        'continuous lines = measurements, dashed lines = model', fontsize=20
    )
    ax.tick_params(labelsize=20)

    return fig, ax


def plot_model_vs_measured_surface_GHG(
    file_GHG_model, water_conc_meas, lake, chamber, meas_type, gas='CH4',
    ax=None, markersize=10, add_legend=True
):
    """
    Plot time series of predicted and measured surface water CH4 or CO2
    concentration.

    Parameters
    ----------
    file_GHG_model : str
        Full or relative path to a file containing modeled CH4 or CO2
        concentration.
    water_conc_meas : pandas.DataFrame
        Table containing surface water GHG concentration returned by
        the functions 'metlake_data.create_water_concentration_table' or
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    lake : str
        ID of the lake of interest in table 'water_conc_meas'.
    chamber : int
        ID of the chamber of interest in table 'water_conc_meas'.
    meas_type : {'individual_chamber', 'average_chambers'}
        If 'individual_chamber' is used, plot data from one specific chamber
        (chamber ID given by 'chamber'). It requires 'water_conc_meas' to come
        from the function 'metlake_data.create_water_concentration_table'.
        If 'average_chambers' is used, plot average data of all chambers.
        It requires 'water_conc_meas' to come from the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    gas : {'CH4', 'CO2'}, default: 'CH4'
        Gas of interest.
    ax : matplotlib Axes object or None, default: None
        Axe on which to plot the data. If None, create a new figure.
    markersize : int, default: 10
        Marker size for the measurements.
    add_legend : bool, default: True
        Wether a legend should be added to the figure or not.
    """

    if 'methane_series' in file_GHG_model:
        GHG_model, labels = import_methane_time_series_file(file_GHG_model)
        GHG_model = GHG_model['CH4aq,surf_[mol*m-3]']*1e3
    elif 'methane_water_soil' in file_GHG_model:
        GHG_model = import_time_vs_depth_file(file_GHG_model)
        GHG_model = GHG_model[0.0]*1e-3
    if meas_type == 'individual_chamber':
        GHG_meas = water_conc_meas.loc[
            np.logical_and(
                water_conc_meas['Lake'] == lake,
                water_conc_meas['Chamber'] == chamber
            ), ['Datetime', f'{gas}_[uM]']
        ].set_index('Datetime')
    elif meas_type == 'average_chambers':
        GHG_meas = water_conc_meas.loc[
            water_conc_meas['Lake'] == lake, ['sampling_middle', f'{gas}_[uM]']
        ].set_index('sampling_middle')

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    ax.plot(GHG_model, c='k', label='LAKE model')
    ax.plot(GHG_meas, '.', c='r', ms=markersize, label='Observations')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Month-Day', fontsize=20)
    if gas == 'CH4':
        ax.set_ylabel(r'Surface water CH$_4$ conc. [$\mu$M]', fontsize=20)
    elif gas == 'CO2':
        ax.set_ylabel(r'Surface water CO$_2$ conc. [$\mu$M]', fontsize=20)
    ax.tick_params(labelsize=20)
    if add_legend:
        ax.legend(fontsize=14)
    ax.grid()

    return fig, ax


def plot_model_vs_measured_surface_GHG_all_lakes(
    run, chambers=None, meas_type='average_chambers', gas='CH4', savefigs=False,
    folder_save='figures', fmt_fig='tif', aggregate_figures=True
):
    """
    Plot time series of modeled vs measured surface water CH4 or CO2
    concentration in all lakes.

    Parameters
    ----------
    run : int or str
        Run ID.
    chambers : pandas.DataFrame or None, default: None
        Table returned by the function 'metlake_data.import_chambers_data'.
        If None, the table is imported by the function.
    meas_type : {'individual_chamber', 'average_chambers'},
                default: 'average_chambers'
        If 'individual_chamber' is used, plot data from one specific chamber
        (chamber ID given by 'chamber'). It requires 'water_conc_meas' to come
        from the function 'metlake_data.create_water_concentration_table'.
        If 'average_chambers' is used, plot average data of all chambers.
        It requires 'water_conc_meas' to come from the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    gas : {'CH4', 'CO2'}, default: 'CH4'
        Gas of interest.
    savefigs : bool, default: False
        If True, save the figures.
    folder_save : str, default: 'figures'
        Folder in /home/jonathan/OneDrive/VM/Metlake/ProcessBasedModel_manuscript
        where to save the figures.
    fmt_fig : str, default: 'tif'
        Format to use when saving figures.
    aggregate_figures : bool, default: True
        If True, plot data for all lakes in a few figures separated in subplots.
        If False, plot data for each lake in a separate figure.
    """

    path_modelres = '/home/jonathan/OneDrive/VM/Metlake/LAKE_results/essential'
    path_save = os.path.join(
        '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake',
        'ProcessBasedModel_manuscript', folder_save
    )

    if chambers is None:
        chambers = metlake_data.import_chambers_data()
    water_conc = metlake_data.create_water_concentration_table(chambers)
    water_conc['Lake'].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    if meas_type == 'average_chambers':
        info, area, volume = metlake_data.import_info_lakes()
        _, _, water_conc, _, _, _ = \
            metlake_data.calculate_chamber_data_average_per_deployment_and_year(
                chambers, water_conc, area
            )

    chamber_centre = {
        'BD3': 8, 'BD4': 8, 'BD6': 12, 'PAR': 12, 'VEN': 8, 'SOD': 12,
        'SGA': 4, 'GUN': 12, 'GRI': 4, 'LJE': 8, 'LJR': 4, 'NAS': 4, 'NBJ': 8,
        'DAM': 8, 'NOR': 12, 'GRA': 8, 'KLI': 4, 'GYS': 12, 'LAM': 12
    }

    if aggregate_figures:
        fig, ax_all = plt.subplots(6, 3, figsize=(8, 12))
        subplot_loc = {
            'BD3': (0, 0), 'BD4': (0, 1), 'BD6': (0, 2),
            'NAS': (1, 0), 'NBJ': (1, 1), 'LJR': (1, 2), 'LJE': None,
            'NOR': (2, 0), 'GRA': (2, 1), 'DAM': (2, 2),
            'VEN': (3, 0), 'SOD': (3, 1), 'PAR': (3, 2),
            'GUN': (4, 0), 'GRI': (4, 1), 'SGA': (4, 2),
            'LAM': (5, 0), 'KLI': (5, 1), 'GYS': (5, 2)
        }

    n = 0
    for lake, chamber in chamber_centre.items():
        if aggregate_figures and subplot_loc[lake] is not None:
            x, y = subplot_loc[lake]
        if isinstance(run, str):
            folder_run = f'{lake}_run{run}'
        elif isinstance(run, int):
            folder_run = f'{lake}_run{str(run).zfill(3)}'
        if folder_run not in os.listdir(path_modelres):
            if aggregate_figures and subplot_loc[lake] is not None:
                fig.delaxes(ax_all[x, y])
            continue
        file_GHG_model = os.path.join(
            path_modelres, folder_run, 'methane_water_soil  1  1.dat'
        )
        if aggregate_figures:
            ax = ax_all[x, y]
            markersize = 6
            add_legend = False
        else:
            ax = None
            markersize = 10
            add_legend = True
        fig, ax = plot_model_vs_measured_surface_GHG(
            file_GHG_model, water_conc, lake, chamber, meas_type, gas=gas,
            ax=ax, markersize=markersize, add_legend=add_legend
        )
        if aggregate_figures:
            # Select the year
            if lake in ['BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD']:
                year = 2018
            elif lake in ['SGA', 'GUN', 'GRI', 'LJE', 'LJR', 'NAS', 'NBJ']:
                year = 2019
            elif lake in ['DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM']:
                year = 2020
            # Adjust axes limits
            ax.set_xlim(datetime(year, 3, 1), datetime(year + 1, 1, 1))
            ax.set_ylim(0, 4)
            # Keep axes labels only on selected subplots
            if x == 5 and y == 1:
                ax.set_xlabel('Month', fontsize=16)
            else:
                ax.set_xlabel('')
            ax.set_ylabel('')
            if x == 2 and y == 0:
                ax.text(
                    -0.3, -1.4, r'Surface water CH$_4$ conc. [$\mu$M]',
                    fontsize=16, rotation='vertical', transform=ax.transAxes
                )
            # Set position of ticks on the axes
            ax.set_xticks([datetime(year, n, 1) for n in [3, 6, 9, 12]])
            ax.set_yticks(range(5))
            # Keep tick labels only on selected subplots and change fontsize
            ax.tick_params(labelsize=14)
            if x == 5:
                ax.xaxis.set_ticklabels(['Mar', 'Jun', 'Sep', 'Dec'])
            else:
                ax.xaxis.set_ticklabels([])
            if y == 0:
                ax.yaxis.set_ticklabels([0, '', 2, '', 4])
            else:
                ax.yaxis.set_ticklabels([])
            # Add lake name as text on each subplot
            if lake == 'PAR':
                lake = 'PRS'
            ax.text(
                0.15, 0.8, lake, fontsize=14, fontweight='bold',
                transform=ax.transAxes
            )
            # Add legend
            if x == 0 and y == 1:
                ax.legend(
                    loc='center', bbox_to_anchor=(0.5, 1.2),
                    bbox_transform=ax.transAxes,
                    ncol=2, fontsize=12, frameon=False
                )
        else:
            if lake == 'PAR':
                lake = 'PRS'
            ax.set_title(lake, fontsize=24)
        if savefigs and not aggregate_figures:
            fig.savefig(
                os.path.join(path_save, f'{lake}_CH4_surface.{fmt_fig}')
            )
        n += 1
    if savefigs and aggregate_figures:
        fig.savefig(
            os.path.join(path_save, f'All_lakes_CH4_surface.{fmt_fig}')
        )

    return fig, ax_all


def plot_model_vs_measured_GHG_flux(
    file_GHG_model, flux_meas, lake, chamber, model_daily_avg=True
):
    """
    Plot time series of predicted and measured surface water CH4 or CO2
    concentration.

    Parameters
    ----------
    file_GHG_model : str
        Full or relative path to a file containing modeled CH4 or CO2
        concentration.
    flux_meas : pandas.DataFrame
        Table containing GHG flux measurements.
        If 'chamber' is a digit, this table must have the same format as
        the table returned by the function 'metlake_data.import_chambers_data'
        ('ch4_flux' key).
        If 'chamber' is 'average', this table must have the same format as
        the table returned by the function
        'metlake_data.calculate_average_flux_per_deployment_group'.
    lake : str
        ID of the lake of interest in table 'flux_meas'.
    chamber : int or str
        'average': whole-lake flux average will be used.
        any other value: ID of the chamber of interest in table 'flux_meas'.
    model_daily_avg : bool, default: True
        True: Use daily averaged model values.
        False: Use raw model values.
    """

    col_lake = ('General', 'Lake', 'Unnamed: 0_level_2')
    col_chamber = ('General', 'Chamber ID', 'Unnamed: 1_level_2')
    cols_keep = [
        ('General', 'Initial sampling', 'Date and time'),
        ('Flux calculation', 'CH4 diffusive flux', 'mmol m-2 d-1')
    ]
    GHG_model, labels = import_methane_time_series_file(file_GHG_model)
    GHG_model = GHG_model['FCH4turb,surf_[mol*m-2*s-1]']*86400*1e3
    if model_daily_avg:
        GHG_model = GHG_model.resample('D').mean()
    if chamber == 'average':
        GHG_meas = flux_meas.loc[
            flux_meas['lake'] == lake, ['deployment_start', 'diff_flux']
        ].set_index('deployment_start')
    else:
        GHG_meas = flux_meas.loc[
            np.logical_and(
                flux_meas[col_lake] == lake, flux_meas[col_chamber] == chamber
            ), cols_keep
        ].set_index(cols_keep[0])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(GHG_model, c='k', label='LAKE model')
    ax.plot(GHG_meas, '.', c='r', ms=10, label='measured')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Month-Day', fontsize=20)
    ax.set_ylabel(
        ('Turbulent diffusive CH$_4$ flux to atmosphere\n'
         '[mmol m$^{-2}$ d$^{-1}$]'),
        fontsize=20
    )
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=14)
    ax.grid()

    return fig, ax


def plot_model_vs_measured_GHG_flux_all_lakes(
    run, chambers=None, savefigs=False, folder_save='figures', fmt_fig='tif',
    meas_spatial_average=True, model_daily_avg=True
):
    """
    Plot time series of modeled vs measured CH4 fluxes in all lakes.

    Parameters
    ----------
    run : int or str
        Run ID.
    chambers : pandas.DataFrame or None, default: None
        Table returned by the function 'metlake_data.import_chambers_data'.
        If None, the table is imported by the function.
    savefigs : bool, default: False
        If True, save the figures.
    folder_save : str, default: 'figures'
        Folder in /home/jonathan/OneDrive/VM/Metlake/ProcessBasedModel_manuscript
        where to save the figures.
    fmt_fig : str, default: 'tif'
        Format to use when saving figures.
    meas_spatial_average : bool, default: True
        True: Use whole-lake flux average.
        False: Use flux measured at the chamber located the closest to
            the center of the lake.
    model_daily_avg : bool, default: True
        True: Use daily averaged model values.
        False: Use raw model values.
    """

    path_modelres = '/home/jonathan/OneDrive/VM/Metlake/LAKE_results/essential'
    path_save = os.path.join(
        '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake',
        'ProcessBasedModel_manuscript', folder_save
    )

    if chambers is None:
        chambers = metlake_data.import_chambers_data()
    fluxes = chambers['ch4_flux']
    fluxes[('General', 'Lake', 'Unnamed: 0_level_2')].replace(
        {'BD03': 'BD3', 'BD04': 'BD4', 'BD06': 'BD6'}, inplace=True
    )
    if meas_spatial_average:
        info, area, volume = metlake_data.import_info_lakes()
        metlake_data.create_deployment_groups_indices(
            fluxes,
            ('General', 'Lake', 'Unnamed: 0_level_2'),
            ('General', 'Initial sampling', 'Date and time'),
            ('General', 'Final sampling', 'Date and time'),
            ('General', 'Deployment group', '')
        )
        area.columns = area.columns.droplevel(1)
        fluxes = metlake_data.calculate_average_flux_per_deployment_group(
            fluxes, area
        )

    chamber_centre = {
        'BD3': 8, 'BD4': 8, 'BD6': 12, 'PAR': 12, 'VEN': 8, 'SOD': 12,
        'SGA': 4, 'GUN': 12, 'GRI': 4, 'LJE': 8, 'LJR': 4, 'NAS': 4, 'NBJ': 8,
        'DAM': 8, 'NOR': 12, 'GRA': 8, 'KLI': 4, 'GYS': 12, 'LAM': 12
    }

    for lake, chamber in chamber_centre.items():
        if meas_spatial_average:
            chamber = 'average'
        if isinstance(run, str):
            folder_run = f'{lake}_run{run}'
        elif isinstance(run, int):
            folder_run = f'{lake}_run{str(run).zfill(3)}'
        if folder_run not in os.listdir(path_modelres):
            continue
        file_GHG_model = os.path.join(
            path_modelres, folder_run, 'methane_series  1  1.dat'
        )
        fig, ax = plot_model_vs_measured_GHG_flux(
            file_GHG_model, fluxes, lake, chamber, model_daily_avg
        )
        ax.set_title(lake, fontsize=24)
        if savefigs:
            if model_daily_avg:
                file_save = f'{lake}_CH4_flux_daily_avg.{fmt_fig}'
            else:
                file_save = f'{lake}_CH4_flux.{fmt_fig}'
            fig.savefig(os.path.join(path_save, file_save))


def plot_contourf_model_and_measured_temperature(T_model, T_meas):
    """
    Draw contour plots of predicted and measured water temperature.

    Parameters
    ----------
    T_model : pd.DataFrame
        Data table containing modeled temperature.
    T_meas : pd.DataFrame
        Data table containing measured temperature.
    """

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    contour1 = ax1.contourf(
        T_model.index, T_model.columns, T_model.T, np.arange(0, 30, 2)
    )
    ax1.invert_yaxis()
    ax1.set_title('Model', fontsize=20)
    plt.colorbar(contour1, ax=ax1)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    contour2 = ax2.contourf(
        T_meas.index, T_meas.columns, T_meas.T, np.arange(0, 30, 2)
    )
    ax2.invert_yaxis()
    ax2.set_title('Measured', fontsize=20)
    plt.colorbar(contour2, ax=ax2)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    T_diff = T_model - T_meas
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    contour3 = ax3.contourf(T_diff.index, T_diff.columns, T_diff.T)
    ax3.invert_yaxis()
    ax3.set_title('Model - Measured', fontsize=20)
    plt.colorbar(contour3, ax=ax3)

    return fig1, ax1, fig2, ax2, fig3, ax3


def plot_contourf_model_and_measured_temperature_all_lakes(
    run, savefigs=False, folder_save='figures', fmt_fig='tif'
):
    """
    Draw contour plots of predicted and measured water temperature.

    Parameters
    ----------
    run : int or str
        Run ID.
    savefigs : bool, default: False
        If True, save the figures.
    folder_save : str, default: 'figures'
        Folder in /home/jonathan/OneDrive/VM/Metlake/ProcessBasedModel_manuscript
        where to save the figures.
    fmt_fig : str, default: 'tif'
        Format to use when saving figures.
    """

    path_tchains = '/home/jonathan/OneDrive/VM/Metlake/Data/ThermistorChains/'
    path_modelres = '/home/jonathan/OneDrive/VM/Metlake/LAKE_results/essential'
    path_save = os.path.join(
        '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake',
        'ProcessBasedModel_manuscript', folder_save
    )

    lakes = [
        'BD3', 'BD4', 'BD6', 'PAR', 'VEN', 'SOD', 'SGA', 'GUN', 'GRI',
        'LJE', 'LJR', 'NAS', 'NBJ', 'DAM', 'NOR', 'GRA', 'KLI', 'GYS', 'LAM'
    ]
    chains_lakes = {
        'BD3': 2, 'BD4': 3, 'BD6': 1, 'PAR': 3, 'VEN': 2, 'SOD': 3,
        'SGA': 1, 'GUN': 3, 'GRI': 1, 'LJE': 1, 'LJR': 1, 'NAS': 1, 'NBJ': 2,
        'DAM': 2, 'NOR': 3, 'GRA': 2, 'KLI': 1, 'GYS': 3, 'LAM': 1
    }

    for lake, chain in chains_lakes.items():
        file_T_meas = os.path.join(
            path_tchains, f'Thermistors_{lake}_T{chain}_raw.mat'
        )
        if isinstance(run, str):
            folder_run = f'{lake}_run{run}'
        elif isinstance(run, int):
            folder_run = f'{lake}_run{str(run).zfill(3)}'
        if folder_run not in os.listdir(path_modelres):
            continue
        file_T_model = os.path.join(
            path_modelres, folder_run, 'water_temp  1  1.dat'
        )
        T_model, T_meas = import_and_homogenize_model_and_measured_T(
            file_T_model, file_T_meas, 'loggers'
        )
        fig1, ax1, fig2, ax2, fig3, ax3 = \
                plot_contourf_model_and_measured_temperature(T_model, T_meas)
        ax1.set_title(f'{lake} (Model)', fontsize=20)
        ax1.set_xlabel('Month-Day', fontsize=14)
        ax1.set_ylabel('Depth [m]', fontsize=14)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.tick_params(labelsize=12)
        ax2.set_title(f'{lake} (Measured)', fontsize=20)
        ax2.set_xlabel('Month-Day', fontsize=14)
        ax2.set_ylabel('Depth [m]', fontsize=14)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.tick_params(labelsize=12)
        ax3.set_title(f'{lake} (Model - Measured)', fontsize=20)
        ax3.set_xlabel('Month-Day', fontsize=14)
        ax3.set_ylabel('Depth [m]', fontsize=14)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.tick_params(labelsize=12)
        if savefigs:
            fig1.savefig(os.path.join(path_save, f'{lake}_model.{fmt_fig}'))
            fig2.savefig(os.path.join(
                path_save,f'{lake}_measurements.{fmt_fig}')
            )
            fig3.savefig(os.path.join(
                path_save, f'{lake}_difference.{fmt_fig}')
            )


def plot_contourf_model_results(file_model_results, func=None, cont_range=10):
    """
    Draw a contour plot from a 'time_series' file.

    Parameters
    ----------
    file_model_results : str
        Full or relative path to a 'time_series' model output file.
    func : function or None, default: None
        Function to apply to the data before plotting.
    cont_range : int or array-like, default: 10
        Range of values or number of values to use for contours.
    """

    data = import_time_vs_depth_file(file_model_results)
    if func is not None:
        data = func(data)
    X, Y = np.meshgrid(data.index, data.columns)
    fig, ax = plt.subplots(figsize=(12, 10))
    contour = ax.contourf(X, Y, data.T, cont_range)
    ax.invert_yaxis()
    ax.tick_params(labelsize=20)
    plt.colorbar(contour, ax=ax)

    return fig, ax


def plot_relative_error_all_calibration_runs(
    lakes, meas, param, freq_mode, param_mode
):
    """
    Plot data obtained with the function 'calculate_stats_calibration_runs'.

    Parameters
    ----------
    lakes : list
        List of lakes for which files will be compared.
        Possibilities are:
        'PAR', 'VEN', 'SOD', 'BD3', 'BD4', 'BD6', 'SGA', 'GRI', 'GUN', 'LJE',
        'LJR', 'NAS', 'NBJ', 'DAM', 'GRA', 'NOR', 'GYS', 'LAM', 'KLI'.
    meas : pd.DataFrame
        Data table containing measurements to compare with the model results.
    param : {'CH4_[uM]', 'diff_flux', 'ebul_flux'}
        Indicate if concentration or flux data should be used to evaluate
        the runs.
        If 'CH4_[uM]' is used, the data table 'meas' should be the table
        'avg_conc_groups_uM' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
        If 'diff_flux' or 'ebul_flux' are used, the data table 'meas' should be
        the table 'avg_fluxes_groups' returned by the function
        'metlake_data.calculate_chamber_data_average_per_deployment_and_year'.
    freq_mode : {'average', 'individual'}
        Indicate if seasonal average data or individual measurement should
        be used to evaluate the runs.
    param_mode : {'values', 'ratio'}
        Indicate if the values of the parameters or their ratio should be used
        to sort the runs.
    """

    #fig, ax = plt.subplots()

    path_params_lists = os.path.join(
        '/', 'home', 'jonathan', 'OneDrive', 'VM', 'Metlake', 'LAKE_results'
    )

    for n, lake in enumerate(lakes):
        fig, ax = plt.subplots()
        if n <= 10:
            marker = '.'
        else:
            marker = '*'
        for f in os.listdir(path_params_lists):
            if len(f.split('_')) > 2 and f.split('_')[2] == lake:
                break
        if param_mode == 'values':
            params_list = pd.read_csv(
                os.path.join(path_params_lists, f), index_col=0
            ).sort_values(['r0methprod', 'VmaxCH4aeroboxid', 'khsCH4'])
        elif param_mode == 'ratio':
            params_list = pd.read_csv(
                os.path.join(path_params_lists, f), index_col=0
            )
            params_list['ratio'] = np.log10(
                params_list['r0methprod']*params_list['khsCH4']/ \
                params_list['VmaxCH4aeroboxid']
            )
            params_list.sort_values('ratio', inplace=True)
        calib_res = calculate_stats_calibration_runs(
            meas, lake, params_list, param, freq_mode
        )
        ax.plot(
            calib_res['relative error'].values, linestyle='-',
            marker=marker, label=lake
        )
        ax.grid()
        ax.legend()
        ax.hlines([-1, 1], 0, 125, color='k')
        ax.set_title(lake)

    return fig, ax


def plot_CH4aq_vs_RMSE_CH4aq(data):
    """
    Plot CH4aq concentration vs. RMSE of the modelled CH4aq.

    Parameters
    ----------
    data : pandas.DataFrame
        Data table returned by the function
        'calculate_RMSE_measured_vs_modelled_CH4aq' for run 120.
    """

    X1 = data['CH4_avg'].values.reshape(-1, 1)
    y1 = data['RMSE'].values.reshape(-1, 1)
    lm1 = LinearRegression().fit(X1, y1)
    intercept1, slope1 = lm1.intercept_[0], lm1.coef_[0][0]
    data_noGUN = data[data.index != 'GUN']
    X2 = data_noGUN['CH4_avg'].values.reshape(-1, 1)
    y2 = data_noGUN['RMSE'].values.reshape(-1, 1)
    lm2 = LinearRegression().fit(X2, y2)
    intercept2, slope2 = lm2.intercept_[0], lm2.coef_[0][0]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data['CH4_avg'], data['RMSE'], c='k', s=70)
    ax.axline((0.5, 0.5), slope=1, linestyle=':', c='k', label='1:1')
    ax.axline(
        (0., intercept1), slope=slope1, linestyle='--', c='k',
        label='lin. regr. all lakes'
    )
    ax.axline(
        (0., intercept2), slope=slope2, linestyle='-', c='k',
        label='lin. regr. without GUN'
    )

    ax.set_xlim(0.1, 1.2)
    ax.set_ylim(0.0, 1.1)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Average surface water CH$_4$ conc. [µM]', fontsize=20)
    ax.set_ylabel('RMSE modelled surface water CH$_4$ conc. [µM]', fontsize=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.text(0.13, 0.81, 'GUN', fontsize=14)

    return fig, ax


def plot_mean_depth_vs_RMSE_CH4aq(data):
    """
    Plot mean lake depth vs. RMSE of the modelled CH4aq.

    Parameters
    ----------
    data : pandas.DataFrame
        Data table returned by the function
        'calculate_RMSE_measured_vs_modelled_CH4aq' for run 120.
    """

    info, _, _ = metlake_data.import_info_lakes()
    info = info[['Lake', 'LakeDepth_Mean_[m]']].set_index('Lake')
    data = data.join(info)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data['LakeDepth_Mean_[m]'], data['RMSE'], c='k', s=70)
    ax.vlines(3., -0.1, 1.1, linestyle=':', color='k')
    ax.set_xlim(0., 11.)
    ax.set_ylim(0., 1.1)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Mean lake depth [m]', fontsize=20)
    ax.set_ylabel('RMSE modelled surface water CH$_4$ conc. [µM]', fontsize=20)
    ax.text(1.8, 0.94, 'NOR', fontsize=14)

    return fig, ax
