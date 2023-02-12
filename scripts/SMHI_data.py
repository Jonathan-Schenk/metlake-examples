import os
from datetime import datetime, timedelta
import pickle
import requests
import urllib
import urllib.request as urlreq
import numpy as np
import pandas as pd
import eccodes


def check_data_availability_at_station(station, period):
    """
    Print a list of available parameters for a given station.

    Parameters
    ----------
    station : int
        Code of the station.
         81130: Ljungskile (close to SGA) !!!OLD!!!
         82090: Asbracka-Torpabron V (close to SGA)
         82190: Trollhattan Flygplats (close to GRI, GUN) !!!OLD!!!
         82230: Vanersborg (close to GRI, GUN)
         82260: Satenas (close to GRI)
         82620: Trollhattan (close to GRI, GUN) !!!OLD!!!
         85240: Malmslatt (close to LiU)
         86340: Norrkoping ("close" to VEN, PAR, SOD)
        140480: Umea Flygplats ("close" to NAS, NBJ, LJE, LJR)
        148040: Fredrika A (close to NAS, NBJ, LJE, LJR)
        148050: Fredrika (close to NAS, NBJ, LJE, LJR) !!!OLD!!!
        188820: Katterjakk (close to BD3, BD4, BD6)
        188850: Katterjakk A (close to BD3, BD4, BD6)
    period : {'latest-hour', 'latest-day', 'latest-months', 'corrected-archive'}
        Period of interest.
    """

    for n in range(50):
        url = (
            f'https://opendata-download-metobs.smhi.se/api/version/1.0/'
            f'parameter/{n}/station/{station}/period/{period}/data.csv'
        )
        try:
            with urlreq.urlopen(url) as response:
                data = response.read().decode('utf-8').split('\n')
        except(urllib.error.HTTPError, IndexError):
            continue
        print(n, data[4])


def load_station_data(param, station, period):
    """
    Load data for a given parameter and station directly from the SMHI server.

    Parameters
    ----------
    param : int
        Code of the parameter of interest.
         1: temperature (instant value) [degree Celsius]
         2: temperature (average over 1 day) [degree Celsius]
         3: wind direction (mean value over 10 min) [degree]
         4: wind speed (mean value over 10 min) [m/s]
         5: precipitation amount (over 1 day) [mm]
         6: relative humidity (instant value) [percent]
         7: precipitation amount (over 1 hour) [mm]
         8: snow thickness (instant value) [m]
         9: pressure (at sea level, instant value) [hPa]
        10: sunshine time (sum over 1 hour) [s]
        11: global radiation (mean value over 1 hour) [W/m2]
        12: visibility (instant value) [m]
        13: present weather (instant value) [code]
        14: precipitation amount (over 15 min) [mm]
        15: precipiation intensity (max over 15 min) [mm/s]
        16: total cloud cover (instant value) [percent]
        18: precipition (at 18:00) [code]
        19: temperature (minimum over 1 day) [degree Celsius]
        20: temperature (maximum over 1 day) [degree Celsius]
        21: wind gust (maximum) [m/s]
        22: temperature (mean over 1 month) [degree Celsius]
        23: precipitation amount (sum over 1 month) [mm]
        24: longwave radiation (mean value over 1 hour) [W/m2]
        25: wind speed (max of 10 min mean over 3 hours) [m/s]
        26: temperature (minimum over 12 hours) [degree Celsius]
        27: temperature (maximum over 12 hours) [degree Celsius]
        28: altitude of cloud base (layer 1, instant value) [m]
        29: amount of cloud (layer 1, instant value) [code]
        30: altitude of cloud base (layer 2, instant value) [m]
        31: amount of cloud (layer 2, instant value) [code]
        32: altitude of cloud base (layer 3, instant value) [m]
        33: amount of cloud (layer 3, instant value) [code]
        34: altitude of cloud base (layer 4, instant value) [m]
        35: amount of cloud (layer 4, instant value) [code]
        36: altitude of cloud base (lowest, instant value) [m]
        38: precipiation intensity (max of mean over 15 min) [mm/s]
    station : int
        Code of the station.
         81130: Ljungskile (close to SGA) !!!OLD!!!
         82090: Asbracka-Torpabron V (close to SGA)
         82190: Trollhattan Flygplats (close to GRI, GUN) !!!OLD!!!
         82230: Vanersborg (close to GRI, GUN)
         82260: Satenas (close to GRI)
         82620: Trollhattan (close to GRI, GUN) !!!OLD!!!
         85240: Malmslatt (close to LiU)
         86340: Norrkoping ("close" to VEN, PAR, SOD)
        140480: Umea Flygplats ("close" to NAS, NBJ, LJE, LJR)
        148040: Fredrika A (close to NAS, NBJ, LJE, LJR)
        148050: Fredrika (close to NAS, NBJ, LJE, LJR) !!!OLD!!!
        188820: Katterjakk (close to BD3, BD4, BD6)
        188850: Katterjakk A (close to BD3, BD4, BD6)
    period : {'latest-hour', 'latest-day', 'latest-months', 'corrected-archive'}
        Period of interest.
    """

    url = (
        f'https://opendata-download-metobs.smhi.se/api/version/1.0/'
        f'parameter/{param}/station/{station}/period/{period}/data.csv'
    )

    with urlreq.urlopen(url) as response:
        try:
            data = response.read().decode('utf-8').split('\n')
        except urllib.error.HTTPError:
            print(
                f'No data available for parameter {param} at station {station}'
            )
            return None

    while data[0][:5] != 'Datum':
        data.remove(data[0])

    columns = ['Date', 'Time', 'Value', 'Quality', '', 'Information']
    data = [line.split(';') for line in data]
    data = pd.DataFrame(data[1:], columns=columns)

    dt = []
    values = []
    for _, r in data.iterrows():
        if r['Date'] and r['Time']:
            dt.append(
                datetime.strptime(r['Date'] + r['Time'], '%Y-%m-%d%H:%M:%S')
            )
        else:
            dt.append(pd.NaT)
        if r['Value']:
            values.append(float(r['Value']))
        else:
            values.append(np.nan)

    data['Datetime'] = dt
    data['Value'] = values

    return data


def load_station_data_file(pathname, sep=';', skiprows=range(10)):
    """
    Load a CSV data file downloaded from SMHI website.

    Parameters
    ----------
    pathname : str
        Path and filename of the file.
    sep : str, default: ';'
        Separator used in the file.
    skiprows : iterable, default: range(10)
        Rows to skip at the beginning of the file.
    """

    data = pd.read_csv(pathname, sep=sep, skiprows=skiprows)
    data['Datetime'] = [
        datetime.strptime(r['Datum'] + r['Tid (UTC)'], '%Y-%m-%d%H:%M:%S')
        for _, r in data.iterrows()
    ]

    return data


def load_MESAN_data(
    dts, dte, deltat, coord, rot=False,
    path_out='/home/jonathan/Documents/Metlake/SMHI/SMHI_data/'
):
    """
    Load MESAN data in GRIB format from the SMHI server for given dates and
    coordinates.

    Parameters
    ----------
    dts : datetime.datetime
        Datetime object of the first time of interest.
    dte : datetime.datetime
        Datetime object of the last time of interest.
    deltat : datetime.timedelta
        Timedelta object of the interval between each data point retrieved
        from the server.
    coord : list
        List of 2x1 tuples containing the coordinates (longitude, latitude)
        of each point of interest.
    rot : bool, default: False
        True if the grid needs to be rotated first.
    path_out : str, default: '/home/jonathan/Documents/Metlake/SMHI/SMHI_data/'
        Path to the folder where data should be saved.

    Example
    -------
    For ID_points = ['Venasjon', 'Parsen', 'SodraTeden',
                     'BD03', 'BD04', 'BD06', 'Malmslatt']
    from datetime import datetime, timedelta
    load_MESAN_data(
        dts=datetime(2018, 01, 01, 00, 00),
        dte=datetime(2018, 12, 31, 23, 00),
        deltat=timedelta(seconds=3600),
        coord=[
            (16.1825, 58.4563), (16.2053, 58.3421), (16.0205, 58.3432),
            (18.1332, 68.4477), (18.1581, 68.4474), (18.1657, 68.4429),
            (15.5327, 58.4004)
        ]
    )
    """

    # Determine if time period corresponds to the old or to the new
    # MESAN version
    if dts >= datetime(2016, 6, 1, 0, 0) and dte >= datetime(2016, 6, 1, 0, 0):
        feed = 6
    elif dts < datetime(2016, 6, 1, 0, 0) and dte < datetime(2016, 6, 1, 0, 0):
        feed = 4
    else:
        print(('Warning: Change in MESAN system between May and June 2016.\n'
               'Load data from periods before and after separately.'))
        return None

    # Initialize dt and print starting time
    dt = dts
    t_start = datetime.now()
    print('Start:', t_start)

    # Set up a DataFrame with time as Index and parameters as columns
    # Remark 1: cb_sig_b is labelled cb_sig in the GRIB files but the name
    #           is modified here to avoid having two columns with same label.
    # Remark 2: the DataFrame is reset for every new day of data in
    #           the "for" loop.
    params = [
        't', 'tmax', 'tmin', 'Tiw', 'gust', 'u', 'v', 'r', 'prec1h', 'prec3h',
        'prec12h', 'prec24h', 'frsn1h', 'frsn3h', 'frsn12h', 'frsn24h',
        'vis', 'MSL', 'tcc', 'c_sigfr', 'cb_sig', 'cb_sig_b', 'ct_sig', 'lcc',
        'prtype', 'prsort', 'mcc', 'hcc', 'sfgrd'
    ]
    time = [dt + n*deltat for n in range(24)]
    data = pd.DataFrame(columns=params, index=time)

    # Coordinates of Venasjon, Parsen, Sodra Teden, BD03, BD04, BD06,
    # Malmslatt respectively
    if rot:
        coord = [
            regrot(coord[0], coord[1], np.nan, np.nan, 15, -30)
            for coord in coord
        ]

    # Common part of the URL to access the data
    base_URL = f'http://opendata-download-grid-archive.smhi.se/data/{feed}/'

    # Variable to keep track of the progress and save files from time to time
    old_day = -1

    while True:
        # Stop condition. Save last data retrieved before exiting.
        if dt > dte:
            if dt.day != old_day:
                f_out = datetime.strftime(dt - timedelta(1), '%Y%m%d.pkl')
            else:
                f_out = datetime.strftime(dt, '%Y%m%d.pkl')
            pickle.dump(data, open(os.path.join(path_out, f_out), 'wb'))
            break

        print(dt)

        # Save file and start a new DataFrame for each day
        if dt.day != old_day and old_day != -1:
            f_out = datetime.strftime(dt - timedelta(1), '%Y%m%d.pkl')
            pickle.dump(data, open(os.path.join(path_out, f_out), 'wb'))
            time = [dt + n*deltat for n in range(24)]
            data = pd.DataFrame(columns=params, index=time)

        # URL to access the data
        month = datetime.strftime(dt, '%Y%m/')
        filename = datetime.strftime(dt, 'MESAN_%Y%m%d%H%M+000H00M')
        URL = base_URL + month + filename

        # Save data in memory
        r = requests.get(URL)
        with open(os.path.join(path_out, 'data.grib'), 'wb') as f_out:
            f_out.write(r.content)

        # Read data with ecCodes and process data
        with eccodes.GribFile(os.path.join(path_out, 'data.grib')) as grib:
            ind_closest = [np.nan]*len(coord)
            p_prev = ''
            for i in range(len(grib)):
                msg = eccodes.GribMessage(grib)
                p = msg['shortName']
                if np.isnan(ind_closest).all():
                    for ind, coor in enumerate(coord):
                        dist = (msg['longitudes'] - coor[0])**2 + \
                                (msg['latitudes'] - coor[1])**2
                        ind_closest[ind] = np.where(dist == dist.min())[0][0]
                if p == p_prev:
                    data.loc[dt, p + '_b'] = msg['values'][ind_closest]
                else:
                    data.loc[dt, p] = msg['values'][ind_closest]
                p_prev = p

        old_day = dt.day
        dt += deltat

    t_end = datetime.now()
    print('End:', t_end)
    print('Total processing time:', t_end - t_start)


def regrot(pxreg,pyreg,pxrot,pyrot,pxcen,pycen):
    """
    conversion between regular and rotated spherical coordinates.
    pxreg       longitudes of the regular coordinates
    pyreg       latitudes of the regular coordinates
    pxrot       longitudes of the rotated coordinates
    pyrot       latitudes of the rotated coordinates
                all coordinates given in degrees N (negative values for S)
                and degrees E (negative values for W)
    pxcen       regular longitude of the south pole of the rotated grid
    pycen       regular latitude of the south pole of the rotated grid
    """

    zrad = np.pi/180.
    zradi = 1./zrad
    zsycen = np.sin(zrad*(pycen + 90.))
    zcycen = np.cos(zrad*(pycen + 90.))

    if np.isnan(pxrot) and np.isnan(pyrot):
        zxmxc = zrad*(pxreg - pxcen)
        zsxmxc = np.sin(zxmxc)
        zcxmxc = np.cos(zxmxc)
        zsyreg = np.sin(zrad*pyreg)
        zcyreg = np.cos(zrad*pyreg)
        zsyrot = zcycen*zsyreg - zsycen*zcyreg*zcxmxc
        zsyrot = max([zsyrot, -1.0])
        zsyrot = min([zsyrot, 1.0])

        pyrot = np.arcsin(zsyrot)*zradi

        zcyrot = np.cos(pyrot*zrad)
        zcxrot = (zcycen*zcyreg*zcxmxc + zsycen*zsyreg)/zcyrot
        zcxrot = max([zcxrot, -1.0])
        zcxrot = min([zcxrot, 1.0])
        zsxrot = zcyreg*zsxmxc/zcyrot

        pxrot = np.arccos(zcxrot)*zradi

        if zsxrot < 0.:
            pxrot = -pxrot

        return pxrot, pyrot

    elif np.isnan(pxreg) and np.isnan(pyreg):
        zsxrot = np.sin(zrad*pxrot)
        zcxrot = np.cos(zrad*pxrot)
        zsyrot = np.sin(zrad*pyrot)
        zcyrot = np.cos(zrad*pyrot)
        zsyreg = zcycen*zsyrot + zsycen*zcyrot*zcxrot
        zsyreg = max([zsyreg, -1.0])
        zsyreg = min([zsyreg, 1.0])

        pyreg = np.arcsin(zsyreg)*zradi

        zcyreg = np.cos(pyreg*zrad)
        zcxmxc = (zcycen*zcyrot*zcxrot - zsycen*zsyrot)/zcyreg
        zcxmxc = max([zcxmxc, -1.0])
        zcxmxc = min([zcxmxc, 1.0])
        zsxmxc = zcyrot*zsxrot/zcyreg
        zxmxc = np.arccos(zcxmxc)*zradi
        if zsxmxc < 0.:
            zxmxc = -zxmxc

        pxreg = zxmxc + pxcen

        return pxreg, pyreg

    else:
        print('Regular OR rotated coordinates have to be NaN')


def sort_MESAN_data(path, ID_points):
    """
    Sort data downloaded with the function "load_MESAN_data" for several points.

    Data are stored in a dictionary of pd.DataFrame. Each key corresponds
    to the ID of one point for which data were downloaded from the SMHI server.

    Parameters
    ----------
    path : str
        Path to the directory where the data files are located.
    ID_points : list
        List of ID numbers or names for each point that has some data saved in
        the data files.
    """

    params = [
        't', 'tmax', 'tmin', 'Tiw', 'gust', 'u', 'v', 'r', 'prec1h', 'prec3h',
        'prec12h', 'prec24h', 'frsn1h', 'frsn3h', 'frsn12h', 'frsn24h',
        'vis', 'MSL', 'tcc', 'c_sigfr', 'cb_sig', 'cb_sig_b', 'ct_sig', 'lcc',
        'prtype', 'prsort', 'mcc', 'hcc', 'sfgrd'
    ]

    data = {ID: pd.DataFrame(columns=params) for ID in ID_points}

    def select(x, n):
        if isinstance(x, np.ndarray):
            return x[n]
        elif np.isnan(x):
            return x

    for fname in os.listdir(path):
        if fname[-3:] == 'pkl':
            d = pd.read_pickle(os.path.join(path, fname))
            for n, key in enumerate(ID_points):
                data[key] = pd.concat(
                    [data[key], d.applymap(lambda x: select(x, n))]
                )

    return data


def postprocess_MESAN_data(data_dict, h):
    """
    Calculate additional parameters (converting to different units
    or combining some initial parameters) and sort data according to time.

    Parameters
    ----------
    data_dict : dict
        Variable returned by the function "sort_MESAN_data".
    h : dict
        Altitude of the location to which data are related for each DataFrame
        contained in data_dict.
    """

    for key, data in data_dict.items():
        data['Temperature'] = data['t'] - 273.15
        data['Temperature max'] = data['tmax'] - 273.15
        data['Temperature min'] = data['tmin'] - 273.15
        data['Temperature wet-bulb'] = data['Tiw'] - 273.15
        data['Wind gust'] = data['gust']
        data['Wind speed'] = np.sqrt(data['u']**2 + data['v']**2)
        data['Wind direction'] = (90 - np.arctan2(-data['v'], -data['u'])*\
                                  180/np.pi)%360
        data['Relative humidity'] = data['r']*100
        data['Barometric pressure'] = data['MSL']/100*\
                (1 - 0.0065*h[key]/(data['t'] + 0.0065*h[key]))**5.257
        data.sort_index(inplace=True)

    return data_dict


def save_MESAN_data_to_Excel(data, path):
    """
    Save data returned by the function "sort_MESAN_data" to Excel.

    Parameters
    ----------
    data : dict
        Dictionary returned by the function "sort_MESAN_data".
    path : str
        Path to the file where data should be saved.
    """

    writer = pd.ExcelWriter(os.path.join(path))

    for key, d in data.items():
        d.to_excel(writer, sheet_name=key)

    writer.save()


def load_MESAN_file(pathname, sheet, h=0.0):
    """
    Load a file saved using the function "save_MESAN_data_to_Excel".
    Calculate additional parameters (wind speed and wind direction).

    Parameters
    ----------
    pathname : str
        Path and file name of the data file of interest.
    sheet : str
        Worksheet to be loaded.
    h : float, default: 0.0
        Altitude of the location to which data are related.
    """

    data = pd.read_excel(pathname, sheet_name=sheet)

    data['Temperature'] = data['t'] - 273.15
    data['Temperature max'] = data['tmax'] - 273.15
    data['Temperature min'] = data['tmin'] - 273.15
    data['Temperature wet-bulb'] = data['Tiw'] - 273.15
    data['Wind gust'] = data['gust']
    data['Wind speed'] = np.sqrt(data['u']**2 + data['v']**2)
    data['Wind direction'] = (90 - np.arctan2(-data['v'], -data['u'])\
                              *180/np.pi)%360
    data['Relative humidity'] = data['r']*100
    data['Barometric pressure'] = data['MSL']/100*\
            (1 - 0.0065*h/(data['t'] + 0.0065*h))**5.257

    return data


def load_STRANG_data(param, dts, dte, lat, lon):
    """
    Load STRANG data from the dedicated website.

    Parameters
    ----------
    param : int
        Code of the parameter of interest.
        116: CIE UV irradiance [mW/m2]
        117: Global irradiance [W/m2]
        118: Direct normal irradiance [W/m2]
        120: PAR [W/m2]
        121: Direct horizontal irradiance [W/m2]
        122: Diffuse irradiance [W/m2]
    dts : datetime.datetime
        Date and time of the first time of interest.
    dte : datetime.datetime
        Date and time of the last time of interest.
    lat : float
        Latitude of the point of interest in degrees.
    lon : float
        Longitude of the point of interest in degrees.
    """

    y1 = dts.year
    m1 = dts.month
    d1 = dts.day
    h1 = dts.hour
    y2 = dte.year
    m2 = dte.month
    d2 = dte.day
    h2 = dte.hour

    url = (
        f'http://strang.smhi.se/extraction/getseries.php?par={param}&m1={m1}'
        f'&d1={d1}&y1={y1}&h1={h1}&m2={m2}&d2={d2}&y2={y2}&h2={h2}&lat={lat}'
        f'&lon={lon}&lev=0'
    )

    with urlreq.urlopen(url) as url:
        data = url.read().decode('utf8')

    data = pd.DataFrame(
        data = [r.split(' ') for r in data.split('\n') if r],
        columns=['year', 'month', 'day', 'hour', 'value']
    )
    data['datetime'] = data.apply(
        lambda x: datetime.strptime(
            x['year'] + x['month'] + x['day'] + x['hour'], '%Y%m%d%H'
        ), axis=1
    )
    data['value'] = pd.to_numeric(data['value'])

    return data
