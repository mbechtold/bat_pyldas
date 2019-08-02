import pandas as pd
import os, sys
import datetime

input_path = '/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Brunei/water_level'
output_path = '/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Brunei/processed/WTD'
input_path_p = '/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Brunei/'
output_path_p = '/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Brunei/processed/Precipitation'

list_sites = os.listdir(input_path)

# Precipitation (same for all stations):
df_p = pd.read_csv(input_path_p + '/' + 'throughfall.csv', sep=r'\t')
df_p.columns = ['date', 'precipitation']
df_p['date'] = pd.to_datetime(df_p['date'])
df_p = df_p.set_index('date')

df_p_Hourly = df_p.resample('H').sum()  # Hourly precipitation.
df_p_Daily = df_p.resample('D').sum()  # Daily precipitation.

df_p.index = df_p.index.strftime('%Y/%m/%d %H:%M')  # Subhourly format.
df_p.index.names = ['date']

df_p_Hourly.index = df_p_Hourly.index.strftime('%Y/%m/%d %H:%M')  # Hourly format.
df_p_Hourly.index.names = ['date']

df_p_Daily.index = df_p_Daily.index.strftime('%Y/%m/%d')  # Daily format.
df_p_Daily.index.names = ['date']

file_name = 'Brunei_Darussalam.csv'

# Save subhourly data.
df_p.to_csv(output_path_p + '/Subhourly/' + file_name)

# Save hourly data.
df_p_Hourly.to_csv(output_path_p + '/Hourly/' + file_name)

# Save daily data.
df_p_Daily.to_csv(output_path_p + '/Daily/' + file_name)


# wtd:
for i,site_name in enumerate(list_sites):
    df = pd.read_csv(input_path + '/' + site_name, sep=r'\t')   # Data is read in subhourly format.
    df.columns = ['date', 'wtd']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df_Hourly = df.resample('H').median()   # Hourly median wtd.
    df_Daily = df.resample('D').median()   # Daily median wtd.

    df.index = df.index.strftime('%Y/%m/%d %H:%M') # Subhourly format.
    df.index.names = ['date']

    df_Hourly.index = df_Hourly.index.strftime('%Y/%m/%d %H:%M')    # Hourly format.
    df_Hourly.index.names = ['date']

    df_Daily.index = df_Daily.index.strftime('%Y/%m/%d')    # Daily format.
    df_Daily.index.names = ['date']

    A = site_name.split("_")
    site_ID = 'BR_' + A[2] + '_' + A[3] + '_' + A[4]

    # Save subhourly data.
    df.to_csv(output_path + '/Subhourly/' + site_ID)

    # Save hourly data.
    df_Hourly.to_csv(output_path + '/Hourly/' + site_ID)

    # Save daily data.
    df_Daily.to_csv(output_path + '/Daily/' + site_ID)