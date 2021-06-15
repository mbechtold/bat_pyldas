import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')
    matplotlib.use('QT4Agg')
import os
import os.path
import glob
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bat_pyldas.functions import *
from validation_good_practice.ancillary import metrics
import copy
from sympy import symbols, diff
import statsmodels.api as sm
from pyldas.functions import find_files

Epena = 0
Brunei = 0
waterlevel = 1

if Epena == 1:
    MERRA = 1  # 0 is GPM IMERG, 1 is MERRA-2
    TIME = 'condense' # late, early, condense or no value
    step = 'daily' # monthly or no value

    # load to in situ P data sets -QC is hard quality check
    insitu_P=pd.read_csv('/data/leuven/317/vsc31786/peatland_data/tropics/Rainfall/Epena/Epena_P.csv', sep=';', parse_dates=['date'])
    insitu_P=insitu_P.set_index('date')
    insitu_P.columns = ['rainfall']
    insitu_P['rainfall']=insitu_P['rainfall'].fillna(0)
    insitu_P_QC=pd.read_csv('/data/leuven/317/vsc31786/peatland_data/tropics/Rainfall/Epena/Epena_P_QC.csv', sep=';', parse_dates=['date'])
    insitu_P_QC=insitu_P_QC.set_index('date')
    insitu_P_QC.columns = ['rainfall']
    insitu_P_QC['rainfall']=insitu_P_QC['rainfall'].fillna(0)
    insitu_P_QC_monthly = insitu_P.resample('M').sum()

    # site coordinates for Epena
    [lon,lat]=[17.45,1.35]

    #load model rainfall data for Congo
    GPM=xr.open_dataset("/staging/leuven/stg_00024/OUTPUT/sebastiana/3B-HHR.MS.MRG.3IMERG.all.nc4")
    io = LDAS_io('daily', 'CONGO_AP_M09_PEATCLSMTN_v01', 'SMAP_EASEv2_M09', '/staging/leuven/stg_00024/OUTPUT/sebastiana')
    M2 = io.read_ts('Rainf', lon, lat, lonlat=True)

    # GPM lon lat extraction and preprocessing
    GPMloc = GPM.sel(lon=lon,lat=lat, method='nearest')
    GPMloc= GPMloc.to_dataframe()
    GPMloc['precipitationCal'] = GPMloc.precipitationCal.shift(periods=2, fill_value=0)
    GPMloc['precipitationCal'] = GPMloc['precipitationCal'].divide(2)
    GPMloc_daily = GPMloc.resample('D').sum()
    GPMloc_monthly= GPMloc.resample('M').sum()

    #####################################
    # to compare to M2 instead of GPM
    M2loc_daily = M2.to_frame()
    M2loc_daily.columns = ['precipitationCal']
    M2loc_monthly = M2loc_daily.resample('M').sum()

    if step == 'monthly':
        if MERRA == 1:
            df_tmp_p = pd.concat((insitu_P_QC_monthly, M2loc_monthly['precipitationCal']), axis=1)
            df_tmp_p = df_tmp_p.replace(0, pd.np.nan).dropna(axis=0, how='any').fillna(0).astype(int)
        else:
            df_tmp_p = pd.concat((insitu_P_QC_monthly, GPMloc_monthly['precipitationCal']), axis=1)
            df_tmp_p = df_tmp_p.replace(0, pd.np.nan).dropna(axis=0, how='any').fillna(0).astype(int)
    else:
        if MERRA == 1:
            df_tmp_p = pd.concat((insitu_P_QC, M2loc_daily['precipitationCal']), axis=1)
        else:
            df_tmp_p = pd.concat((insitu_P_QC, GPMloc_daily['precipitationCal']), axis=1)

    df_tmp_p.columns = ['data_obs', 'data_mod']

    #timerange
    if TIME == 'early':
        df_tmp_p=df_tmp_p.loc['2000-1-1 00:00:00' : '2007-12-31 00:00:00']
    elif TIME == 'late':
        df_tmp_p=df_tmp_p.loc['2015-1-1 00:00:00' : '2019-12-31 00:00:00']
    elif TIME == 'condense':
        df_tmp_p=df_tmp_p.loc['2000-1-1 00:00:00' : '2016-12-31 00:00:00']
    else:
        df_tmp_p=df_tmp_p

    df_tmp_p2 = copy.deepcopy(df_tmp_p)

    # metric calculation with df creation
    bias = metrics.bias(df_tmp_p2)  # Bias = bias_site[0]
    ubRMSD = metrics.ubRMSD(df_tmp_p2)  # ubRMSD = ubRMSD_site[0]
    pearson_R = metrics.Pearson_R(df_tmp_p2)  # Pearson_R = pearson_R_site[0]

    #plot figure
    plt.figure(figsize=(16, 8.5))
    fontsize = 14

    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3, fig=None)
    ax1.set_ylim(0,380)

    Title = 'bias = ' + str(bias[0]) + ', ubRMSD = ' + str(ubRMSD[0]) + ', R = ' + str(pearson_R[0])
    plt.title(Title,fontsize=fontsize)

    ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=3, fig=None)
    ax2.set_ylim(0,380)
    ax3 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3, fig=None)
    #ax3.set_ylim(0,350)

    df_tmp_p['data_obs'].plot(ax=ax1, fontsize=fontsize, style=['-'], color=['lawngreen'], linewidth=1)
    ax1.set_ylabel('precipitation [mm/day]',fontsize=fontsize)
    ax1.xaxis.set_visible(False)

    df_tmp_p['data_mod'].plot(ax=ax2, fontsize=fontsize, style=['-'], color=['magenta'], linewidth=1)
    #ax2.set_ylabel('precipitation [mm/month]', fontsize=fontsize)
    ax2.xaxis.set_visible(False)

    #df_tmp_GPM.plot(ax=ax3, fontsize=fontsize, style=['-', '-'], color=['lawngreen','magenta'], linewidth=.5)
    #ax3.set_ylabel('precipitation [mm/d]', fontsize=fontsize)
    #ticklabels = ['']*len(df_tmp_p.index)
    #ticklabels[::300] = [item.strftime('%b %d\n%Y') for item in df_tmp_p.index[::300]]
    #ax3.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    ax3.xaxis.set_visible(True)
    ax3.legend(['In situ precipitation Epena', 'IMERG GPM'], fontsize=fontsize)

    # cumulative rainfall plot
    df_tmp_p_sum = df_tmp_p.dropna(axis=0, how='any')
    df_tmp_p_sum = df_tmp_p_sum[['data_obs', 'data_mod']].cumsum(skipna=True)
    df_tmp_p_sum.plot(ax=ax3, fontsize=fontsize, style=['-', '-'], color=['lawngreen','magenta'], linewidth=1)
    ax3.set_ylabel('Cumulative precipitation [mm]',fontsize=fontsize)

    if MERRA == 1:
        if TIME == 'early':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_MERRA-2'+'_cum_early', dpi=350)
        elif TIME == 'late':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_MERRA-2'+'_cum_late', dpi=350)
        elif TIME == 'condense':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_MERRA-2'+'_cum_condense', dpi=350)
        else:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_MERRA-2'+'_cum', dpi=350)

    else:
        if TIME == 'early':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_GPMIMERG'+'_cum_early', dpi=350)
        elif TIME == 'late':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_GPMIMERG'+'_cum_late', dpi=350)
        elif TIME == 'condense':
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_GPMIMERG' + '_cum_condense',dpi=350)
        else:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/EpenaP'+'_GPMIMERG'+'_cum', dpi=350)

    plt.close()


elif Brunei == 1:

    MERRA = 0  # 0 is GPM IMERG, 1 is MERRA-2
    step = 'daily'  # monthly or no value

    #plot for Alex Cobb's trial precipitation
    insitu_P=pd.read_csv('/data/leuven/317/vsc31786/peatland_data/tropics/Rainfall/Cobb_Brunei/processed/Daily/BR_mdm_trail_10_daily.csv', sep=',', parse_dates=['datetime'])
    insitu_P=insitu_P.set_index('datetime')
    insitu_P.columns = ['rainfall']
    insitu_P['rainfall']=insitu_P['rainfall'].fillna(0)
    insitu_P_monthly = insitu_P.resample('M').sum()

    [lon,lat]=[114.355,4.371]

    GPM=xr.open_dataset("/staging/leuven/stg_00024/OUTPUT/sebastiana/mendaram_gpm_imerg.nc4")
    io = LDAS_io('daily', 'INDONESIA_AP_M09_PEATCLSMTN_v01', 'SMAP_EASEv2_M09', '/staging/leuven/stg_00024/OUTPUT/sebastiana')
    M2 = io.read_ts('Rainf', lon, lat, lonlat=True)

    GPMloc = GPM.sel(lon=lon,lat=lat, method='nearest')
    GPMloc= GPMloc.to_dataframe()
    GPMloc['precipitationCal'] = GPMloc.precipitationCal.shift(periods=16, fill_value=0)
    GPMloc['precipitationCal'] = GPMloc['precipitationCal'].divide(2)
    GPMloc_daily = GPMloc.resample('D').sum()
    GPMloc_monthly= GPMloc.resample('M').sum()

    M2loc_daily = M2.to_frame()
    M2loc_daily.columns = ['precipitationCal']
    M2loc_monthly = M2loc_daily.resample('M').sum()

    if step == 'monthly':
        if MERRA ==1:
            df_tmp_p = pd.concat((insitu_P_monthly, M2loc_monthly['precipitationCal']), axis=1)
            df_tmp_p = df_tmp_p.replace(0, pd.np.nan).dropna(axis=0, how='any').fillna(0).astype(int)
        else:
            df_tmp_p = pd.concat((insitu_P_monthly, GPMloc_monthly['precipitationCal']), axis=1)
            df_tmp_p = df_tmp_p.replace(0, pd.np.nan).dropna(axis=0, how='any').fillna(0).astype(int)
    else:
        if MERRA ==1:
            df_tmp_p = pd.concat((insitu_P, M2loc_daily['precipitationCal']), axis=1)
        else:
            df_tmp_p = pd.concat((insitu_P, GPMloc_daily['precipitationCal']), axis=1)

    df_tmp_p = df_tmp_p.loc['2012-2-1 00:00:00': '2013-1-31 00:00:00']
    df_tmp_p.columns = ['data_obs', 'data_mod']

    df_tmp_p2 = copy.deepcopy(df_tmp_p)

    # metric calculation with df creation
    bias = metrics.bias(df_tmp_p2)  # Bias = bias_site[0]
    ubRMSD = metrics.ubRMSD(df_tmp_p2)  # ubRMSD = ubRMSD_site[0]
    pearson_R = metrics.Pearson_R(df_tmp_p2)  # Pearson_R = pearson_R_site[0]

    # plot figure
    plt.figure(figsize=(16, 8.5))
    fontsize = 14

    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3, fig=None)

    Title = 'bias = ' + str(bias[0]) + ', ubRMSD = ' + str(ubRMSD[0]) + ', R = ' + str(pearson_R[0])
    plt.title(Title, fontsize=fontsize)

    ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=3, fig=None)
    ax3 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3, fig=None)

    if step == 'monthly':
        ax2.set_ylim(0, 450)
        ax1.set_ylim(0, 450)
    else:
        ax2.set_ylim(0, 100)
        ax1.set_ylim(0, 100)

    df_tmp_p['data_obs'].plot(ax=ax1, fontsize=fontsize, style=['-o'], color=['lawngreen'], linewidth=1)
    ax1.set_ylabel('precipitation [mm/day]', fontsize=fontsize)
    ax1.xaxis.set_visible(False)

    df_tmp_p['data_mod'].plot(ax=ax2, fontsize=fontsize, style=['-o'], color=['magenta'], linewidth=1)
    # ax2.set_ylabel('precipitation [mm/month]', fontsize=fontsize)
    ax2.xaxis.set_visible(False)

    # df_tmp_GPM.plot(ax=ax3, fontsize=fontsize, style=['-', '-'], color=['lawngreen','magenta'], linewidth=.5)
    # ax3.set_ylabel('precipitation [mm/d]', fontsize=fontsize)
    # ticklabels = ['']*len(df_tmp_p.index)
    # ticklabels[::300] = [item.strftime('%b %d\n%Y') for item in df_tmp_p.index[::300]]
    # ax3.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    ax3.xaxis.set_visible(True)
    ax3.legend(['In situ precipitation Epena', 'IMERG GPM'], fontsize=fontsize)

    # cumulative rainfall plot
    df_tmp_p_sum = df_tmp_p.dropna(axis=0, how='any')
    df_tmp_p_sum = df_tmp_p_sum[['data_obs', 'data_mod']].cumsum(skipna=True)
    df_tmp_p_sum.plot(ax=ax3, fontsize=fontsize, style=['-', '-'], color=['lawngreen', 'magenta'], linewidth=1)
    ax3.set_ylabel('Cumulative precipitation [mm]', fontsize=fontsize)

    if step == 'monthly':
        if MERRA == 1:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/Brunei' + '_MERRA-2_monthly',dpi=350)
        else:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/Brunei' + '_GPMIMERG_monthly',dpi=350)
    else:
        if MERRA ==1:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/Brunei' + '_MERRA-2',dpi=350)
        else:
            plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/Brunei' + '_GPMIMERG',dpi=350)



elif waterlevel == 1:
    #load hourly MERRA-2 data
    #period from 2000-2020
    M2_path ='/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing/'
    filenames1 = glob.glob(M2_path + '*300/*/*/*/*_flx_*.nc4', recursive =True)
    filenames2 = glob.glob(M2_path + '*400/*/*/*/*_flx_*.nc4', recursive =True)
    filenames=filenames1+filenames2

    #construct variables to exclude because the use of 'data_vars' argument doesn't work...
    testname = glob.glob(M2_path + 'MERRA2_300/diag/Y2005/M04/*_flx_*.nc4')
    M2_datavars_list = xr.open_mfdataset(testname)
    varlist = list(M2_datavars_list.variables)
    #varlist.remove(['lon' 'lat' 'PRECTOTCORR'])
    M2_hourly = xr.open_mfdataset(filenames)

    [lon,lat]= []
    [lon,lat]= []


#/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing/MERRA2_300/diag/Y2005/M04

else:
    print('no comparison')