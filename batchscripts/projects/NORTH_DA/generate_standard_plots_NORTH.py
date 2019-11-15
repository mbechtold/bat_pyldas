#!/usr/bin/env python

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *
from bat_pyldas.functions import *
import getpass

root='/staging/leuven/stg_00024/OUTPUT/michelb'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp'

if getpass.getuser()=='michel':
    root='/home/michel/backup_HPC/output'
    outpath = '/home/michel/backup_HPC/FIG_tmp'

#root='/scratch/leuven/317/vsc31786/output'

exp = 'SMAP_EASEv2_M09_SMOSfw_DA'
domain = 'SMAP_EASEv2_M09'

exp1 = 'SMAP_EASEv2_M09_CLSM_SMOSfw_DA'
exp2 = 'SMAP_EASEv2_M09_SMOSfw_DA'

dti = pd.date_range('2018-08-06', periods=2, freq='3H')
dti_kg = pd.date_range('2010-01-01', periods=365*9*8, freq='3H')

os.makedirs(outpath,exist_ok=True)

#plot_innov_std_quatro(exp, domain, root, outpath)
#plot_kalman_gain(exp, domain, root, outpath, dti_kg)
#plot_innov_quatro(exp, domain, root, outpath, dti)
#plot_innov_norm_std_quatro(exp, domain, root, outpath)
#plot_innov_delta_std_quatro(exp1, exp2, domain, root, outpath)
#plot_innov_delta_std_single(exp1, exp2, domain, root, outpath)
#anomaly_JulyAugust_zbar(exp, domain, root, outputpath)
#anomaly_JulyAugust_zbar(exp, domain, root, outpath)
#plot_anomaly_JulyAugust_zbar(exp, domain, root, outpath)
plot_peat_and_sites(exp, domain, root, outpath)
#plot_filter_diagnostics(exp, domain, root, outpath)
#plot_filter_diagnostics_delta(exp1, exp2, domain, root, outpath)
#plot_filter_diagnostics_gs(exp, domain, root, outpath)
#plot_filter_diagnostics_short(exp, domain, root, outpath)
#plot_timeseries(exp, domain, root, outpath, lat=48.0, lon=51.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=48.0, lon=51.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=52.0, lon=63.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=52.7, lon=41.)
#plot_timeseries(exp, domain, root, outpath, lat=53.0, lon=38.5)
#plot_increments_number(exp, domain, root, outpath)
#plot_catparams(exp, domain, root, outpath)
#plot_scaling_parameters(exp, domain, root, outpath)
#plot_scaling_parameters_average(exp, domain, root, outpath)
#plot_scaling_delta(exp1, exp2, domain, root, outpath)
#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
#plot_increments_std(exp, domain, root, outpath)
#plot_increments(exp, domain, root, outpath, dti)
#plot_Fcst_std_delta(exp1, exp2, domain, root, outpath)
#plot_lag1_autocor(exp, domain, root, outpath)
#plot_waterstorage_std(exp, domain, root, outpath)
#plot_increments_std_delta(exp1, exp2, domain, root, outpath)
#plot_obs_std_single(exp, domain, root, outpath)
#plot_obs_ana_std_quatro(exp, domain, root, outpath)
#plot_obs_obs_std_quatro(exp, domain, root, outpath)
#plot_obs_fcst_std_quatro(exp, domain, root, outpath)
#plot_sfmc_std(exp, domain, root, outpath)
#plot_daily_delta(exp1, exp2, domain, root, outpath)

