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
root='/scratch/leuven/317/vsc31786/output'
#root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
#root='/staging/leuven/stg_00024/OUTPUT/michelb'
#root='/scratch/leuven/317/vsc31786/output'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/'
#outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/RSE_revision'
outpath = '/data/leuven/317/vsc31786/FIG_tmp/D3'

if getpass.getuser()=='michel':
    root='/home/michel/backup_HPC/output'
    outpath = '/home/michel/backup_HPC/FIG_tmp/RSE_revision'

#root='/scratch/leuven/317/vsc31786/output'

exp = 'SMAP_EASEv2_M09_CLSM_SMOSfw_SMOSv102'
exp='GLOB_M36_7Thv_TWS_FOV0_M2'
exp = 'SMAP_EASEv2_M09_CLSM_SMOSfw'
exp = 'PEATREV_HWSD'
exp = 'SMAP_EASEv2_M09_INLv2_PEATMAP_SMOSfw'
exp = 'CONGO_M09_PEATCLSMTN_v02_SMOSfw_OL'
exp = 'INDONESIA2_M09_PEATCLSMTN_v01'
#exp = 'SMAP_EASEv2_M09_CLSM_SMOSfw'
#exp= 'SMOSrw_mwRTM_EASEv2_M09_CLSM_NORTH'
#exp = 'SMAP_EASEv2_M09_CLSM_SMOSfw_L4_SM_v4_RTMparam'
domain = 'SMAP_EASEv2_M09'
#domain = 'SMAP_EASEv2_M36_GLOB'

exp1 = 'SMAP_EASEv2_M09_CLSM_SMOSfw_DA'
exp2 = 'SMAP_EASEv2_M09_SMOSfw_DA'
exp1 = 'INDONESIA_M09_PEATCLSMTN_v01'
exp2 = 'INDONESIA2_M09_PEATCLSMTN_v01'
exp = 'CONGO_M09_PEATCLSMTN_v02_spinup'
exp1 = 'CONGO_M09_PEATCLSMTN_v02_spinup'
exp2 = 'CONGO_M09_PEATCLSMTN_v02_spinup'

exp1 = 'SMOS_mwRTM_EASEv2_M09_CLSM_IcNO_DA'
exp2 = 'SMOS_mwRTM_EASEv2_M09_PEATCLSM_IcPmNO_DA'

exp1 = 'SMOS_mwRTM_EASEv2_M09_PEATCLSM_IcPmNO_D2'
exp2 = 'SMOS_mwRTM_EASEv2_M09_PEATCLSM_IcPmNO_D3'
exp = 'SMOS_mwRTM_EASEv2_M09_PEATCLSM_IcPmNO_D3'

dti = pd.date_range('2018-08-06', periods=2, freq='3H')
dti_kg = pd.date_range('2010-01-01', periods=365*9*8, freq='3H')

os.makedirs(outpath,exist_ok=True)

#plot_zoom_example(exp, domain, root, outpath, param='daily')

plot_maps=0
if plot_maps==1:
    # Catchment Parameters
    # plot_catparams(exp, domain, root, outpath)
    # Temporal mean and standard deviation of variables
    plot_all_variables_temporal_moments(exp, domain, root, outpath, param='daily')

#plot_innov_std_quatro(exp, domain, root, outpath)
#plot_kalman_gain(exp, domain, root, outpath, dti_kg)
#plot_innov_quatro(exp, domain, root, outpath, dti)
#plot_innov_norm_std_quatro(exp, domain, root, outpath)
#plot_innov_delta_std_quatro(exp1, exp2, domain, root, outpath)
#plot_innov_delta_std_single(exp1, exp2, domain, root, outpath)
#anomaly_JulyAugust_zbar(exp, domain, root, outputpath)
#anomaly_JulyAugust_zbar(exp, domain, root, outpath)
#plot_anomaly_JulyAugust_zbar(exp, domain, root, outpath)
#plot_peat_and_sites(exp, domain, root, outpath)
#plot_filter_diagnostics(exp, domain, root, outpath)

#plot_filter_diagnostics_delta(exp1, exp2, domain, root, outpath)

#plot_filter_diagnostics_gs(exp, domain, root, outpath)
#plot_filter_diagnostics_short(exp, domain, root, outpath)
#plot_timeseries(exp, domain, root, outpath, lat=48.0, lon=51.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=48.0, lon=51.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=52.0, lon=63.0)
#plot_timeseries(exp, exp1, domain, root, outpath, lat=52.7, lon=41.)
#plot_timeseries_fsw_change(exp, domain, root, outpath)
plot_map_cum_fsw_change_for_same_wtd(exp, domain, root, outpath)
#plot_timeseries_RSEpaper(exp1, exp2, domain, root, outpath, lat=52.745719, lon=-83.9750042)

#plot_timeseries(exp, domain, root, outpath, lat=53.0, lon=38.5)
#plot_increments_number(exp, domain, root, outpath)
#plot_catparams(exp, domain, root, outpath)
#ToGeoTiff_catparams(exp, domain, root, outpath)
#plot_scatter_daily_output(exp1, domain, root, outpath)
#plot_timeseries_daily_compare(exp1, exp2, domain, root, outpath)
#plot_scaling_parameters_timeseries(exp, domain, root, outpath,
#                        scalepath=os.path.join(root,exp,'output/SMAP_EASEv2_M09/stats/z_score_clim/pentad/'),
#                        scalename='Thvf_TbSM_001_SMOS_zscore_stats_2010_p37_2012_p73_hscale_0.00_W_9p_Nmin_20',
#                                   lon=74.,
#                                   lat=52.)
#plot_scaling_parameters(exp, domain, root, outpath,
#                        scalepath=os.path.join(root,exp,'output/SMAP_EASEv2_M09/stats/z_score_clim/pentad/'),
#                   scalename='Thvf_TbSM_001_SMOS_zscore_stats_2010_p37_2012_p73_hscale_0.00_W_9p_Nmin_20')
#plot_scaling_parameters_average(exp, domain, root, outpath)
#plot_RTMparams_delta(exp1, exp2, domain, root, outpath)
#plot_scaling_delta(exp1, exp2, domain, root, outpath)
#plot_RTMparams(exp, domain, root, outpath)
#plot_RTMparams_filled(exp, domain, root, outpath)
#plot_RTMparams_filled_peat(exp, domain, root, outpath)
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

