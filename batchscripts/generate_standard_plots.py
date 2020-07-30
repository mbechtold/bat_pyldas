#!/usr/bin/env python

# batch script to generate standard plots from netcdf files of ldas output
# + first evaluation of simulation experiment with availabe in situ data

import os
import platform
import csv
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# pyldas and related own packages
from bat_pyldas.plotting import *
from bat_pyldas.functions import read_wtd_data
from validation_good_practice.ancillary import metrics

plot_maps = 0
plot_map_peat_sites = 0
plot_insitu = 1
plot_ET_insitu = 0
plot_insitu_multiple_exp = 0

# maps
if plot_maps==1:
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'INDONESIA2_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/whitelist/Natural/IN2'
    # Catchment Parameters
    plot_catparams(exp, domain, root, outpath)
    # Temporal mean and standard deviation of variables
    #plot_all_variables_temporal_moments(exp, domain, root, outpath)

if plot_map_peat_sites==1:
    root='/staging/leuven/stg_00024/OUTPUT/michelb'
    exp = 'PEATREV_PEATMAPHWSD'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison'
    ###
    plot_peat_and_sites(exp, domain, root, outpath)

# insitu
if plot_insitu==1:
    ## in situ data
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'INDONESIA2_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/whitelist/Natural/IN2'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE_ALLDorN.csv'
    ####
    wtd_obs, wtd_mod, precip_obs, precip_mod = read_wtd_data(insitu_path, mastertable_filename, exp, domain, root)
    plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, precip_mod, exp, outpath)

if plot_ET_insitu == 1:
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'INDONESIA2_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/whitelist/Natural/IN2/ET'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics'
    mastertable_filename = 'ET_TROPICS_MASTER_TABLE.csv'
    ####
    et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod, le_obs, le_mod , zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod, Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc = read_et_data(insitu_path, mastertable_filename, exp, domain, root)
    plot_skillmetrics_comparison_et(et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod, le_obs, le_mod, zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod, Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc, exp, outpath)

# insitu
if plot_insitu_multiple_exp==1:
    ## in situ data
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp1 = 'CONGO_M09_CLSM_v01'
    exp2 = 'CONGO_M09_PEATCLSM_v01'
    exp3a = 'CONGO_M09_PEATCLSMTN_v01'
    exp3b = 'CONGO_M09_PEATCLSMTD_v01'
    exp1 = 'INDONESIA_M09_CLSM_v01'
    exp2 = 'INDONESIA_M09_PEATCLSM_v01'
    exp3a = 'INDONESIA_M09_PEATCLSMTN_v01'
    exp3b = 'INDONESIA_M09_PEATCLSMTD_v01'
    exp1 = 'WHITELIST_M09_PEATCLSMTN_v01'
    exp2 = 'INDONESIA_M09_PEATCLSMTN_v01'
    exp3a = 'CONGO_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/whitelist/Natural/IN'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data'
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
    ####
    wtd_obs, wtd_mod_exp1, precip_obs, precip_mod_exp1 = read_wtd_data(insitu_path, mastertable_filename, exp1, domain, root)
    wtd_obs, wtd_mod_exp2, precip_obs,  precip_mod_exp2 = read_wtd_data(insitu_path, mastertable_filename, exp2, domain, root)
    wtd_obs, wtd_mod_exp3a, precip_obs, precip_mod_exp3a = read_wtd_data(insitu_path, mastertable_filename, exp3a, domain, root)
    wtd_obs, wtd_mod_exp3b, precip_obs, precip_mod_exp3b = read_wtd_data(insitu_path, mastertable_filename, exp3b, domain, root)
    #plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod_exp3a, precip_obs, exp3a, outpath)
    wtd_mod = [wtd_mod_exp1, wtd_mod_exp2, wtd_mod_exp3a, wtd_mod_exp3b]
    plot_skillmetrics_comparison_wtd_multimodel(wtd_obs, wtd_mod, precip_obs, exp3a, outpath)

# to be checked:
#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
# time series
#plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=1.344, lon=101.411)
