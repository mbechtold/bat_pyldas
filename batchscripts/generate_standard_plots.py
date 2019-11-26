#!/usr/bin/env python

# batch script to generate standard plots from netcdf files of ldas output
# + first evaluation of simulation experiment with availabe in situ data

import os
import platform
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
plot_insitu = 1
plot_insitu_multiple_exp = 0

# maps
if plot_maps==1:
    root='/scratch/leuven/317/vsc31786/output/TROPICS'
    exp = 'INDONESIA_M09_PEATCLSMTD_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Drained'
    # Catchment Parameters
    plot_catparams(exp, domain, root, outpath)
    # Temporal mean and standard deviation of variables
    plot_all_variables_temporal_moments(exp, domain, root, outpath)

# insitu
if plot_insitu==1:
    ## in situ data
    root='/scratch/leuven/317/vsc31786/output/TROPICS'
    exp = 'CONGO_M09_PEATCLSMTN_v01'
    exp = 'INDONESIA_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Natural'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
    ####
    wtd_obs, wtd_mod, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp, domain, root)
    plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, exp, outpath)

# insitu
if plot_insitu_multiple_exp==1:
    ## in situ data
    root='/scratch/leuven/317/vsc31786/output/TROPICS'
    exp1 = 'CONGO_M09_CLSM_v01'
    exp2 = 'CONGO_M09_PEATCLSM_v01'
    exp3a = 'CONGO_M09_PEATCLSMTN_v01'
    exp3b = 'CONGO_M09_PEATCLSMTD_v01'
    exp1 = 'INDONESIA_M09_CLSM_v01'
    exp2 = 'INDONESIA_M09_PEATCLSM_v01'
    exp3a = 'INDONESIA_M09_PEATCLSMTN_v01'
    exp3b = 'INDONESIA_M09_PEATCLSMTD_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Northern'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data'
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
    ####
    wtd_obs, wtd_mod_exp1, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp1, domain, root)
    wtd_obs, wtd_mod_exp2, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp2, domain, root)
    wtd_obs, wtd_mod_exp3a, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp3a, domain, root)
    wtd_obs, wtd_mod_exp3b, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp3b, domain, root)
    #plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod_exp3a, precip_obs, exp3a, outpath)
    wtd_mod = [wtd_mod_exp2, wtd_mod_exp3a, wtd_mod_exp3b]
    plot_skillmetrics_comparison_wtd_multimodel(wtd_obs, wtd_mod, precip_obs, exp3a, outpath)

# to be checked:
#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
# time series
#plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=1.344, lon=101.411)
