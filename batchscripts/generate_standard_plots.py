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
plot_insitu = 0
plot_figure_maps = 0
plot_map_peat_sites = 0
plot_ET_insitu = 0
plot_insitu_multiple_exp = 0

# maps
if plot_maps==1:
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    root='/scratch/leuven/317/vsc31786/output'
    exp = 'SMOS_mwRTM_EASEv2_M09_PEATCLSM_IcPmNO_D3'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/SA/Natural/'
    # Catchment Parameters
    plot_catparams(exp, domain, root, outpath)
    # Temporal mean and standard deviation of variables
    plot_all_variables_temporal_moments(exp, domain, root, outpath)

if plot_figure_maps==1:
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'SAMERICA_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/paper'
    # Temporal mean and standard deviation of variables
    plot_all_temporal_maps(exp, domain, root, outpath)

if plot_map_peat_sites==1:
    root='/staging/leuven/stg_00024/OUTPUT/michelb/'
    exp = 'PEATREV_PEATMAPHWSD'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison'
    ###
    plot_peat_and_sites(exp, domain, root, outpath)

# insitu
if plot_insitu==1:
    ## in situ data
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'SAMERICA_AP_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/SA/Natural/AP'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE_ALLDorN.csv'
    mastertable_filename = 'WTD_peatlands_global_WGS84.csv'
    ####
    wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc, rzexc, catdef = read_wtd_data(insitu_path, mastertable_filename, exp, domain, root)
    plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc, rzexc, catdef, exp, outpath)

if plot_ET_insitu == 1:
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'INDONESIA_AP_M09_PEATCLSM_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Northern/AP/Natural_sites/ET'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics'
    mastertable_filename = 'ET_TROPICS_MASTER_TABLE.csv'
    ####
    et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod, le_obs, le_mod , zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod, Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc,  rzmc, srfexc, rzexc, catdef, Qair, vpd_obs, Wind = read_et_data(insitu_path, mastertable_filename, exp, domain, root)
    plot_skillmetrics_comparison_et(et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod, le_obs, le_mod, zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod, Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc, rzmc, srfexc, rzexc, catdef, Qair, vpd_obs, Wind, exp, outpath)

# insitu
if plot_insitu_multiple_exp==1:
    ## in situ data
    root='/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp1 = 'CONGO_M09_CLSM_v01'
    exp2 = 'CONGO_M09_PEATCLSM_v01'
    #exp3a = 'CONGO_M09_PEATCLSMTN_v01'
    #exp3b = 'CONGO_M09_PEATCLSMTD_v01'
    #exp1 = 'INDONESIA_M09_CLSM_v01'
    #exp2 = 'INDONESIA_M09_PEATCLSM_v01'
    #exp3a = 'INDONESIA_M09_PEATCLSMTN_v01'
    #exp3b = 'INDONESIA_M09_PEATCLSMTD_v01'
    exp1 = 'INDONESIA3_M09_PEATCLSMTN_v01'
    exp2 = 'INDONESIA_M09_CLSM_v01'
    #exp3a = 'CONGO_M09_PEATCLSMTN_v01'
    domain = 'SMAP_EASEv2_M09'
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/CLSM/Comparison'
    os.makedirs(outpath,exist_ok=True)
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
    ####
    wtd_obs, wtd_mod_exp1, precip_obs, precip_mod_exp1, sfmc_mod_exp1, rzmc_mod_exp1, srfexc_exp1, rzexc_exp1, catdef_exp1 = read_wtd_data(insitu_path, mastertable_filename, exp1, domain, root)
    wtd_obs, wtd_mod_exp2, precip_obs,  precip_mod_exp2, sfmc_mod_exp2, rzmc_mod_exp2, srfexc_exp2, rzexc_exp2, catdef_exp2 = read_wtd_data(insitu_path, mastertable_filename, exp2, domain, root)
    #wtd_obs, wtd_mod_exp3a, precip_obs, precip_mod_exp3a = read_wtd_data(insitu_path, mastertable_filename, exp3a, domain, root)
    #wtd_obs, wtd_mod_exp3b, precip_obs, precip_mod_exp3b = read_wtd_data(insitu_path, mastertable_filename, exp3b, domain, root)
    #plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod_exp3a, precip_obs, exp3a, outpath)
    wtd_mod = [wtd_mod_exp1, wtd_mod_exp2]
    precip_mod = [precip_mod_exp1, precip_mod_exp2]
    sfmc_mod = [sfmc_mod_exp1, sfmc_mod_exp2]
    rzmc_mod = [rzmc_mod_exp1, rzmc_mod_exp2]
    srfexc = [srfexc_exp1, srfexc_exp2]
    rzexc = [rzexc_exp1, rzexc_exp2]
    catdef = [catdef_exp1, catdef_exp2]
    plot_skillmetrics_comparison_wtd_multimodel(wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc, rzexc, catdef, exp1, outpath)

# to be checked:
#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
# time series
#plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=1.344, lon=101.411)
