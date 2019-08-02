#!/usr/bin/env python

root='/scratch/leuven/317/vsc31786/output'
root='/staging/leuven/stg_00024/OUTPUT/michelb'
exp = 'SMAP_EASEv2_M09_SI_SMOSfw_DA'
domain = 'SMAP_EASEv2_M09'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp'

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *

os.makedirs(outpath,exist_ok=True)

# time series
plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=60.0, lon=75.0)

# maps
#plot_catparams(exp, domain, root, outpath)
#plot_zbar_std(exp, domain, root, outpath)
#plot_waterstorage_std(exp, domain, root, outpath)
#plot_sfmc_std(exp, domain, root, outpath)


#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
