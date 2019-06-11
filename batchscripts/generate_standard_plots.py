#!/usr/bin/env python

root='/scratch/leuven/329/vsc32924/output'
exp = 'INDONESIA_M09_v01_spinup'
domain = 'SMAP_EASEv2_M09'


outpath = '/vsc-hard-mounts/leuven-data/329/vsc32924/figures_temp'

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *

os.makedirs(outpath,exist_ok=True)

# time series
plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=1.344, lon=101.411)

# maps
#plot_catparams(exp, domain, root, outpath)
#plot_zbar_std(exp, domain, root, outpath)
#plot_waterstorage_std(exp, domain, root, outpath)
#plot_sfmc_std(exp, domain, root, outpath)


#plot_RTMparams(exp, domain, root, outpath)
#plot_rtm_parameters(exp, domain, root, outpath)
