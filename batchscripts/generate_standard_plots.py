#!/usr/bin/env python

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')


root='/scratch/leuven/329/vsc32924/output'
exp = 'INDONESIA_M09_v01_spinup'
domain = 'SMAP_EASEv2_M09'


outpath = '/vsc-hard-mounts/leuven-data/329/vsc32924/figures_temp'
os.makedirs(outpath,exist_ok=True)

from bat_pyldas.plotting import plot_catparams
plot_catparams(exp, domain, root, outpath)

from bat_pyldas.plotting import plot_RTMparams
#plot_RTMparams(exp, domain, root, outpath)

from pyldas.visualize.plots import plot_rtm_parameters
#plot_rtm_parameters(exp, domain, root, outpath)

from bat_pyldas.plotting import plot_waterstorage_std
plot_waterstorage_std(exp, domain, root, outpath)

from bat_pyldas.plotting import plot_sfmc_std
plot_sfmc_std(exp, domain, root, outpath)

