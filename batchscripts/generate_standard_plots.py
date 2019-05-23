#!/usr/bin/env python

import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')

outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp'

root='/scratch/leuven/317/vsc31786/output'
exp = 'SMAP_EASEv2_M09_SI_CLSM_SMOSfw_DA'
domain = 'SMAP_EASEv2_M09'

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

