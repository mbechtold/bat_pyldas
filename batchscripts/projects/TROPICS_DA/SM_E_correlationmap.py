#!/usr/bin/env python

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *
from bat_pyldas.functions import *
import getpass
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from netCDF4 import Dataset
from pyldas.interface import LDAS_io


#this is a function that creates netcdf file with right dimensions
def ncfile_init(fname, lats, lons, species, tags):

    #create the ncfile which is writable in netcdf4 format
    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat','lon','species']
    dimvals = [lats, lons, species]
    chunksizes = [len(lats), len(lons), 1]
    dtypes = ['float32','float32','uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        if tag.find('pearsonR') != -1:
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[0:-1], chunksizes=chunksizes[0:-1], fill_value=-9999., zlib=True)

    return ds



#generate the drained nc correlation file
def filter_diagnostics_evaluation():

    # read in the data (LSM and Observation)
    root = '/staging/leuven/stg_00024/OUTPUT/michelb/TROPICS'
    outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/TROPICS/sm_sensitivity_test'
    exp = 'INDONESIA_M09_PEATCLSMTD_v01_SMOSfw'
    domain = 'SMAP_EASEv2_M09'

    lsm = LDAS_io('inst', exp=exp, domain=domain, root=root)
    ObsFcstAna = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)

    #generated nc file stored in working directory under this name
    result_file = r'SM_E_correlationmap_drained.nc'

    tags = ['pearsonR']

    lons = np.unique(lsm.grid.tilecoord['com_lon'].values)
    lats = np.unique(lsm.grid.tilecoord['com_lat'].values)[::-1]

    species = ObsFcstAna.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    for lats
        for lons
            if poros>0.6

                for i_spc,spc in enumerate(species):

                ds['pearsonR'][:,:,i_spc] = pearsonr([0]['obs_obs'][i_spc] - [0]['obs_fcst'][i_spc]).mean(dim='time').values

                tmp = [0]['obs_obs'][i_spc].values
                np.place(tmp, ~np.isnan(tmp), 1.)
                np.place(tmp, np.isnan(tmp), 0.)
                ds['pearsonR'][:, :, i_spc] = tmp.sum(axis=2)

            ds.close()

if __name__ == '__main__':
    filter_diagnostics_evaluation()
