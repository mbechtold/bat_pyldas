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
    result_file = '/scratch/leuven/324/vsc32460/output/TROPICS/SM_E_correlationmap_drained.nc'

    tags = ['pearsonR']

    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(lsm)
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    tc = lsm.grid.tilecoord
    tg = lsm.grid.tilegrids
    poros[tc.j_indg.values, tc.i_indg.values] = params['poros'].values

    #lons1D = np.unique(lsm.grid.tilecoord['com_lon'].values)
    #lats1D = np.unique(lsm.grid.tilecoord['com_lat'].values)[::-1]

    species = ObsFcstAna.timeseries['species'].values

    ds = ncfile_init(result_file, lats[:,0], lons[0,:], species, tags)

    #[col, row] = get_M09_ObsFcstAna(ObsFcstAna,lon,lat)
    #[col, row] = get_M09_ObsFcstAna(ObsFcstAna,lons.min()+2,lats.min()+2)
    #col = col%4

    for row in range(poros.shape[0]):
        print("row: " + str(row))
        for col in range(poros.shape[1]):
            if poros[row,col]>0.6:
                for i_spc,spc in enumerate(species):
                    [col_obs, row_obs] = get_M09_ObsFcstAna(ObsFcstAna,col,row,lonlat=False)
                    ts_sfmc = lsm.read_ts('sfmc', col, row, lonlat=False)
                    ts_tp1 = lsm.read_ts('tp1', col, row, lonlat=False)
                    ts_obsobs = ObsFcstAna.read_ts('obs_obs', col_obs, row_obs, species=i_spc+1, lonlat=False)
                    ts_obsobs.name = 'obsobs'

                    df = pd.concat((ts_sfmc,ts_tp1,ts_obsobs),axis=1)
                    ts_emissivity = df['obsobs']/df['tp1']

                    df = pd.concat((ts_sfmc,ts_emissivity),axis=1)
                    if not np.isnan(df.corr().values[0,1]):
                        ds.variables['pearsonR'][row,col,i_spc] = df.corr().values[0,1]

                    #pearsonr([0]['obs_obs'][i_spc] - [0]['obs_fcst'][i_spc]).mean(dim='time').values
                    #tmp = [0]['obs_obs'][i_spc].values
                    #np.place(tmp, ~np.isnan(tmp), 1.)
                    #np.place(tmp, np.isnan(tmp), 0.)
                    #ds['pearsonR'][:, :, i_spc] = tmp.sum(axis=2)

    #plt.imshow(-1.0*ds['pearsonR'][:,:,2])
    #data = np.full(lons.shape, np.nan)
    #data[tc.j_indg.values, tc.i_indg.values] =
    for i_spc,spc in enumerate(species):
        data = np.ma.masked_invalid(ds['pearsonR'][:,:,i_spc])
        #tmp_data = obs_M09_to_M36(data)
        fname = 'R_eSM_sp'+str(i_spc)
        figpath='/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Drained/DA_sensitivity'
        #drained zoom
        latmin = -1.5
        latmax = 0.5
        lonmin = 102.2
        lonmax = 104.2

        #natural zoom
        #latmin = -3.9
        #latmax = -1.9
        #lonmin = 113.1
        #lonmax = 115.1
        cmin=-0.85
        cmax=-0.25
        [data_zoom, lons_zoom, lats_zoom, llcrnrlat_zoom, urcrnrlat_zoom, llcrnrlon_zoom, urcrnrlon_zoom] =figure_zoom(data, lons, lats, latmin, latmax, lonmin, lonmax)
        figure_single_default_zoom(data=data_zoom, lons=lons_zoom, lats=lats_zoom, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat_zoom, urcrnrlat=urcrnrlat_zoom,
                                  llcrnrlon=llcrnrlon_zoom, urcrnrlon=urcrnrlon_zoom, outpath=figpath, exp=exp, fname=fname + '_zoom_' +figpath[52:59], plot_title='R (-), ' + fname + ' ,zoom ,' + figpath[52:59], cmap='jet')

    ds.close()

if __name__ == '__main__':
    filter_diagnostics_evaluation()
