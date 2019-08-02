#!/usr/bin/env python

from pyldas.interface import LDAS_io
import subprocess
import os
from bat_pyldas.functions import calc_tau_and_lag1_autocor
from netCDF4 import Dataset
import numpy as np
from bat_pyldas.functions import setup_grid_grid_for_plot
from shutil import copyfile
from shutil import move

# set names
root='/scratch/leuven/329/vsc32924/output/'
exp='INDONESIA_M09_v01_spinup2'
domain='SMAP_EASEv2_M09'

# processing
proc_ObsFcstAna = 0
proc_incr = 0
proc_daily = 1
proc_tau_and_lag1_autocor = 0


if proc_ObsFcstAna==1:
    io = LDAS_io('ObsFcstAna', exp, domain, root)
    # # io.bin2netcdf(overwrite=True)
    io.bin2netcdf()
    os.chdir(io.paths.ana+'/ens_avg')
    ntime = io.images.time.values.__len__()
    ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ObsFcstAna_images.nc ObsFcstAna_timeseries.nc" %  (ntime)
    subprocess.call(ncks_command, shell=True)

if proc_incr==1:
    io = LDAS_io('incr', exp, domain, root)
    #io.bin2netcdf(overwrite=True)
    io.bin2netcdf()
    os.chdir(io.paths.ana+'/ens_avg')
    ntime = io.images.time.values.__len__()
    ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 incr_images.nc incr_timeseries.nc" %  (ntime)
    subprocess.call(ncks_command, shell=True)

if proc_daily==1:
    io = LDAS_io('daily', exp, domain, root)
    io.bin2netcdf(overwrite=True)
    #io.bin2netcdf()
    ### add zbar
    os.makedirs(io.paths.root +'/' + exp + '/output_postprocessed/',exist_ok=True)
    fn = io.paths.root +'/' + exp + '/output_postprocessed/daily_zbar_images.nc'
    os.remove(fn) if os.path.exists(fn) else None
    copyfile(io.paths.cat+'/ens_avg'+'/daily_images.nc', fn)
    dataset = Dataset(io.paths.root +'/' + exp + '/output_postprocessed/daily_zbar_images.nc','r+')
    catparam = io.read_params('catparam')
    catdef = dataset.variables['catdef']
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
    zbar = -1.0*(np.sqrt(0.000001+catdef/bf1)-bf2)
    dataset.createVariable('zbar', 'f4', ('time', 'lat', 'lon'),chunksizes=(1, catdef.shape[1], catdef.shape[2]), fill_value=-9999.0)
    dataset.variables['zbar'][:] = zbar
    dataset.close()
    move(fn, io.paths.cat+'/ens_avg'+'/daily_images.nc')
    os.chdir(io.paths.cat+'/ens_avg')
    ntime = io.images.time.values.__len__()
    ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 daily_images.nc daily_timeseries.nc" %  (ntime)
    subprocess.call(ncks_command, shell=True)

if proc_tau_and_lag1_autocor==1:
    calc_tau_and_lag1_autocor(io)
