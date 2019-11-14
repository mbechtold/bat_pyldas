#!/usr/bin/env python

# batch script to convert ldas binary files to netcdf files
# and other processed output files
#
# proc_daily: converts daily binary files to netcdf files and adds zbar as variable
# --> daily_images.nc and daily_timeseries.nc
# proc_ObsFcstAna: converts ObsFcstAna binary files to netcdf files
# proc_incr: converts increments binary files to netcdf files
# proc_tau_and_lag1_autocor (experimental) --> for autocorrelation in time series
proc_daily = 1
proc_ObsFcstAna = 0
proc_incr = 0
proc_tau_and_lag1_autocor = 0
#### constrain processing time period and domain (by default experimental domain and time period of binary files)
date_from=None
date_to=None
latmin=-90.
latmax=90.
lonmin=-180.
lonmax=180.

##########################################
import subprocess
from pyldas.interface import LDAS_io
import os
import sys, getopt
from bat_pyldas.functions import *
from netCDF4 import Dataset
import numpy as np
from bat_pyldas.functions import setup_grid_grid_for_plot
from shutil import copyfile
from shutil import move
##########################################

def main(argv):
    root='/scratch/leuven/317/vsc31786/output/00TEST/'
    exp='CONGO_M09_PEATCLSMTN_v01'
    domain='SMAP_EASEv2_M09'
    #vscgroup = os.getenv("HOME").split("/")[3]
    #vscname = os.getenv("HOME").split("/")[4]
    try:
        opts, args = getopt.getopt(argv,"hr:e:d:v:",["root=","experiment=","domain="])
    except getopt.GetoptError:
        print('process_ldas_output_default.py -r <root> -e <outputfile> -d <domain>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('process_ldas_output_default.py -r <root> -e <experiment> -d <domain>')
            sys.exit()
        elif opt in ("-r", "--root"):
            root = arg
        elif opt in ("-e", "--experiment"):
            experiment = arg
        elif opt in ("-d", "--domain"):
            domain = arg

    if proc_daily==1:
        io = LDAS_io('daily', exp, domain, root)
        io.bin2netcdf(overwrite=True,date_from=date_from,date_to=date_to,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        #io.bin2netcdf()
        ### add zbar
        os.makedirs(io.paths.root +'/' + exp + '/output_postprocessed/',exist_ok=True)
        fn = io.paths.root +'/' + exp + '/output_postprocessed/daily_zbar_images.nc'
        os.remove(fn) if os.path.exists(fn) else None
        copyfile(io.paths.cat+'/ens_avg'+'/daily_images.nc', fn)
        dataset = Dataset(io.paths.root +'/' + exp + '/output_postprocessed/daily_zbar_images.nc','r+')
        catparam = io.read_params('catparam',latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        catdef = dataset.variables['catdef']
        [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
        bf1 = np.full(lons.shape, np.nan)
        bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
        bf2 = np.full(lons.shape, np.nan)
        bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
        ## latmin lonmin
        #domainlons = io.grid.ease_lons[np.min(io.grid.tilecoord.i_indg):(np.max(io.grid.tilecoord.i_indg)+1)]
        #domainlats = io.grid.ease_lats[np.min(io.grid.tilecoord.j_indg):(np.max(io.grid.tilecoord.j_indg)+1)]
        #bf1 = bf1[np.argmin(np.abs(domainlats-latmax)):(np.argmin(np.abs(domainlats-latmin))+1),np.argmin(np.abs(domainlons-lonmin)):(np.argmin(np.abs(domainlons-lonmax))+1)]
        #bf2 = bf2[np.argmin(np.abs(domainlats-latmax)):(np.argmin(np.abs(domainlats-latmin))+1),np.argmin(np.abs(domainlons-lonmin)):(np.argmin(np.abs(domainlons-lonmax))+1)]
        zbar = -1.0*(np.sqrt(0.000001+catdef/bf1)-bf2)
        dataset.createVariable('zbar', 'f4', ('time', 'lat', 'lon'),chunksizes=(1, catdef.shape[1], catdef.shape[2]), fill_value=-9999.0)
        dataset.variables['zbar'][:] = zbar
        dataset.close()
        move(fn, io.paths.cat+'/ens_avg'+'/daily_images.nc')
        os.chdir(io.paths.cat+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 daily_images.nc daily_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)

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

    if proc_tau_and_lag1_autocor==1:
        calc_tau_and_lag1_autocor(io)

if __name__ == "__main__":
    main(sys.argv[1:])
