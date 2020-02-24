#!/usr/bin/env python

from pyldas.interface import LDAS_io
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

param='daily'
# processing
proc_daily = 1
proc_inst = 0
proc_ObsFcstAna = 1
proc_incr = 0
proc_ensstd = 0
proc_total_water = 1
proc_daily_stats = 1
proc_ensstd_stats = 0
proc_filter_diagnostics = 1
proc_filter_diagnostics_gs = 0
proc_filter_diagnostics_incr = 0
proc_scaling = 1
proc_tau_and_lag1_autocor = 0

#date_to='2010-08-01'
date_from='2010-01-01'
date_to='2019-10-31'
latmin=-90.
latmax=90.
lonmin=-180.
lonmax=180.
#latmin=53.
#latmax=71.
#lonmin=-160.
#lonmax=-100.

def main(argv):
    root='/scratch/leuven/317/vsc31786/output/'
    #root='/staging/leuven/stg_00024/OUTPUT/michelb/'
    #exp='SMOSrw_mwRTM_EASEv2_M09_CLSM_NORTH'
    #exp='GLOB_M36_7Thv_TWS_FOV0_M2'
    exp='SMAP_EASEv2_M09_SMOSfw'
    domain='SMAP_EASEv2_M09'
    #domain='SMAP_EASEv2_M36_GLOB'
    #vscgroup = os.getenv("HOME").split("/")[3]
    #vscname = os.getenv("HOME").split("/")[4]
    try:
        opts, args = getopt.getopt(argv,"hr:e:d:",["root=","experiment=","domain="])
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
            exp = arg
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
    
    if proc_inst==1:
        io = LDAS_io('inst', exp, domain, root)
        io.bin2netcdf(overwrite=True,date_from=date_from,date_to=date_to,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        #io.bin2netcdf()
        if 'catdef' in list(io.images.data_vars):
            ### add zbar
            os.makedirs(io.paths.root +'/' + exp + '/output_postprocessed/',exist_ok=True)
            fn = io.paths.root +'/' + exp + '/output_postprocessed/inst_zbar_images.nc'
            os.remove(fn) if os.path.exists(fn) else None
            copyfile(io.paths.cat+'/ens_avg'+'/inst_images.nc', fn)
            dataset = Dataset(io.paths.root +'/' + exp + '/output_postprocessed/inst_zbar_images.nc','r+')
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
            move(fn, io.paths.cat+'/ens_avg'+'/inst_images.nc')
        os.chdir(io.paths.cat+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 inst_images.nc inst_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)
    
    if proc_total_water==1:
        io = LDAS_io('daily', exp, domain, root)
        ### add total_water
        os.makedirs(io.paths.root +'/' + exp + '/output_postprocessed/',exist_ok=True)
        fn = io.paths.root +'/' + exp + '/output_postprocessed/daily_total_water_images.nc'
        os.remove(fn) if os.path.exists(fn) else None
        copyfile(io.paths.cat+'/ens_avg'+'/daily_images.nc', fn)
        dataset = Dataset(io.paths.root +'/' + exp + '/output_postprocessed/daily_total_water_images.nc','r+')
        catdef = dataset.variables['catdef']
        dataset.createVariable('total_water', 'f4', ('time', 'lat', 'lon'),chunksizes=(1, catdef.shape[1], catdef.shape[2]), fill_value=-9999.0)
        dataset.variables['total_water'][:] = dataset.variables['catdef'][:] + dataset.variables['rzexc'][:] + dataset.variables['srfexc'][:]
        dataset.close()
        move(io.paths.cat+'/ens_avg'+'/daily_images.nc',io.paths.root +'/' + exp + '/output_postprocessed/daily_total_water_images_old.nc')
        move(fn, io.paths.cat+'/ens_avg'+'/daily_images.nc')
        os.chdir(io.paths.cat+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 daily_images.nc daily_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)
    
    if proc_ObsFcstAna==1:
        io = LDAS_io('ObsFcstAna', exp, domain, root)
        io.bin2netcdf(overwrite=True,date_from=date_from,date_to=date_to,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        #io.bin2netcdf()
        os.chdir(io.paths.ana+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ObsFcstAna_images.nc ObsFcstAna_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)
    
    if proc_incr==1:
        io = LDAS_io('incr', exp, domain, root)
        io.bin2netcdf(overwrite=True,date_from=date_from,date_to=date_to,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        #io.bin2netcdf()
        os.chdir(io.paths.ana+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 incr_images.nc incr_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)
    
    if proc_daily_stats==1:
        io = LDAS_io(param, exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        daily_stats(exp, domain, root, outputpath,'std',param=param)
        daily_stats(exp, domain, root, outputpath,'mean',param=param)
    
    if proc_ensstd==1:
        io = LDAS_io('ensstd', exp, domain, root)
        #io.bin2netcdf(overwrite=True,date_from=date_from,date_to=date_to,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        #io.bin2netcdf()
        os.makedirs(io.paths.root +'/' + exp + '/output_postprocessed/',exist_ok=True)
        fn = io.paths.root +'/' + exp + '/output_postprocessed/ensstd_zbar_images.nc'
        os.remove(fn) if os.path.exists(fn) else None
        copyfile(io.paths.ana+'/ens_avg'+'/ensstd_images.nc', fn)
        dataset = Dataset(io.paths.root +'/' + exp + '/output_postprocessed/ensstd_zbar_images.nc','r+')
        catparam = io.read_params('catparam',latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
        catdef = dataset.variables['catdef']
        [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
        bf1 = np.full(lons.shape, np.nan)
        bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
        bf2 = np.full(lons.shape, np.nan)
        bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
        io_daily = LDAS_io('daily', exp, domain, root)
        catdef_daily = io_daily.images['catdef']
        zbar1 = -1.0*(np.sqrt(0.000001+(catdef_daily+0.5*io.images['catdef'])/bf1)-bf2)
        zbar2 = -1.0*(np.sqrt(0.000001+(catdef_daily)/bf1)-bf2)
        zbar = 2.*(zbar2-zbar1)
        dataset.createVariable('zbar', 'f4', ('time', 'lat', 'lon'),chunksizes=(1, catdef.shape[1], catdef.shape[2]), fill_value=-9999.0)
        dataset.variables['zbar'][:] = zbar
        dataset.close()
        move(fn, io.paths.ana+'/ens_avg'+'/ensstd_images.nc')
        os.chdir(io.paths.ana+'/ens_avg')
        ntime = io.images.time.values.__len__()
        ncks_command = "ncks -O -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ensstd_images.nc ensstd_timeseries.nc" %  (ntime)
        subprocess.call(ncks_command, shell=True)
    
    if proc_ensstd_stats==1:
        io = LDAS_io('ensstd', exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        #ensstd_stats(exp, domain, root, outputpath,'std')
        ensstd_stats(exp, domain, root, outputpath,'mean')
    
    if proc_filter_diagnostics==1:
        io = LDAS_io('ObsFcstAna', exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        filter_diagnostics_evaluation(exp, domain, root, outputpath)
    
    if proc_filter_diagnostics_gs==1:
        io = LDAS_io('ObsFcstAna', exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        filter_diagnostics_evaluation_gs(exp, domain, root, outputpath)
    
    if proc_filter_diagnostics_incr==1:
        io = LDAS_io('ObsFcstAna', exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        filter_diagnostics_evaluation_incr(exp, domain, root, outputpath)
    
    if proc_scaling==1:
        io = LDAS_io('ObsFcstAna', exp, domain, root)
        outputpath = io.paths.root +'/' + exp + '/output_postprocessed/'
        os.makedirs(outputpath,exist_ok=True)
        bin2nc_scaling(exp, domain, root, outputpath,
                       scalepath=os.path.join(root,exp,'output/SMAP_EASEv2_M09/stats/z_score_clim/pentad/'),
                       scalename='Thvf_TbSM_001_SMOS_zscore_stats_2010_p37_2019_p48_hscale_0.00_W_9p_Nmin_20')
    if proc_tau_and_lag1_autocor==1:
        calc_tau_and_lag1_autocor(io)

if __name__ == "__main__":
    main(sys.argv[1:])
