

import os

import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta

from pyldas.grids import EASE2
from scipy.integrate import quad
from scipy import stats

from pyldas.interface import LDAS_io
import scipy.optimize as optimization
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from collections import OrderedDict

from pyldas.interface import LDAS_io

from netCDF4 import Dataset

def read_wtd_data(insitu_path, exp, domain, root):

    master_table = pd.read_csv('/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Mastertable/WTD_TROPICS_MASTER_TABLE.csv',sep=';')
    blacklist = pd.read_csv('/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data/Blacklist/Blacklisted_stations.csv', sep=';')

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    first_site = True

    for i,site_ID in enumerate(master_table.iloc[:,0]):

        if not site_ID.startswith('IN') and not site_ID.startswith('BR'):
            # Only use sites in Indonesia and Brunei.
            continue

        if blacklist.iloc[:, 0].str.contains(site_ID).any() or master_table.iloc[i,4] == 0:
            # If site is on the blacklist (contains bad data), or if "comparison" =  0, then don include that site in the dataframe.
            continue

        # Get lat lon from master table for site.
        lon = master_table.iloc[i,1]
        lat = master_table.iloc[i,2]

        # Get porosity for site lon lat.
        # Get M09 rowcol with data.
        col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
        # Get poros for col row.
        siteporos = poros[row, col]


        if siteporos <= 0.7:
            # If the porosity of the site is 0.7 or lower the site is not classified as peatland in the model.
            continue

        if site_ID.startswith('IN'):
            folder_wtd = insitu_path + '/Sipalaga/processed/WTD/Daily/'
            folder_p = insitu_path + '/Sipalaga/processed/Precipitation/Daily/'
            site_precip = site_ID
        elif site_ID.startswith('BR'):
            folder_wtd = insitu_path + '/Brunei/processed/WTD/Daily/'
            folder_p = insitu_path + '/Brunei/processed/Precipitation/Daily/'
            site_precip = 'Brunei_Darussalam'   # For Brunei the throughfall data are the average of data from four throughfall
            # gauges along a 100 m transect. And Therefore itś the same for all four stations.

        if first_site == True:
            # Load in situ data.
            # wtd:
            wtd_obs = pd.read_csv(folder_wtd + site_ID + '.csv')
            wtd_obs.rename({'date': 'time', 'wtd': site_ID}, axis=1, inplace=True)
            wtd_obs['time'] = pd.to_datetime(wtd_obs['time'])
            wtd_obs = wtd_obs.set_index('time')
            # Precipitation:
            precip_obs = pd.read_csv(folder_p + site_precip + '.csv')
            precip_obs.rename({'date': 'time', 'precipitation': site_ID}, axis=1, inplace=True)
            precip_obs['time'] = pd.to_datetime(precip_obs['time'])
            precip_obs = precip_obs.set_index('time')

            # Load model wtd data.
            wtd_mod = io.read_ts('zbar', lon, lat, lonlat=True)


            # Check if overlapping data.
            df_check = pd.concat((wtd_obs, wtd_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False
        else:
            # Load in situ data.
            # wtd:
            wtd_obs_tmp = pd.read_csv(folder_wtd + site_ID + '.csv')
            wtd_obs_tmp.rename({'date': 'time', 'wtd': site_ID}, axis=1, inplace=True)
            wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'])
            wtd_obs_tmp = wtd_obs_tmp.set_index('time')
            # Precipitation:
            precip_obs_tmp = pd.read_csv(folder_p + site_precip + '.csv')
            precip_obs_tmp.rename({'date': 'time', 'precipitation': site_ID}, axis=1, inplace=True)
            precip_obs_tmp['time'] = pd.to_datetime(precip_obs_tmp['time'])
            precip_obs_tmp = precip_obs_tmp.set_index('time')


            # Load model data.
            wtd_mod_tmp = io.read_ts('zbar', lon, lat, lonlat=True)


            # Check if overlaping data.
            df_check = pd.concat((wtd_obs_tmp, wtd_mod_tmp), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                wtd_obs = pd.concat((wtd_obs, wtd_obs_tmp), axis=1)
                wtd_mod = pd.concat((wtd_mod, wtd_mod_tmp), axis=1)
                precip_obs = pd.concat((precip_obs, precip_obs_tmp), axis=1)


    wtd_mod.columns = wtd_obs.columns

    return wtd_obs, wtd_mod, precip_obs


def ncfile_init(fname, lats, lons, species, tags):

    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['species','lat','lon']
    dimvals = [species, lats, lons]
    chunksizes = [1, len(lats), len(lons)]
    dtypes = ['uint8', 'float32','float32']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=(dim,), chunksizes=(chunksize,), zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        if (tag.find('innov') != -1) | (tag.find('obsvar') != -1) | (tag.find('fcstvar') != -1):
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[1:], chunksizes=chunksizes[1:], fill_value=-9999., zlib=True)

    ds.variables['lon'].setncatts({'long_name': 'longitude', 'units':'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude', 'units':'degrees_north'})

    return ds

def ncfile_init_incr(fname, dimensions, tags):

    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat','lon']
    dimvals = [dimensions['lat'], dimensions['lon']]
    chunksizes = [len(dimvals[0]), len(dimvals[1])]
    dtypes = ['float32','float32']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=(dim,), chunksizes=(chunksize,), zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)

    ds.variables['lon'].setncatts({'long_name': 'longitude', 'units':'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude', 'units':'degrees_north'})

    return ds


def ncfile_init_test(fname, dimensions, variables):
    """"
    Method to initialize dimensions/variables of a image-chunked netCDF file

    Parameters
    ----------
    fname : str
        Filename of the netCDF file to be created
    dimensions : dict
        Dictionary containing the dimension names and values
    variables : list
        list of variables to be created with the specified dimensions

    Returns
    -------
    ds : filencfile_initid
        File ID of the created netCDF file

    """

    ds = Dataset(fname, mode='w')
    timeunit = 'hours since 2000-01-01 00:00'

    # Initialize dimensions
    chunksizes = []
    for dim in dimensions:

        # convert pandas Datetime Index to netCDF-understandable numeric format
        if dim == 'time':
            dimensions[dim] = date2num(dimensions[dim].to_pydatetime(), timeunit).astype('int32')

        # Files are per default image chunked
        if dim in ['lon','lat']:
            chunksize = len(dimensions[dim])
        else:
            chunksize = 1
        chunksizes.append(chunksize)

        dtype = dimensions[dim].dtype
        ds.createDimension(dim, len(dimensions[dim]))
        ds.createVariable(dim,dtype,
                          dimensions=(dim,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds.variables[dim][:] = dimensions[dim]

    # Coordinate attributes following CF-conventions
    if "time" in ds.variables:
        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})
    ds.variables['lon'].setncatts({'long_name': 'longitude',
                                   'units':'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude',
                                    'units':'degrees_north'})

    # Initialize variables
    for var in variables:
        ds.createVariable(var, 'float32',
                          dimensions=list(dimensions.keys()),
                          chunksizes=chunksizes,
                          fill_value=-9999.,
                          zlib=True)

    return ds

def ncfile_init_scaling(fname, lats, lons, pentad, AD, tags):

    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat','lon','pentad','AD']
    dimvals = [lats, lons, pentad, AD]
    chunksizes = [len(lats), len(lons),1 ,1]
    dtypes = ['float32','float32','uint8','uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim] = dimval

    for tag in tags:
        ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)

    return ds

def ncfile_init_multiruns(fname, lats, lons, runs, species, tags):

    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat','lon','run','species']
    dimvals = [lats, lons, runs, species]
    chunksizes = [len(lats), len(lons), 1, 1]
    dtypes = ['float32','float32','uint8', 'uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim] = dimval

    for tag in tags:
        if tag.find('innov') != -1:
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[0:-1], chunksizes=chunksizes[0:-1], fill_value=-9999., zlib=True)

    return ds

def bin2nc_scaling(exp, domain, root, outputpath):

    angle = 40
    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'scaling.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('ObsFcstAna',exp,domain,root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    pentad = np.arange(1,74)
    AD = np.arange(0,2)
    scalepath = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalepath'].split()[0]
    scalename = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalename'][5:].split()[0]
    print('scalepath')
    print(scalepath)
    print(scalename)
    tags = ['m_mod_H_%2i'%angle,'m_mod_V_%2i'%angle,'m_obs_H_%2i'%angle,'m_obs_V_%2i'%angle]

    ds = ncfile_init_scaling(result_file, lats[:,0], lons[0,:], pentad, AD, tags)

    for i_AscDes,AscDes in enumerate(list(["A","D"])):
        for i_pentad in np.arange(1,74):

            logging.info('pentad %i' % (i_pentad))
            fname = scalepath+scalename+'_'+AscDes+'_p'+"%02d" % (i_pentad,)+'.bin'
            tmp = io.read_scaling_parameters(fname=fname)

            res = tmp[['lon','lat','m_mod_H_%2i'%angle,'m_mod_V_%2i'%angle,'m_obs_H_%2i'%angle,'m_obs_V_%2i'%angle]]
            res.replace(-9999.,np.nan,inplace=True)

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_H_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_mod_H_%2i'%angle][:,:,i_pentad-1,i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_V_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_mod_V_%2i'%angle][:,:,i_pentad-1,i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_H_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_obs_H_%2i'%angle][:,:,i_pentad-1,i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_V_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_obs_V_%2i'%angle][:,:,i_pentad-1,i_AscDes] = data

    ds.close()

def filter_diagnostics_evaluation(exp, domain, root, outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna',exp, domain, root)
    #cvars = DA_innov.timeseries.data_vars
    #cond = DA_innov.timeseries['obs_assim']==-1
    #tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    #np.place(tmp, np.isnan(tmp), 1.)
    #np.place(tmp, tmp==False, 0.)
    #np.place(tmp, tmp==-1, 1.)
    #DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    #cvars = ['obs_obs']
    #for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    #del cond1
    DA_incr = LDAS_io('incr',exp, domain, root)
    
    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily',exp, domain, root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['innov_mean','innov_var',
            'norm_innov_mean','norm_innov_var',
            'obsvar_mean','fcstvar_mean',
            'n_valid_innov',
            'incr_catdef_mean','incr_catdef_var',
            'incr_rzexc_mean','incr_rzexc_var',
            'incr_srfexc_mean','incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][:,:,:].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    mask_scaling_old = 0
    if mask_scaling_old == 1:
        mask_innov = np.full_like(DA_innov.timeseries['obs_obs'][:,0,:,:].values, False)
        mask_incr = np.full_like(DA_incr.timeseries['catdef'][:,:,:].values, False)
        n_innov_nyr = np.zeros([DA_innov.timeseries.dims['lat'],DA_innov.timeseries.dims['lon']])
        for j,yr in enumerate(np.unique(pd.Series(DA_innov.timeseries['time']).dt.year)):
            n_innov_yr = np.zeros([DA_innov.timeseries.dims['lat'],DA_innov.timeseries.dims['lon']])
            for i_spc,spc in enumerate(species):
                logging.info('species %i' % (i_spc))
                obs_obs_tmp = DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values
                #np.place(obs_obs_tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
                tmp = obs_obs_tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:].values
                np.place(tmp, tmp==0., np.nan)
                np.place(tmp, tmp==-9999., np.nan)
                np.place(tmp, ~np.isnan(tmp), 1.)
                n_innov_tmp = np.nansum(tmp[pd.Series(DA_innov.timeseries['time']).dt.year==yr,:,:],axis=0)
                np.place(n_innov_tmp, n_innov_tmp==0., np.nan)
                n_innov_tmp = obs_M09_to_M36(n_innov_tmp)
                np.place(n_innov_tmp, np.isnan(n_innov_tmp), 0.)
                n_innov_yr = n_innov_yr + n_innov_tmp
            n_innov_yr = n_innov_yr/2.0  # H-pol and V-pol count as one obs
            cond = (n_innov_yr<3) & (n_innov_yr>0)
            n_innov_nyr[n_innov_yr>=3] = n_innov_nyr[n_innov_yr>=3] + 1
            cmask = np.repeat(cond[np.newaxis, :, :], np.where((pd.Series(DA_innov.timeseries['time']).dt.year==yr))[0].__len__(), axis=0)
            mask_innov[pd.Series(DA_innov.timeseries['time']).dt.year==yr,:,:] = cmask
            cmask = np.repeat(cond[np.newaxis, :, :], np.where((pd.Series(DA_incr.timeseries['time']).dt.year==yr))[0].__len__(), axis=0)
            mask_incr[pd.Series(DA_incr.timeseries['time']).dt.year==yr,:,:] = cmask
        del cmask,n_innov_yr,cond,tmp,obs_obs_tmp
        cond2 = n_innov_nyr <= 1
        cmask = np.repeat(cond2[np.newaxis, :, :], np.where((pd.Series(DA_innov.timeseries['time']).dt.year!=-9999))[0].__len__(), axis=0)
        np.place(mask_innov, cmask, True)
        cmask = np.repeat(cond2[np.newaxis, :, :], np.where((pd.Series(DA_incr.timeseries['time']).dt.year!=-9999))[0].__len__(), axis=0)
        np.place(mask_incr, cmask, True)

    for i_spc,spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        tmp = DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values
        np.place(tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
        if mask_scaling_old == 1:
            np.place(tmp, mask_innov,np.nan)
        ds['obsvar_mean'][i_spc,:,:] = (DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['fcstvar_mean'][i_spc,:,:] = (DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['innov_mean'][i_spc,:,:] = (tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['innov_var'][i_spc,:,:] = (tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).var(dim='time',skipna=True).values
        ds['norm_innov_mean'][i_spc,:,:] = ((tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
                                                  np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).mean(dim='time',skipna=True).values
        ds['norm_innov_var'][i_spc,:,:] = ((tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
                                                 np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).var(dim='time',skipna=True).values

        np.place(tmp, tmp==0., np.nan)
        np.place(tmp, tmp==-9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc,:, :] = tmp.sum(axis=0)
        del tmp

    if mask_scaling_old == 1:
        del mask_innov
    np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
    if mask_scaling_old == 1:
        np.place(DA_incr.timeseries['catdef'].values, mask_incr, np.nan)
        np.place(DA_incr.timeseries['rzexc'].values, mask_incr, np.nan)
        np.place(DA_incr.timeseries['srfexc'].values, mask_incr, np.nan)
    # map from daily to hourly
    #cmap = np.floor(np.linspace(0,io.timeseries['time'].size,incr.size+1))
    ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time',skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time',skipna=True).values
    ndays = io.timeseries['time'].size
    n3hourly = DA_incr.timeseries['time'].size
    # overwrite if exp does not contain CLSM
    if exp.find("CLSM")<0:
        for row in range(DA_incr.timeseries['catdef'].shape[1]):
            logging.info('PCLSM row %i' % (row))
            for col in range(DA_incr.timeseries['catdef'].shape[2]):
                if poros[row,col]>0.7:
                    catdef1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'), data=io.timeseries['catdef'][:,row,col].values)
                    catdef1 = catdef1.resample('3H').max().interpolate()
                    ar1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'), data=io.timeseries['ar1'][:,row,col].values)
                    ar1 = ar1.resample('3H').max().interpolate()
                    incr = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,row,col].values)
                    #incr = DA_incr.timeseries['catdef'][:,row,col]
                    df = pd.concat((catdef1,incr,ar1),axis=1)
                    df.columns = ['catdef1','incr','ar1']
                    catdef2 = df['catdef1']+df['incr']
                    catdef1 = df['catdef1']
                    ar1 = df['ar1']
                    incr = df['incr']
                    zbar1 = -1.0*(np.sqrt(0.000001+catdef1/bf1[row,col])-bf2[row,col])
                    zbar2 = -1.0*(np.sqrt(0.000001+catdef2/bf1[row,col])-bf2[row,col])
                    incr_ar1 = -(zbar2-zbar1)*1000*ar1   # incr surface water storage but expressed as deficit change, so '-'
                    #data2 = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,0,0].values)
                    #cdata = pd.concat(data1,data2)
                    #zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                    #incr = DA_incr.timeseries['catdef'][:,row,col]
                    #catdef = np.full_like(incr,np.nan)
                    #for i in range(n3hourly):
                    #    catdef1 = io.timeseries['catdef'][int(np.floor(i/8.0)),row,col]
                    #    catdef2 = io.timeseries['catdef'][int(np.floor(i/8.0)),row,col] + incr[i].item()
                    #    zbar1 = -1.0*(np.sqrt(0.000001+catdef1/bf1[row,col])-bf2[row,col])
                    #    zbar2 = -1.0*(np.sqrt(0.000001+catdef2/bf1[row,col])-bf2[row,col])
                    #    if np.isnan(zbar1-zbar2)==False:
                    #        res, err = quad(normal_distribution_function, zbar1, zbar2)
                    #        catdef[i] = incr[i].item() + res
                    #    else:
                    #        catdef[i] = 0.0
                    incr_total = incr_ar1 + incr
                    ds['incr_catdef_mean'][row, col] = incr_total.mean(skipna=True)
                    ds['incr_catdef_var'][row, col] = incr_total.var(skipna=True)
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time',skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time',skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time',skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time',skipna=True).values

    ds.close()

def filter_diagnostics_evaluation_incr(exp, domain, root, outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna',exp, domain, root)
    #cvars = DA_innov.timeseries.data_vars
    #cond = DA_innov.timeseries['obs_assim']==-1
    #tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    #np.place(tmp, np.isnan(tmp), 1.)
    #np.place(tmp, tmp==False, 0.)
    #np.place(tmp, tmp==-1, 1.)
    #DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    #cvars = ['obs_obs']
    #for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    #del cond1
    DA_incr = LDAS_io('incr',exp, domain, root)

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily',exp, domain, root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['innov_mean','innov_var',
            'norm_innov_mean','norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean','incr_catdef_var',
            'incr_rzexc_mean','incr_rzexc_var',
            'incr_srfexc_mean','incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][:,:,:].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    mask_innov = np.full_like(DA_innov.timeseries['obs_obs'][:,0,:,:].values, False)
    mask_incr = np.full_like(DA_incr.timeseries['catdef'][:,:,:].values, False)
    mask_scaling_old = 0
    if mask_scaling_old == 1:
        n_innov_nyr = np.zeros([DA_innov.timeseries.dims['lat'],DA_innov.timeseries.dims['lon']])
        for j,yr in enumerate(np.unique(pd.Series(DA_innov.timeseries['time']).dt.year)):
            n_innov_yr = np.zeros([DA_innov.timeseries.dims['lat'],DA_innov.timeseries.dims['lon']])
            for i_spc,spc in enumerate(species):
                logging.info('species %i' % (i_spc))
                obs_obs_tmp = DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values
                np.place(obs_obs_tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
                tmp = obs_obs_tmp - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:].values
                np.place(tmp, tmp==0., np.nan)
                np.place(tmp, tmp==-9999., np.nan)
                np.place(tmp, ~np.isnan(tmp), 1.)
                n_innov_tmp = np.nansum(tmp[pd.Series(DA_innov.timeseries['time']).dt.year==yr,:,:],axis=0)
                np.place(n_innov_tmp, n_innov_tmp==0., np.nan)
                n_innov_tmp = obs_M09_to_M36(n_innov_tmp)
                np.place(n_innov_tmp, np.isnan(n_innov_tmp), 0.)
                n_innov_yr = n_innov_yr + n_innov_tmp
            n_innov_yr = n_innov_yr/2.0  # H-pol and V-pol count as one obs
            cond = (n_innov_yr<3) & (n_innov_yr>0)
            n_innov_nyr[n_innov_yr>=3] = n_innov_nyr[n_innov_yr>=3] + 1
            cmask = np.repeat(cond[np.newaxis, :, :], np.where((pd.Series(DA_innov.timeseries['time']).dt.year==yr))[0].__len__(), axis=0)
            mask_innov[pd.Series(DA_innov.timeseries['time']).dt.year==yr,:,:] = cmask
            cmask = np.repeat(cond[np.newaxis, :, :], np.where((pd.Series(DA_incr.timeseries['time']).dt.year==yr))[0].__len__(), axis=0)
            mask_incr[pd.Series(DA_incr.timeseries['time']).dt.year==yr,:,:] = cmask
        del cmask,n_innov_yr,cond,tmp,obs_obs_tmp
        cond2 = n_innov_nyr <= 1
        cmask = np.repeat(cond2[np.newaxis, :, :], np.where((pd.Series(DA_innov.timeseries['time']).dt.year!=-9999))[0].__len__(), axis=0)
        np.place(mask_innov, cmask, True)
        cmask = np.repeat(cond2[np.newaxis, :, :], np.where((pd.Series(DA_incr.timeseries['time']).dt.year!=-9999))[0].__len__(), axis=0)
        np.place(mask_incr, cmask, True)

    for i_spc,spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values, mask_innov,np.nan)
        #ds['innov_mean'][i_spc,:,:] = (DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['innov_var'][i_spc,:,:] = (DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).var(dim='time',skipna=True).values
        #ds['norm_innov_mean'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
        #                                          np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).mean(dim='time',skipna=True).values
        #ds['norm_innov_var'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
        #                                         np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).var(dim='time',skipna=True).values

        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values, DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values==0., np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values, DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values==-9999., np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values, ~np.isnan(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values), 1.)
        np.place(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values, np.isnan(DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values), 0.)
        ds['n_valid_innov'][i_spc,:, :] = DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values.sum(axis=0)

    del mask_innov
    np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
    np.place(DA_incr.timeseries['catdef'].values, mask_incr, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, mask_incr, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, mask_incr, np.nan)
    # map from daily to hourly
    #cmap = np.floor(np.linspace(0,io.timeseries['time'].size,incr.size+1))
    #ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time',skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time',skipna=True).values
    ndays = io.timeseries['time'].size
    n3hourly = DA_incr.timeseries['time'].size
    # overwrite if exp does not contain CLSM
    if exp.find("CLSM")<0:
        for row in range(DA_incr.timeseries['catdef'].shape[1]):
            logging.info('PCLSM row %i' % (row))
            for col in range(DA_incr.timeseries['catdef'].shape[2]):
                if poros[row,col]>0.7:
                    catdef1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'), data=io.timeseries['catdef'][:,row,col].values)
                    catdef1 = catdef1.resample('3H').max().interpolate()
                    ar1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'), data=io.timeseries['ar1'][:,row,col].values)
                    ar1 = ar1.resample('3H').max().interpolate()
                    incr = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,row,col].values)
                    #incr = DA_incr.timeseries['catdef'][:,row,col]
                    df = pd.concat((catdef1,incr,ar1),axis=1)
                    df.columns = ['catdef1','incr','ar1']
                    catdef2 = df['catdef1']+df['incr']
                    catdef1 = df['catdef1']
                    ar1 = df['ar1']
                    incr = df['incr']
                    zbar1 = -1.0*(np.sqrt(0.000001+catdef1/bf1[row,col])-bf2[row,col])
                    zbar2 = -1.0*(np.sqrt(0.000001+catdef2/bf1[row,col])-bf2[row,col])
                    incr_ar1 = -(zbar2-zbar1)*1000*ar1   # incr surface water storage but expressed as deficit change, so '-'
                    #data2 = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,0,0].values)
                    #cdata = pd.concat(data1,data2)
                    #zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                    #incr = DA_incr.timeseries['catdef'][:,row,col]
                    #catdef = np.full_like(incr,np.nan)
                    #for i in range(n3hourly):
                    #    catdef1 = io.timeseries['catdef'][int(np.floor(i/8.0)),row,col]
                    #    catdef2 = io.timeseries['catdef'][int(np.floor(i/8.0)),row,col] + incr[i].item()
                    #    zbar1 = -1.0*(np.sqrt(0.000001+catdef1/bf1[row,col])-bf2[row,col])
                    #    zbar2 = -1.0*(np.sqrt(0.000001+catdef2/bf1[row,col])-bf2[row,col])
                    #    if np.isnan(zbar1-zbar2)==False:
                    #        res, err = quad(normal_distribution_function, zbar1, zbar2)
                    #        catdef[i] = incr[i].item() + res
                    #    else:
                    #        catdef[i] = 0.0
                    incr_total = incr_ar1 + incr
                    ds['incr_catdef_mean'][row, col] = incr_total.mean(skipna=True)
                    ds['incr_catdef_var'][row, col] = incr_total.var(skipna=True)
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time',skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time',skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time',skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time',skipna=True).values

    ds.close()


def daily_stats(exp, domain, root, outputpath, stat):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'daily_'+stat+'.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('daily',exp, domain, root)
    cvars = io.timeseries.data_vars

    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    dimensions = OrderedDict([('lat',lats), ('lon',lons)])

    #cvars = ['total_water']
    ds = ncfile_init_incr(result_file, dimensions, cvars)
    for i,cvar in enumerate(cvars):
        logging.info('cvar %s' % (cvar))
        if stat=='mean':
            ds[cvar][:,:] = io.timeseries[cvar][:,:,:].mean(dim='time',skipna=True).values
        elif stat=='std':
            ds[cvar][:,:] = io.timeseries[cvar][:,:,:].std(dim='time',skipna=True).values

    x,y = get_cdf_integral_xy()
    if exp.find("CLSM")<0:
        for row in range(io.timeseries['catdef'].shape[1]):
            for col in range(io.timeseries['catdef'].shape[2]):
                if poros[row,col]>0.7:
                    S_ar1 = np.interp(io.timeseries['zbar'][:,row,col].values,x,y)
                    catdef = io.timeseries['catdef'][:,row,col].values
                    catdef_total = -S_ar1*1000.0 + catdef
                    total_water = catdef_total + io.timeseries['rzexc'][:,row,col].values + io.timeseries['srfexc'][:,row,col].values
                    if stat=='mean':
                        ds['catdef'][row, col] = np.nanmean(catdef_total)
                        ds['total_water'][row, col] = np.nanmean(total_water)
                    elif stat=='std':
                        ds['catdef'][row, col] = np.nanstd(catdef_total)
                        ds['total_water'][row, col] = np.nanstd(total_water)

    ds.close()

def ensstd_stats(exp, domain, root, outputpath, stat):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'ensstd_'+stat+'.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('ensstd',exp, domain, root)
    cvars = io.timeseries.data_vars

    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    dimensions = OrderedDict([('lat',lats), ('lon',lons)])

    cvars = ['sfmc','rzmc','catdef','total_water','tp1','tsurf']
    #cvars = ['catdef','total_water']
    cvars = ['zbar']
    ds = ncfile_init_incr(result_file, dimensions, cvars)
    for i,cvar in enumerate(cvars):
        logging.info('cvar %s' % (cvar))
        if cvar=='total_water':
            if stat=='mean':
                ds[cvar][:,:] = (io.timeseries['catdef'][:,:,:]+io.timeseries['srfexc'][:,:,:]+io.timeseries['rzexc'][:,:,:]).mean(dim='time',skipna=True).values
            elif stat=='std':
                ds[cvar][:,:] = (io.timeseries['catdef'][:,:,:]+io.timeseries['srfexc'][:,:,:]+io.timeseries['rzexc'][:,:,:]).std(dim='time',skipna=True).values
        else:
            if stat=='mean':
                ds[cvar][:,:] = io.timeseries[cvar][:,:,:].mean(dim='time',skipna=True).values
            elif stat=='std':
                ds[cvar][:,:] = io.timeseries[cvar][:,:,:].std(dim='time',skipna=True).values

    x,y = get_cdf_integral_xy()
    if exp.find("CLSM")<0:
        for row in range(io.timeseries['catdef'].shape[1]):
            for col in range(io.timeseries['catdef'].shape[2]):
                if poros[row,col]>0.7:
                    catdef = io.timeseries['catdef'][:,row,col].values
                    zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                    S_ar1 = np.interp(zbar,x,y)
                    catdef_total = -S_ar1*1000.0 + catdef
                    total_water = catdef_total + io.timeseries['rzexc'][:,row,col].values + io.timeseries['srfexc'][:,row,col].values
                    if stat=='mean':
                        ds['catdef'][row, col] = np.nanmean(catdef_total)
                        ds['total_water'][row, col] = np.nanmean(total_water)
                    elif stat=='std':
                        ds['catdef'][row, col] = np.nanstd(catdef_total)
                        ds['total_water'][row, col] = np.nanstd(total_water)

    ds.close()


def get_M09_ObsFcstAna(io,lon,lat):
    # get M09 row col with data from 4x4 square
    col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
    dcols = [-1,0,1,2]
    drows = [-1,0,1,2]
    cfind=False
    for icol,dcol in enumerate(dcols):
        for icol,drow in enumerate(drows):
            ts_obs = io.read_ts('obs_obs', col+dcol, row+drow, species=2, lonlat=False)
            if np.isnan(ts_obs.max(skipna=True))==False:
                cfind = True
                break
        if cfind==True:
            break
    col = col+dcol
    row = row+drow
    return col,row

def normal_distribution_function(x):
    #x1 = -1
    #x2 = 0
    #res, err = quad(normal_distribution_function, x1, x2)
    #print('Normal Distribution (mean,std):',cmean,cstd)
    #print('Integration bewteen {} and {} --> '.format(x1,x2),res)
    #cmean = 0.0
    #cstd = 0.11  # PEAT-CLSM value
    value = stats.norm.cdf(x, 0.0, 0.11)
    return value

def get_cdf_integral_xy():
    x = np.linspace(-2.0,1.0,1000)
    y = np.full_like(x,np.nan)
    for i,cx in enumerate(x):
        res, err = quad(normal_distribution_function, -2, x[i])
        y[i] = res
    return x,y

def filter_diagnostics_evaluation_old(exp, domain, root, outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath,exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna',exp, domain, root)
    #cvars = DA_innov.timeseries.data_vars
    #cond = DA_innov.timeseries['obs_assim']==-1
    #tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    #np.place(tmp, np.isnan(tmp), 1.)
    #np.place(tmp, tmp==False, 0.)
    #np.place(tmp, tmp==-1, 1.)
    DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    #cvars = ['obs_obs']
    #for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    #del cond1
    DA_incr = LDAS_io('incr',exp, domain, root)

    tags = ['innov_mean','innov_var',
            'norm_innov_mean','norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean','incr_catdef_var',
            'incr_rzexc_mean','incr_rzexc_var',
            'incr_srfexc_mean','incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][:,:,:].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    for i_spc,spc in enumerate(species):

        logging.info('species %i' % (i_spc))

        ds['innov_mean'][i_spc,:,:] = (DA_innov.timeseries['obs_obs'][:,i_spc,:,:] - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['innov_var'][i_spc,:,:] = (DA_innov.timeseries['obs_obs'][:,i_spc,:,:] - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).var(dim='time',skipna=True).values
        ds['norm_innov_mean'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:] - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
                                                  np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).mean(dim='time',skipna=True).values
        ds['norm_innov_var'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:] - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
                                                 np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).var(dim='time',skipna=True).values

        tmp = DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:].values
        np.place(tmp, tmp==0., np.nan)
        np.place(tmp, tmp==-9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc,:,:] = tmp.sum(axis=0)

    np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
    ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time',skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time',skipna=True).values
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time',skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time',skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time',skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time',skipna=True).values

    ds.close()


def filter_diagnostics_evaluation_compare(exp1, exp2, domain, root, outputpath):

    result_file = outputpath + 'filter_diagnostics.nc'

    DA_CLSM_innov = LDAS_io('ObsFcstAna',exp1, domain, root)
    DA_PEAT_innov = LDAS_io('ObsFcstAna',exp2, domain, root)

    DA_CLSM_incr = LDAS_io('incr',exp1, domain, root)
    DA_PEAT_incr = LDAS_io('incr',exp2, domain, root)

    runs = OrderedDict([(1,[DA_CLSM_innov.timeseries, DA_CLSM_incr.timeseries]),
                        (2,[DA_PEAT_innov.timeseries, DA_PEAT_incr.timeseries])])

    tags = ['innov_mean','innov_var',
            'norm_innov_mean','norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean','incr_catdef_var',
            'incr_rzexc_mean','incr_rzexc_var',
            'incr_srfexc_mean','incr_srfexc_var']

    tc = DA_CLSM_innov.grid.tilecoord
    tg = DA_CLSM_innov.grid.tilegrids
    lons = DA_CLSM_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = DA_CLSM_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]

    species = DA_CLSM_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, runs.keys(), species, tags)

    for i_run,run in enumerate(runs):
        for i_spc,spc in enumerate(species):

            logging.info('run %i, species %i' % (i_run,i_spc))

            ds['innov_mean'][:,:,i_run,i_spc] = (runs[run][0]['obs_obs'][:,i_spc,:,:] - runs[run][0]['obs_fcst'][:,i_spc,:,:]).mean(dim='time').values
            ds['innov_var'][:,:,i_run,i_spc] = (runs[run][0]['obs_obs'][:,i_spc,:,:] - runs[run][0]['obs_fcst'][:,i_spc,:,:]).var(dim='time').values
            ds['norm_innov_mean'][:,:,i_run,i_spc] = ((runs[run][0]['obs_obs'][:,i_spc,:,:] - runs[run][0]['obs_fcst'][:,i_spc,:,:]) /
                                                      np.sqrt(runs[run][0]['obs_obsvar'][:,i_spc,:,:] + runs[run][0]['obs_fcstvar'][:,i_spc,:,:])).mean(dim='time').values
            ds['norm_innov_var'][:,:,i_run,i_spc] = ((runs[run][0]['obs_obs'][:,i_spc,:,:] - runs[run][0]['obs_fcst'][:,i_spc,:,:]) /
                                                     np.sqrt(runs[run][0]['obs_obsvar'][:,i_spc,:,:] + runs[run][0]['obs_fcstvar'][:,i_spc,:,:])).var(dim='time').values

            tmp = runs[run][0]['obs_obs'][:,i_spc,:,:].values
            np.place(tmp, ~np.isnan(tmp), 1.)
            np.place(tmp, np.isnan(tmp), 0.)
            ds['n_valid_innov'][:, :, i_run, i_spc] = tmp.sum(axis=0)

        if len(runs[run]) == 2:
            np.place(runs[run][1]['catdef'].values, runs[run][1]['catdef'].values == 0, np.nan)
            np.place(runs[run][1]['rzexc'].values, runs[run][1]['rzexc'].values == 0, np.nan)
            np.place(runs[run][1]['srfexc'].values, runs[run][1]['srfexc'].values == 0, np.nan)
            ds['incr_catdef_mean'][:, :, i_run] = runs[run][1]['catdef'].mean(dim='time').values
            ds['incr_catdef_var'][:, :, i_run] = runs[run][1]['catdef'].var(dim='time').values
            ds['incr_rzexc_mean'][:, :, i_run] = runs[run][1]['rzexc'].mean(dim='time').values
            ds['incr_rzexc_var'][:, :, i_run] = runs[run][1]['rzexc'].var(dim='time').values
            ds['incr_srfexc_mean'][:, :, i_run] = runs[run][1]['srfexc'].mean(dim='time').values
            ds['incr_srfexc_var'][:, :, i_run] = runs[run][1]['srfexc'].var(dim='time').values

    ds.close()

def estimate_tau(df, n_lags=180):
    """ Estimate characteristic time lengths for pd.DataFrame columns """

    # df must be already daily
    # df = in_df.copy().resample('1D').last()
    n_cols = len(df.columns)

    # calculate auto-correlation for different lags
    rho = np.full((n_cols,n_lags), np.nan)
    for lag in np.arange(n_lags):
        for i,col in enumerate(df):
            Ser_l = df[col].copy()
            Ser_l.index += pd.Timedelta(lag, 'D')
            rho[i,lag] = df[col].corr(Ser_l)

    #for i in np.arange(n_cols):
    #    plt.plot(rho[i,:],'.')

    # Fit exponential function to auto-correlations and estimate tau
    tau = np.full(n_cols, np.nan)
    for i in np.arange(n_cols):
        try:
            ind = np.where(~np.isnan(rho[i,:]))[0]
            if len(ind) > 20:
                popt = optimization.curve_fit(lambda x, a: np.exp(a * x), np.arange(n_lags)[ind], rho[i,ind],
                                              bounds = [-1., -1. / n_lags])[0]
                tau[i] = np.log(np.exp(-1.)) / popt
        except:
            # If fit doesn't converge, fall back to the lag where calculated auto-correlation actually drops below 1/e
            ind = np.where(rho[i,:] < np.exp(-1))[0]
            tau[i] = ind[0] if (len(ind) > 0) else n_lags # maximum = # calculated lags

    # print tau
    # import matplotlib.pyplot as plt
    # xlim = [0,60]
    # ylim = [-0.4,1]
    # plt.figure(figsize=(14,9))
    # for i in np.arange(n_cols):
    #     plt.subplot(n_cols,1,i+1)
    #     plt.plot(np.arange(n_lags),rho[i,:])
    #     plt.plot(np.arange(0,n_lags,0.1),np.exp(-np.arange(0,n_lags,0.1)/tau[i]))
    #     plt.plot([tau[i],tau[i]],[ylim[0],np.exp(-1)],'--k')
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    #     plt.text(xlim[1]-xlim[1]/15.,0.7,df.columns.values[i],fontsize=14)
    #
    # plt.tight_layout()
    # plt.show()

    return tau

def estimate_lag1_autocorr(in_df, tau=None):
    """ Estimate geometric average median lag-1 auto-correlation """

    df = in_df.copy().resample('1D').last()
    # Get auto-correlation length for all time series
    if tau is None:
        tau = estimate_tau(df)

    # Calculate gemetric average lag-1 auto-corr
    avg_spc_t = np.median((df.index[1::] - df.index[0:-1]).days)
    ac = np.exp(-avg_spc_t/tau)
    avg_ac = ac.prod()**(1./len(ac))

    return tau,ac,avg_ac

def lag1_autocorr_from_numpy_array(data):
    tau = np.empty((data.shape[1],data.shape[2]))*np.nan
    acor_lag1 = np.empty((data.shape[1],data.shape[2]))*np.nan
    nall = data.shape[1]*data.shape[2]
    for i in np.arange(data.shape[1]):
        k=(i+1)*data.shape[2]
        logging.info("%s/%s" % (k,nall))
        df = pd.DataFrame(data[:,i,:].values,index=data['time'].values)
        tau_tmp,ac_tmp,avg_ac_tmp = estimate_lag1_autocorr(df)
        tau[i,:] = tau_tmp
        acor_lag1[i,:] = ac_tmp
    return tau,acor_lag1

def lag1_autocorr_from_numpy_array_slow(data):
    acor_lag1 = np.empty((data.shape[1],data.shape[2]))*np.nan
    nall = data.shape[1]*data.shape[2]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            k=(i+1)*(j+1)
            logging.info("%s/%s" % (k,nall))
            df = data[:,i,j].to_series()
            df = pd.DataFrame(data[:,i,0:data.shape[2]].values,index=data['time'].values)
            # df = pd.DataFrame([df]).swapaxes(0,1).dropna()
            df=df.dropna()
            acor_lag1[i,j] = estimate_lag1_autocorr(df)
    return acor_lag1

def obs_M09_to_M36(data):
    ydim_ind = np.where(np.nanmean(data,0)>-9e30)
    indmin=0
    indmax=np.size(ydim_ind)
    try:
        if np.size(ydim_ind[0])>0:
            if ydim_ind[0][0]<3:
                indmin=1
            if ydim_ind[0][-1]>(data.shape[1]-3):
                indmax= np.size(ydim_ind)-1
            try:
                data[:,ydim_ind[0][indmin:indmax]-1] = data[:,ydim_ind[0][indmin:indmax]].values
                data[:,ydim_ind[0][indmin:indmax]+1] = data[:,ydim_ind[0][indmin:indmax]].values
                data[:,ydim_ind[0][indmin:indmax]-2] = data[:,ydim_ind[0][indmin:indmax]].values
                data[:,ydim_ind[0][indmin:indmax]+2] = data[:,ydim_ind[0][indmin:indmax]].values
            except:
                data[:,ydim_ind[0][indmin:indmax]-1] = data[:,ydim_ind[0][indmin:indmax]]
                data[:,ydim_ind[0][indmin:indmax]+1] = data[:,ydim_ind[0][indmin:indmax]]
                data[:,ydim_ind[0][indmin:indmax]-2] = data[:,ydim_ind[0][indmin:indmax]]
                data[:,ydim_ind[0][indmin:indmax]+2] = data[:,ydim_ind[0][indmin:indmax]]

        xdim_ind = np.where(np.nanmean(data,1)>-9e30)
        indmin=0
        indmax=np.size(xdim_ind)
        if np.size(xdim_ind[0])>0:
            if xdim_ind[0][0]<3:
                indmin=1
            if xdim_ind[0][-1]>(data.shape[0]-3):
                indmax= np.size(xdim_ind)-1
            try:
                data[xdim_ind[0][indmin:indmax]-1,:] = data[xdim_ind[0][indmin:indmax],:].values
                data[xdim_ind[0][indmin:indmax]+1,:] = data[xdim_ind[0][indmin:indmax],:].values
                data[xdim_ind[0][indmin:indmax]-2,:] = data[xdim_ind[0][indmin:indmax],:].values
                data[xdim_ind[0][indmin:indmax]+2,:] = data[xdim_ind[0][indmin:indmax],:].values
            except:
                data[xdim_ind[0][indmin:indmax]-1,:] = data[xdim_ind[0][indmin:indmax],:]
                data[xdim_ind[0][indmin:indmax]+1,:] = data[xdim_ind[0][indmin:indmax],:]
                data[xdim_ind[0][indmin:indmax]-2,:] = data[xdim_ind[0][indmin:indmax],:]
                data[xdim_ind[0][indmin:indmax]+2,:] = data[xdim_ind[0][indmin:indmax],:]
    except:
        pass
    return data

def setup_grid_grid_for_plot(io):
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons_1D = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats_1D = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    tc.i_indg -= tg.loc['domain','i_offg'] # col / lon
    tc.j_indg -= tg.loc['domain','j_offg'] # row / lat
    lons, lats = np.meshgrid(lons_1D, lats_1D)
    llcrnrlat = np.min(lats)
    urcrnrlat = np.max(lats)
    llcrnrlon = np.min(lons)
    urcrnrlon = np.max(lons)
    return lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon

def calc_tau_and_lag1_autocor(self):

    tmp_incr = self.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    tau,acor_lag1 = lag1_autocorr_from_numpy_array(tmp_incr)

    variables = ["tau", "acor_lag1"]

    out_file = os.path.join(self.paths.ana,'..','..','..','output_postprocessed',self.param + '_autocor.nc')

    domainlons = self.grid.ease_lons[np.min(self.grid.tilecoord.i_indg):(np.max(self.grid.tilecoord.i_indg)+1)]
    domainlats = self.grid.ease_lats[np.min(self.grid.tilecoord.j_indg):(np.max(self.grid.tilecoord.j_indg)+1)]

    lonmin = np.min(domainlons)
    lonmax = np.max(domainlons)
    latmin = np.min(domainlats)
    latmax = np.max(domainlats)

    # Use grid lon lat to avoid rounding issues
    tmp_tilecoord = self.grid.tilecoord.copy()
    tmp_tilecoord['com_lon'] = self.grid.ease_lons[self.grid.tilecoord.i_indg]
    tmp_tilecoord['com_lat'] = self.grid.ease_lats[self.grid.tilecoord.j_indg]

    # Clip region based on specified coordinate boundaries
    ind_img = self.grid.tilecoord[(tmp_tilecoord['com_lon']>=lonmin)&(tmp_tilecoord['com_lon']<=lonmax)&
                             (tmp_tilecoord['com_lat']<=latmax)&(tmp_tilecoord['com_lat']>=latmin)].index
    lons = domainlons[(domainlons >= lonmin) & (domainlons <= lonmax)]
    lats = domainlats[(domainlats >= latmin) & (domainlats <= latmax)]
    i_offg_2 = np.where(domainlons >= lonmin)[0][0]
    j_offg_2 = np.where(domainlats <= latmax)[0][0]

    dimensions = OrderedDict([('lat',lats), ('lon',lons)])

    dataset = self.ncfile_init(out_file, dimensions, variables)

    dataset.variables['tau'][:,:]=tau[:,:]
    dataset.variables['acor_lag1'][:,:]=acor_lag1[:,:]
    # Save file to disk and loat it as xarray Dataset into the class variable space
    dataset.close()

def calc_anomaly(Ser, method='moving_average', output='anomaly', longterm=False):

    if (output=='climatology')&(longterm is True):
        output = 'climSer'

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    climSer = pd.Series(index=xSer.index)

    if not method in ['harmonic','mean','moving_average','ma']:
        logging.info('Unknown method: %s' % (method))
        return climSer

    if longterm is True:
        if method=='harmonic':
            clim = calc_clim_harmonic(xSer)
        if method=='mean':
            clim = calc_clim_harmonic(xSer, n=0)
        if (method=='moving_average')|(method=='ma'):
            clim = calc_clim_moving_average(xSer)
        if output == 'climatology':
            return clim
        climSer[:] = clim[doys]

    else:
        years = xSer.index.year
        for yr in np.unique(years):
            if method == 'harmonic':
                clim = calc_clim_harmonic(xSer[years == yr])
            if method == 'mean':
                clim = calc_clim_harmonic(xSer[years == yr], n=0)
            if (method == 'moving_average') | (method == 'ma'):
                clim = calc_clim_moving_average(xSer[years == yr])
            climSer[years == yr] = clim[doys[years == yr]].values

    if output == 'climSer':
        return climSer

    return xSer - climSer

if __name__=='__main__':
    estimae_lag1_autocorr()

