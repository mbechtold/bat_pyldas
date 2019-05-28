

import os

import xarray as xr
import pandas as pd
import numpy as np

from pyldas.grids import EASE2

from pyldas.interface import LDAS_io
import scipy.optimize as optimization
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from collections import OrderedDict

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
    if ydim_ind[0][0]<3:
        indmin=1
    if ydim_ind[0][-1]>(data.shape[1]-3):
        indmax= np.size(ydim_ind)-1
    data[:,ydim_ind[0][indmin:indmax]-1] = data[:,ydim_ind[0][indmin:indmax]]
    data[:,ydim_ind[0][indmin:indmax]+1] = data[:,ydim_ind[0][indmin:indmax]]
    data[:,ydim_ind[0][indmin:indmax]-2] = data[:,ydim_ind[0][indmin:indmax]]
    data[:,ydim_ind[0][indmin:indmax]+2] = data[:,ydim_ind[0][indmin:indmax]]

    xdim_ind = np.where(np.nanmean(data,1)>-9e30)
    indmin=0
    indmax=np.size(xdim_ind)
    if xdim_ind[0][0]<3:
        indmin=1
    if xdim_ind[0][-1]>(data.shape[0]-3):
        indmax= np.size(xdim_ind)-1
    data[xdim_ind[0][indmin:indmax]-1,:] = data[xdim_ind[0][indmin:indmax],:]
    data[xdim_ind[0][indmin:indmax]+1,:] = data[xdim_ind[0][indmin:indmax],:]
    data[xdim_ind[0][indmin:indmax]-2,:] = data[xdim_ind[0][indmin:indmax],:]
    data[xdim_ind[0][indmin:indmax]+2,:] = data[xdim_ind[0][indmin:indmax],:]

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

