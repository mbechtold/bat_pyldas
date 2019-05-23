

import os

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')

from pyldas.grids import EASE2

from pyldas.interface import LDAS_io

from myprojects.timeseries import calc_anomaly

from pyldas_plots.functions import estimate_lag1_autocorr
from pyldas_plots.functions import estimate_tau
from pyldas_plots.functions import lag1_autocorr_from_numpy_array
from pyldas_plots.functions import obs_M09_to_M36
from pyldas_plots.functions import setup_grid_grid_for_plot

from scipy.interpolate import interp2d

def plot_catparams(exp, domain, root, outpath):

    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain).read_params('catparam')

    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)

def plot_RTMparams(exp, domain, root, outpath):

    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain).read_params('RTMparam')

    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)


def plot_increments_std(exp, domain, root, outpath):

    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    data = tmp_incr.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 10

    fname='incr_std'
    my_title='std(increments total water) (mm): m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

def plot_increments_tp1_std(exp, domain, root, outpath):

    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['tp1']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    data = tmp_incr.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = None

    fname='incr_tp1_std'
    my_title='std(increments tp1) (K): m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)


def plot_increments_number(exp, domain, root, outpath):

    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    N_days = (io.driver['DRIVER_INPUTS']['end_time']['year']-io.driver['DRIVER_INPUTS']['start_time']['year'])*365*8
    data = tmp_incr.count(dim='time').values / N_days

    # other parameter definitions
    cmin = 0
    cmax = 0.35

    fname='incr_number'
    my_title='N per day (incr.): m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

def plot_Fcst_std_delta(exp1, exp2, domain, root, outpath):

    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data_1_0 = obs_M09_to_M36(data_std0)
    data_1_1 = obs_M09_to_M36(data_std1)
    data_1_2 = obs_M09_to_M36(data_std2)
    data_1_3 = obs_M09_to_M36(data_std3)

    io = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data_2_0 = obs_M09_to_M36(data_std0)
    data_2_1 = obs_M09_to_M36(data_std1)
    data_2_2 = obs_M09_to_M36(data_std2)
    data_2_3 = obs_M09_to_M36(data_std3)

    data0 = data_2_0 - data_1_0
    data1 = data_2_1 - data_1_1
    data2 = data_2_2 - data_1_2
    data3 = data_2_3 - data_1_3

    # other parameter definitions
    cmin = None
    cmax = None

    fname='delta_Fcst'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=list(['delta std (K) H-Asc','delta std (K) H-Des','delta std (K) V-Asc','delta std (K) V-Des']))

def plot_increments_std_delta(exp1, exp2, domain, root, outpath):

    io = LDAS_io('incr', exp=exp1, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std1 = tmp_incr.std(dim='time',skipna=True).values

    io = LDAS_io('incr', exp=exp2, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std2 = tmp_incr.std(dim='time',skipna=True).values

    data = incr_std2 - incr_std1
    # other parameter definitions
    cmin = None
    cmax = None

    fname='delta_incr'
    plot_title='incr_CLSM - incr_PEATCLSM'
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,plot_title=plot_title)

def plot_sfmc_std(exp, domain, root, outpath):

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    #tmp_data = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    tmp_data = io.timeseries['sfmc']
    tmp_data = tmp_data.where(tmp_data != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    tmp_data = tmp_data.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 0.12

    fname='sfmc_std'
    plot_title='sfmc_std (m3/m3)'
    figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def plot_waterstorage_std(exp, domain, root, outpath):

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_data = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_data = io.timeseries['sfmc']
    tmp_data = tmp_data.where(tmp_data != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    tmp_data = tmp_data.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 100

    fname='waterstorage_std'
    plot_title='waterstorage_std (mm)'
    figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def plot_lag1_autocor(exp, domain, root, outpath):

    # set up grid
    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    tc.i_indg -= tg.loc['domain','i_offg'] # col / lon
    tc.j_indg -= tg.loc['domain','j_offg'] # row / lat
    lons, lats = np.meshgrid(lons, lats)
    llcrnrlat = np.min(lats)
    urcrnrlat = np.max(lats)
    llcrnrlon = np.min(lons)
    urcrnrlon = np.max(lons)

    # calculate variable to plot
    #tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    tau,acor_lag1 = lag1_autocorr_from_numpy_array(tmp_incr)

    # open figure
    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 5)
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(tau)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    parallels = np.arange(0.,81,10.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Increment stdv (mm)', fontsize=fontsize)
    if not os.path.exists(os.path.join(outpath, exp)):
        os.mkdir(os.path.join(outpath, exp))
    fname = os.path.join(outpath, exp, 'incr_tau.png')
    #plt.tight_layout()
    plt.savefig(fname, dpi=f.dpi)
    plt.close()

def plot_obs_std_single(exp, domain, root, outpath):

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['obs_ana']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std0 = tmp_incr[:,0,:,:].std(dim='time',skipna=True).values
    incr_std1 = tmp_incr[:,1,:,:].std(dim='time',skipna=True).values
    incr_std2 = tmp_incr[:,2,:,:].std(dim='time',skipna=True).values
    incr_std3 = tmp_incr[:,3,:,:].std(dim='time',skipna=True).values
    incr_std = np.nanmean(np.stack((incr_std0,incr_std1,incr_std2,incr_std3),axis=2),2)
    data = obs_M09_to_M36(incr_std)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname='obs_std'
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title='obs std (mm)')

def plot_obs_ana_std_quatro(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_ana']
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname='obs_ana_std_quatro'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['obs_ana std (mm) H-Asc','obs_ana std (mm) H-Des','obs_ana std (mm) V-Asc','obs_ana std (mm) V-Des']))

def plot_obs_obs_std_quatro(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs']
    #data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname='obs_obs_std_quatro'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['obs_obs std (mm) H-Asc','obs_obs std (mm) H-Des','obs_obs std (mm) V-Asc','obs_obs std (mm) V-Des']))

    N_days = (io.driver['DRIVER_INPUTS']['end_time']['year']-io.driver['DRIVER_INPUTS']['start_time']['year'])*365
    data_num0 = data[:,0,:,:].count(dim='time').values / N_days
    data_num1 = data[:,1,:,:].count(dim='time').values / N_days
    data_num2 = data[:,2,:,:].count(dim='time').values / N_days
    data_num3 = data[:,3,:,:].count(dim='time').values / N_days
    data_num2 = data_num0 + data_num1
    data_num0[data_num0==0] = np.nan
    data_num1[data_num1==0] = np.nan
    data_num2[data_num2==0] = np.nan
    data_num3[data_num3==0] = np.nan
    data_num0 = obs_M09_to_M36(data_num0)
    data_num1 = obs_M09_to_M36(data_num1)
    data_num2 = obs_M09_to_M36(data_num2)
    data_num3 = obs_M09_to_M36(data_num3)
    data_num3[:,:] = 0.0

    # other parameter definitions
    cmin = 0
    cmax = 0.35
    fname='obs_number_quatro'
    my_title_0='N_obs per day (Asc.): m = %.2f, s = %.2f' % (np.nanmean(data_num0),np.nanstd(data_num0))
    my_title_1='N_obs per day (Des.): m = %.2f, s = %.2f' % (np.nanmean(data_num1),np.nanstd(data_num1))
    my_title_2='N_obs per day (Asc.+Des.): m = %.2f, s = %.2f' % (np.nanmean(data_num2),np.nanstd(data_num2))
    my_title_3='...'
    my_title=list([my_title_0,my_title_1,my_title_2,my_title_3])
    data_all = list([data_num0,data_num1,data_num2,data_num3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=my_title)

def plot_obs_fcst_std_quatro(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname='obs_fcst_std_quatro'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['obs_fcst std (mm) H-Asc','obs_fcst std (mm) H-Des','obs_fcst std (mm) V-Asc','obs_fcst std (mm) V-Des']))

def plot_innov_std_quatro(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 15
    fname='innov_std_quatro'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['innov std (Tb) H-Asc','innov std (Tb) H-Des','innov std (Tb) V-Asc','innov std (Tb) V-Des']))

def plot_innov_norm_std_quatro(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data/np.sqrt(io.timeseries['obs_obsvar']+io.timeseries['obs_fcstvar'])
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0.5
    cmax = 1.5
    fname='innov_norm_std_quatro'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['innov norm std (-) H-Asc','innov norm std (-) H-Des','innov norm std (-) V-Asc','innov norm std (-) V-Des']))

def plot_innov_delta_std_quatro(exp1, exp2, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des
    exp = exp1
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data_exp1_0 = obs_M09_to_M36(data_std0)
    data_exp1_1 = obs_M09_to_M36(data_std1)
    data_exp1_2 = obs_M09_to_M36(data_std2)
    data_exp1_3 = obs_M09_to_M36(data_std3)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:,0,:,:].std(dim='time',skipna=True).values
    data_std1 = data[:,1,:,:].std(dim='time',skipna=True).values
    data_std2 = data[:,2,:,:].std(dim='time',skipna=True).values
    data_std3 = data[:,3,:,:].std(dim='time',skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)
    data0 = data0 - data_exp1_0
    data1 = data1 - data_exp1_1
    data2 = data2 - data_exp1_2
    data3 = data3 - data_exp1_3

    # other parameter definitions
    cmin = None
    cmax = None
    fname='innov_delta_std_quatro'
    data = list([data0,data1,data2,data3])
    figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['d_innov std (Tb) H-Asc','d_innov std (Tb) H-Des','d_innov std (Tb) V-Asc','d_innov std (Tb) V-Des']))

def figure_quatro_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                          llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    # open figure
    figsize = (25, 5)
    cmap = 'jet'
    fontsize = 18
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0,4):
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        cbrange = (cmin, cmax)
        plt.subplot(1,4,i+1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        parallels = np.arange(0.,81,5.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(10.,351.,10.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="7%", pad="4%")
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
        plt.title(plot_title[i], fontsize=fontsize)
    if not os.path.exists(os.path.join(outpath, exp)):
        os.mkdir(os.path.join(outpath, exp))
    fname_long = os.path.join(outpath, exp, fname+'.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

def figure_single_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                              llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    # open figure
    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, cmax)
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    parallels = np.arange(0.,81,5.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title(plot_title, fontsize=fontsize)
    if not os.path.exists(os.path.join(outpath, exp)):
        os.mkdir(os.path.join(outpath, exp))
    fname_long = os.path.join(outpath, exp, fname+'.png')
    #plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def plot_innovation_variance():

    ds = xr.open_dataset(r"D:\work\LDAS\2018-02_scaling\_new\diagnostics\filter_diagnostics.nc")

    innov_var_cal = ds['innov_var'][:, :, 1, :].mean(dim='species').values
    innov_var_uncal = ds['innov_var'][:, :, 3, :].mean(dim='species').values

    lons = ds.lon.values
    lats = ds.lat.values

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 120)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt_img = np.ma.masked_invalid(innov_var_cal)

    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Innovation variance (calibrated RTM)', fontsize=fontsize)

    plt.subplot(212)
    plt_img = np.ma.masked_invalid(innov_var_uncal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Innovation variance (uncalibrated RTM)', fontsize=fontsize)


    plt.tight_layout()
    plt.show()


def plot_normalized_innovation_variance():

    ds = xr.open_dataset(r"D:\work\LDAS\2018-02_scaling\_new\diagnostics\filter_diagnostics.nc")

    innov_var_cal = ds['norm_innov_var'][:, :, 1, :].mean(dim='species').values
    innov_var_uncal = ds['norm_innov_var'][:, :, 3, :].mean(dim='species').values

    lons = ds.lon.values
    lats = ds.lat.values

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 3)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt_img = np.ma.masked_invalid(innov_var_cal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Normalized innovation variance (calibrated RTM)', fontsize=fontsize)

    plt.subplot(212)
    plt_img = np.ma.masked_invalid(innov_var_uncal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Normalized innovation variance (uncalibrated RTM)', fontsize=fontsize)


    plt.tight_layout()
    plt.show()


def plot_ismn_statistics():

    fname = r"D:\work\LDAS\2018-02_scaling\ismn_eval\no_da_cal_uncal_ma_harm\validation_masked.csv"

    res = pd.read_csv(fname, index_col=0)

    variables = ['sm_surface','sm_rootzone','sm_profile']
    runs = ['OL', 'DA_ref', 'DA_pent']
    legend = ['open loop', 'calibrated', 'uncalibrated']

    # networks = ['SCAN','USCRN']
    # res.index = res.network
    # res = res.loc[networks,:]

    r_title = 'Pearson R '
    ubrmsd_title = 'ubRMSD '

    plt.figure(figsize=(7, 8))

    offsets = [-0.2, 0, 0.2]
    cols = ['lightblue', 'lightgreen', 'coral']
    # offsets = [-0.3,-0.1,0.1,0.3]
    # cols = ['lightblue', 'lightgreen', 'coral', 'brown']
    fontsize = 12

    ax = plt.subplot(211)
    plt.grid(color='k', linestyle='--', linewidth=0.25)

    data = list()
    ticks = list()
    pos = list()
    colors = list()

    for i, var in enumerate(variables):
        ticks.append(var)
        for col, offs, run in zip(cols, offsets, runs):
            tmp_data = res['corr_' + run + '_ma_' + var].values
            tmp_data = tmp_data[~np.isnan(tmp_data)]
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(variables)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 1.0)
    for i in np.arange(len(variables)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title(r_title, fontsize=fontsize)

    # ---------------------------------------------------------------------------------------------------------
    ax = plt.subplot(212)
    plt.grid(color='k', linestyle='--', linewidth=0.25)

    data = list()
    ticks = list()
    pos = list()
    colors = list()

    for i, var in enumerate(variables):
        ticks.append(var)
        for col, offs, run in zip(cols, offsets, runs):
            tmp_data = res['ubrmsd_' + run + '_ma_' + var].values
            tmp_data = tmp_data[~np.isnan(tmp_data)]
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    # plt.figlegend((box['boxes'][0:4]),runs,'upper left',fontsize=fontsize)
    plt.xticks(np.arange(len(variables)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0, 0.1)
    for i in np.arange(len(variables)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title(ubrmsd_title, fontsize=fontsize)

    plt.show()

def plot_timeseries():

    # Colorado
    # lat = 39.095962936
    # lon = -106.918945312

    # Nebraska
    # lat = 41.203456192
    # lon = -102.249755859

    # New Mexico
    # lat = 31.522361470
    # lon = -108.528442383

    # Oklahoma
    lat = 35.205233348
    lon = -97.910156250

    cal = LDAS_io('incr','US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('incr','US_M36_SMOS_DA_nocal_scaled_pentadal')

    incr_var_cal = (cal.timeseries['srfexc'] + cal.timeseries['rzexc'] - cal.timeseries['catdef']).var(dim='time').values
    incr_var_uncal = (uncal.timeseries['srfexc'] + uncal.timeseries['rzexc'] - uncal.timeseries['catdef']).var(dim='time').values

    col, row = LDAS_io().grid.lonlat2colrow(lon, lat, domain=True)

    title = 'increment variance (calibrated): %.2f        increment variance (uncalibrated): %.2f' % (incr_var_cal[row,col], incr_var_uncal[row,col])
    # title = ''

    fontsize = 12

    cal = LDAS_io('ObsFcstAna', 'US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('ObsFcstAna', 'US_M36_SMOS_DA_nocal_scaled_pentadal')
    orig = LDAS_io('ObsFcstAna', 'US_M36_SMOS_noDA_unscaled')

    ts_obs_cal = cal.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_cal.name = 'Tb obs (calibrated)'
    ts_obs_uncal = uncal.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_uncal.name = 'Tb obs (uncalibrated)'

    ts_obs_orig = orig.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_orig.name = 'Tb obs (uncalibrated, unscaled)'

    ts_fcst_cal = cal.read_ts('obs_fcst', lon, lat, species=3, lonlat=True)
    ts_fcst_cal.name = 'Tb fcst (calibrated)'
    ts_fcst_uncal = uncal.read_ts('obs_fcst', lon, lat, species=3, lonlat=True)
    ts_fcst_uncal.name = 'Tb fcst (uncalibrated)'

    df = pd.concat((ts_obs_cal, ts_obs_uncal, ts_obs_orig, ts_fcst_cal, ts_fcst_uncal),axis=1).dropna()

    plt.figure(figsize=(19,8))

    ax1 = plt.subplot(211)
    df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    plt.xlabel('')
    plt.title(title, fontsize=fontsize+2)

    cols = df.columns.values
    for i,col in enumerate(df):
        df[col] = calc_anomaly(df[col], method='ma', longterm=True).values
        if i < 3:
            cols[i] = col[0:7] + 'anomaly ' + col[7::]
        else:
            cols[i] = col[0:7] + ' anomaly' + col[7::]
    df.columns = cols
    df.dropna(inplace=True)

    ax2 = plt.subplot(212, sharex=ax1)
    df.plot(ax=ax2, ylim=[-60,60], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    plt.xlabel('')

    plt.tight_layout()
    plt.show()

def plot_ismn_locations():

    fname = r"D:\work\LDAS\2018-02_scaling\ismn_eval\no_da_cal_uncal_ma_harm\validation_masked.csv"

    res = pd.read_csv(fname, index_col=0)

    lats = res['lat'].values
    lons = res['lon'].values

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 5)

    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    x, y = m(lons, lats)
    plt.plot(x, y, 'o', markersize=7, markeredgecolor='k', markerfacecolor='coral')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    plt.title('ISMN station locations', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_model_parameters():

    cal = LDAS_io(exp='US_M36_SMOS_DA_calibrated_scaled').read_params('RTMparam')
    uncal = LDAS_io(exp='US_M36_SMOS_DA_nocal_scaled_pentadal').read_params('RTMparam')

    tc = LDAS_io().tilecoord
    tg = LDAS_io().tilegrids

    tc.i_indg -= tg.loc['domain', 'i_offg']  # col / lon
    tc.j_indg -= tg.loc['domain', 'j_offg']  # row / lat

    lons = np.unique(tc.com_lon.values)
    lats = np.unique(tc.com_lat.values)[::-1]

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64
    figsize = (14, 8)
    cbrange = (0, 1)
    cmap = 'jet'
    fontsize = 16

    plt.figure(figsize=figsize)

    plt.subplot(221)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cal['bh'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bh (calibrated)', fontsize=fontsize)

    plt.subplot(222)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cal['bv'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bv (calibrated)', fontsize=fontsize)

    plt.subplot(223)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = uncal['bh'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bh (uncalibrated)', fontsize=fontsize)

    plt.subplot(224)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = uncal['bv'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bv (uncalibrated)', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    plot_ismn_statistics()
