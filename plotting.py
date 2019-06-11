
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from bat_pyldas.functions import *
from scipy.interpolate import interp2d


def plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=53, lon=25):

    outpath = os.path.join(outpath, exp, 'timeseries')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    fontsize = 12
    # get M09 rowcol with data
    col, row = io.grid.lonlat2colrow(lon, lat, domain=True)

    ts_mod_wtd = io.read_ts('zbar', lon, lat, lonlat=True)
    ts_mod_wtd.name = 'zbar (mod)'

    ts_mod_sfmc = io.read_ts('sfmc', lon, lat, lonlat=True)
    ts_mod_sfmc.name = 'sfmc (mod)'

    df = pd.concat((ts_mod_wtd, ts_mod_sfmc),axis=1)

    plt.figure(figsize=(19,8))

    ax1 = plt.subplot(211)
    df[['zbar (mod)']].plot(ax=ax1, fontsize=fontsize, style=['.-'], linewidth=2)
    plt.ylabel('zbar [m]')

    ax2 = plt.subplot(212)
    df[['sfmc (mod)']].plot(ax=ax2, fontsize=fontsize, style=['.-'], linewidth=2)
    plt.ylabel('sfmc $\mathregular{[m^3/m^3]}$')

    my_title = 'lon=%s, lat=%s' % (lon,lat)
    plt.title(my_title, fontsize=fontsize+2)
    plt.tight_layout()
    fname = 'wtd_sfmc_lon_%s_lat_%s' % (lon,lat)
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

def plot_catparams(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps','catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids


    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')

    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        # cmin cmax definition, best defined for every variable
        if ((param=="poros") | (param=="poros30")):
            cmin = 0
            cmax = 0.92
        else:
            cmin = None
            cmax = None
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)


def plot_sfmc_std(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_data = io.timeseries['sfmc']
    tmp_data = tmp_data.where(tmp_data != 0)
    tmp_data = tmp_data.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 0.12

    fname='sfmc_std'
    plot_title='sfmc_std [m3/m3]'
    figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def plot_zbar_std(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_data = io.timeseries['zbar']
    tmp_data = tmp_data.where(tmp_data != 0)
    tmp_data = tmp_data.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 0.7

    fname='zbar_std'
    plot_title='zbar_std [m]'
    figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def plot_waterstorage_std(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_data = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    tmp_data = tmp_data.where(tmp_data != 0)
    tmp_data = tmp_data.std(dim='time',skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = 100

    fname='waterstorage_std'
    plot_title='waterstorage_std [mm]'
    figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def figure_single_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                              llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    cmap = 'jet'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < 0.0:
        cmax = np.max([-cmin,cmax])
        cmin = -cmax
        cmap = 'seismic'
    # open figure
    fig_aspect_ratio = (0.1*(np.max(lons)-np.min(lons)))/(0.18*(np.max(lats)-np.min(lats)))
    figsize = (fig_aspect_ratio+10,10)
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-80.0,81,5.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(0.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    #lat=53.
    #lon=38.5
    #x,y = m(lon, lat)
    #m.plot(x, y, 'ko', markersize=6,mfc='none')
    #cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "bottom", size="6%", pad="15%",shrink=0.5)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title(plot_title, fontsize=fontsize)
    #plt.tight_layout()
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################



def plot_RTMparams(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')

    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)

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
    parallels = np.arange(-80.0,81,5.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(0.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="15%")
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

def plot_filter_diagnostics(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    ############## n_valid_innov
    #innov_var_cal = ds['innov_var'][:, :, :].mean(dim='species',skipna=True)).values
    data = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data)))

    ############## n_valid_quatro
    data = ds['n_valid_innov'][:, :, 0]/N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 1]/N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 2]/N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 3]/N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov_quatro'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))

    ############## n_valid_incr
    data = ds['n_valid_incr']/N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 1.00
    fname='n_valid_incr'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data)))

    ########## incr std #############
    data0 = ds['innov_mean'][:, :, 0].values
    data1 = ds['innov_mean'][:, :, 1].values
    data2 = ds['innov_mean'][:, :, 2].values
    data3 = ds['innov_mean'][:, :, 3].values
    cmin = 0
    cmax = 4
    fname='innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))

    ########## incr std #############
    data = ds['incr_srfexc_var'].values**0.5
    cmin = 0
    cmax = 10
    fname='incr_srfexc_std'
    my_title='std(\u0394srfexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_rzexc_var'].values**0.5
    cmin = 0
    cmax = 10
    fname='incr_rzexc_std'
    my_title='std(\u0394rzexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_catdef_var'].values**0.5
    cmin = 0
    cmax = 10
    fname='incr_catdef_std'
    my_title='std(\u0394catdef): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

def plot_increments(exp, domain, root, outpath, dti):

    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1==timestamp)[0]
        data = io.images['catdef'][timestep[0],:,:] + io.images['srfexc'][timestep[0],:,:] + io.images['rzexc'][timestep[0],:,:]
        if np.nanmean(data)==0:
            continue
        data = data.where(data!=0)
        # other parameter definitions
        cmin = None
        cmax = None
        fname='incr_'+str(timestamp)[0:13]
        my_title='increments (total water) (mm)'
        figure_single_default(data=-data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
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
    tmp_incr = io.timeseries['catdef'] #+ io.timeseries['srfexc'] + io.timeseries['rzexc']
    #tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    #incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    N_3hourly = (io.driver['DRIVER_INPUTS']['end_time']['year']-io.driver['DRIVER_INPUTS']['start_time']['year'])*365*8
    data = tmp_incr.count(dim='time').values / N_3hourly

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
    cmin = -5
    cmax = 5

    fname='delta_incr'
    plot_title='incr_CLSM - incr_PEATCLSM'
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,plot_title=plot_title)


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

    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days  # for 3hourly ObsFcstAna
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

def plot_innov_quatro(exp, domain, root, outpath, dti):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1==timestamp)[0]
        data = io.images['obs_obs'][timestep[0],:,:,:] - io.images['obs_fcst'][timestep[0],:,:,:]
        if np.nanmean(data)==0:
            continue
        # calculate variable to plot
        data0 = obs_M09_to_M36(data[0,:,:])
        data1 = obs_M09_to_M36(data[1,:,:])
        data2 = obs_M09_to_M36(data[2,:,:])
        data3 = obs_M09_to_M36(data[3,:,:])

        # other parameter definitions
        cmin = None
        cmax = None
        fname='innov_quatro_'+str(timestamp)[0:13]
        data = list([data0,data1,data2,data3])
        figure_quatro_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                              plot_title=list(['O-F (Tb in K) H-Asc','O-F (Tb in K) H-Des','O-F (Tb in K) V-Asc','O-F (Tb in K) V-Des']))

def plot_kalman_gain(exp, domain, root, outpath, dti):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1==timestamp)[0]
        # OF
        data = io.images['obs_obs'][timestep[0],:,:,:] - io.images['obs_fcst'][timestep[0],:,:,:]
        if np.nanmean(data)==0:
            continue
        # calculate variable to plot
        #tmp = data.sum(dim='species',skipna=True)
        tmp = data[0,:,:]
        tmp = tmp.where(tmp!=0)
        OF = obs_M09_to_M36(tmp)
        # incr
        data = io.images['obs_ana'][timestep[0],:,:,:] - io.images['obs_fcst'][timestep[0],:,:,:]
        if np.nanmean(data)==0:
            continue
        # calculate variable to plot
        #tmp = data.sum(dim='species',skipna=True)
        tmp = data[0,:,:]
        tmp = tmp.where(tmp!=0)
        incr = obs_M09_to_M36(tmp)
        # other parameter definitions
        cmin = 0
        cmax = 1
        fname='kalman_gain_'+str(timestamp)[0:13]
        data = incr/OF
        my_title='incr / (O - F) '
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

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

def plot_innov_delta_std_single(exp1, exp2, domain, root, outpath):
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
    data = - 0.25*(data0+data1+data2+data3)
    # other parameter definitions
    cmin = -2
    cmax = 2
    fname='innov_delta_std_single'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='change in std(O-F) (K)')


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
    figsize = (10, 10)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0,4):
        cmap = 'jet'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0:
            cmax = np.max([-cmin,cmax])
            cmin = -cmax
            cmap = 'seismic'
        cbrange = (cmin, cmax)
        plt.subplot(4,1,i+1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(-80.0,81,5.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(0.,351.,10.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="7%", pad="15%")
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

def figure_quatro_scaling(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                          llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    # open figure
    figsize = (10, 10)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0,4):
        cmap = 'jet'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0:
            cmax = np.max([-cmin,cmax])
            cmin = -cmax
            cmap = 'seismic'

        cbrange = (cmin, cmax)
        plt.subplot(4,1,i+1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(-80.0,81,5.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(0.,351.,10.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        lat=53.
        lon=38.5
        x,y = m(lon, lat)
        m.plot(x, y, 'ko', markersize=6,mfc='none')
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="4%", pad="16%")
        # scatter
        #x, y = m(lons, lats)
        #ax.scatter(x, y, s=10, c=res['m_mod_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
        plt.title(plot_title[i], fontsize=fontsize)

    fname_long = os.path.join(outpath, fname+'.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

def figure_double_scaling(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                          llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    # open figure
    figsize = (17, 8)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0,2):
        cmap = 'jet'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0:
            cmax = np.max([-cmin,cmax])
            cmin = -cmax
            cmap = 'seismic'

        cbrange = (cmin, cmax)
        plt.subplot(2,1,i+1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(-80.0,81,5.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(0.,351.,10.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="4%", pad="16%")
        # scatter
        #x, y = m(lons, lats)
        #ax.scatter(x, y, s=10, c=res['m_mod_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
        plt.title(plot_title[i], fontsize=fontsize)

    fname_long = os.path.join(outpath, fname+'.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

def figure_single_default_new(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                              llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    cmap = 'jet'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < 0.0:
        cmax = np.max([-cmin,cmax])
        cmin = -cmax
        cmap = 'seismic'

    cmap = plt.get_cmap(cmap)
    # open figure
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(lons, lats, data, cmap=cmap)
    fig.colorbar(cs,shrink=0.6)
    plt.title(plot_title)
    if not os.path.exists(os.path.join(outpath, exp)):
        os.mkdir(os.path.join(outpath, exp))
    fname_long = os.path.join(outpath, exp, fname+'.png')
    #plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

def figure_single_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                              llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title):
    cmap = 'jet'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < 0.0:
        cmax = np.max([-cmin,cmax])
        cmin = -cmax
        cmap = 'seismic'
    # open figure
    fig_aspect_ratio = (0.1*(np.max(lons)-np.min(lons)))/(0.18*(np.max(lats)-np.min(lats)))
    figsize = (fig_aspect_ratio+10,10)
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-80.0,81,5.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(0.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    #lat=53.
    #lon=38.5
    #x,y = m(lon, lat)
    #m.plot(x, y, 'ko', markersize=6,mfc='none')
    #cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "bottom", size="6%", pad="15%",shrink=0.5)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title(plot_title, fontsize=fontsize)
    #plt.tight_layout()
    fname_long = os.path.join(outpath, fname+'.png')
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

def plot_timeseries(exp, domain, root, outpath, lat=53, lon=25):

    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    io_incr = LDAS_io('incr', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    fontsize = 12

    # get M09 rowcol with data
    col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
    dcols = [-1,0,2,3]
    drows = [-1,0,2,3]
    cfind=False
    for icol,dcol in enumerate(dcols):
        for icol,drow in enumerate(drows):
            ts_obs = io.read_ts('obs_obs', col+dcol, row+drow, species=2, lonlat=False)
            if np.isnan(ts_obs.max(skipna=True))==False:
                cfind = True
                break
        if cfind==True:
            break
    row = row+drow
    col = col+dcol

    ts_obs_1 = io.read_ts('obs_obs', col, row, species=1, lonlat=False)
    ts_obs_1.name = 'Tb obs (species=1)'
    ts_obs_2 = io.read_ts('obs_obs', col, row, species=2, lonlat=False)
    ts_obs_2.name = 'Tb obs (species=2)'
    ts_obs_3 = io.read_ts('obs_obs', col, row, species=3, lonlat=False)
    ts_obs_3.name = 'Tb obs (species=3)'
    ts_obs_4 = io.read_ts('obs_obs', col, row, species=4, lonlat=False)
    ts_obs_4.name = 'Tb obs (species=4)'

    ts_fcst_1 = io.read_ts('obs_fcst', col, row, species=1, lonlat=False)
    ts_fcst_1.name = 'Tb fcst (species=1)'
    ts_fcst_2 = io.read_ts('obs_fcst', col, row, species=2, lonlat=False)
    ts_fcst_2.name = 'Tb fcst (species=2)'
    ts_fcst_3 = io.read_ts('obs_fcst', col, row, species=3, lonlat=False)
    ts_fcst_3.name = 'Tb fcst (species=3)'
    ts_fcst_4 = io.read_ts('obs_fcst', col, row, species=4, lonlat=False)
    ts_fcst_4.name = 'Tb fcst (species=4)'

    ts_innov_1 = ts_obs_1 - ts_fcst_1
    ts_innov_1.name = 'innov (species=1)'
    ts_innov_2 = ts_obs_2 - ts_fcst_2
    ts_innov_2.name = 'innov (species=2)'
    ts_innov_3 = ts_obs_3 - ts_fcst_3
    ts_innov_3.name = 'innov (species=3)'
    ts_innov_4 = ts_obs_4 - ts_fcst_4
    ts_innov_4.name = 'innov (species=4)'

    ts_incr_catdef = io_incr.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
    ts_incr_catdef.name = 'catdef incr'
    ts_incr_srfexc = io_incr.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
    ts_incr_srfexc.name = 'srfexc incr'

    df = pd.concat((ts_innov_1, ts_innov_2, ts_innov_3, ts_innov_4, ts_incr_srfexc),axis=1)
    #df = pd.concat((ts_obs_1, ts_obs_2, ts_obs_3, ts_obs_4, ts_fcst_1, ts_fcst_2, ts_fcst_3, ts_fcst_4, ts_incr_catdef),axis=1).dropna()

    plt.figure(figsize=(19,8))

    ax1 = plt.subplot(211)
    #df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    df[['innov (species=1)','innov (species=2)','innov (species=3)','innov (species=4)']].plot(ax=ax1, fontsize=fontsize, style=['.-','.-','.-','.-'], linewidth=2)
    plt.ylabel('Tb [K]')

    ax2 = plt.subplot(212)
    #df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    df[['srfexc incr']].plot(ax=ax2, fontsize=fontsize, style=['o:'], linewidth=2)
    plt.ylabel('incr [mm]')
    my_title = 'lon=%s, lat=%s' % (lon,lat)
    plt.title(my_title, fontsize=fontsize+2)

    plt.tight_layout()
    fname = 'innov_incr_lon_%s_lat_%s' % (lon,lat)
    fname_long = os.path.join(outpath, exp, fname+'.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


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

def plot_scaling_parameters(exp, domain, root, outpath):

    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('scale',exp,domain,root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    #fpath = os.path.join(root,exp,'output',domain,'stats/z_score_clim/pentad/')
    scalepath = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalepath'].split()[0]
    #fname = 'ScMO_Thvf_TbSM_001_SMOS_zscore_stats_2010_p1_2018_p73_hscale_0.00_W_9p_Nmin_20'
    scalename = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalename'][5:].split()[0]
    for i_AscDes,AscDes in enumerate(list(["A","D"])):
        for i_pentad in np.arange(1,74):
            fname = scalepath+scalename+'_'+AscDes+'_p'+"%02d" % (i_pentad,)+'.bin'
            res = io.read_scaling_parameters(fname=fname)

            res = res[['lon','lat','m_mod_H_%2i'%angle,'m_mod_V_%2i'%angle,'m_obs_H_%2i'%angle,'m_obs_V_%2i'%angle]]
            res.replace(-9999.,np.nan,inplace=True)

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_H_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data0 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_V_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data1 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_H_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data2 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_V_%2i'%angle].values
            img = obs_M09_to_M36(img)
            data3 = np.ma.masked_invalid(img)
            # other parameter definitions
            cmin = 210
            cmax = 300
            fname='scaling'+'_'+AscDes+'_p'+"%02d" % (i_pentad,)
            data = list([data0,data1,data2,data3])
            figure_quatro_scaling(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                                  llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                                  plot_title=list(['m_mod_H_%2i'%angle,'m_mod_V_%2i'%angle,'m_obs_H_%2i'%angle,'m_obs_V_%2i'%angle]))


def plot_scaling_parameters_average(exp, domain, root, outpath):

    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('ObsFcstAna',exp,domain,root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # get scaling nc
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'scaling.nc')

    AscDes = ['A','D']
    data0 = ds['m_obs_H_%2i'%angle][:,:,:,0].count(axis=2)
    data1 = ds['m_obs_H_%2i'%angle][:,:,:,1].count(axis=2)
    data2 = ds['m_obs_V_%2i'%angle][:,:,:,0].count(axis=2)
    data3 = ds['m_obs_V_%2i'%angle][:,:,:,1].count(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data0.where(data1!=0)
    data2 = data0.where(data2!=0)
    data3 = data0.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 73
    fname='n_valid_scaling'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['N pentads (H-Asc)', 'N pentads (H-Des)','N pentads (V-Asc)', 'N pentads (V-Des)']))

    AscDes = ['A','D']
    tmp = ds['m_mod_H_%2i'%angle][:,:,:,0].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data0 = tmp.sum(axis=2)
    tmp = ds['m_mod_H_%2i'%angle][:,:,:,1].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data1 = tmp.sum(axis=2)
    tmp = ds['m_mod_V_%2i'%angle][:,:,:,0].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data2 = tmp.sum(axis=2)
    tmp = ds['m_mod_V_%2i'%angle][:,:,:,1].values
    np.place(tmp, tmp==0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data3 = tmp.sum(axis=2)
    np.place(data0, data0==0, np.nan)
    np.place(data1, data1==0, np.nan)
    np.place(data2, data2==0, np.nan)
    np.place(data3, data3==0, np.nan)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 73
    fname='n_valid_scaling_mod'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['N pentads (H-Asc)', 'N pentads (H-Des)','N pentads (V-Asc)', 'N pentads (V-Des)']))

def plot_filter_diagnostics_short(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics_short.nc')

    ############## n_valid_innov
    #innov_var_cal = ds['innov_var'][:, :, :].mean(dim='species',skipna=True)).values
    data = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    data = data.where(data != 0.0)
    data = obs_M09_to_M36(data)
    data = data * N_days
    cmin = 0.
    cmax = 20.
    fname='n_valid_innov_short'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='N total: m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data)))

    ############## n_valid_quatro
    data = ds['n_valid_innov'][:, :, 0]/N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 1]/N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 2]/N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][:, :, 3]/N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov_short_quatro'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))


if __name__=='__main__':
    plot_ismn_statistics()
