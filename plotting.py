
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from bat_pyldas.functions import *
from scipy.stats import zscore
from scipy.interpolate import interp2d
from validation_good_practice.ancillary import metrics
import sys
import seaborn
from scipy import stats
import pymannkendall as mk

def assign_units(var):

    # Function to assign units to the variable 'var'.
    if var == 'srfexc' or var == 'rzexc' or var == 'catdef':
        unit = '[mm]'
    elif var == 'ar1' or var == 'ar2':
        unit = '[-]'
    elif var == 'sfmc' or var == 'rzmc':
        unit = '[m^3/m^3]'
    elif var == 'tsurf' or var == 'tp1' or var == 'tpN':
        unit = '[K]'
    elif var == 'shflux' or var == 'lhflux':
        unit = '[W/m^2]'
    elif var == 'evap' or var == 'runoff':
        unit = '[mm/day]'
    elif var == 'zbar':
        unit = '[m]'
    else:
        unit = ''
    return(unit)

def plot_all_variables_temporal_moments(exp, domain, root, outpath):

    # plot temporal mean and standard deviation of variables
    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)


    # mean
    m1 = io.timeseries.mean(axis=0)
    for varname, da in m1.data_vars.items():
        tmp_data = da
        #cmin = 0
        #cmax = 0.7
        cmin = None
        cmax = None
        plot_title=varname
        if varname=='zbar':
            cmin=-0.6
            cmax=-0.1
        if varname=='runoff':
            cmin=0
            cmax=5
        if varname=='evap':
            cmin=0
            cmax=5
        plot_title=varname+" "+assign_units(varname)
        fname=varname+'_mean'
        #plot_title='zbar [m]'
        figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)
    m2 = io.timeseries.std(axis=0)
    for varname, da in m2.data_vars.items():
        tmp_data = da
        #cmin = 0
        #cmax = 0.7
        cmin = None
        cmax = None
        fname=varname+'_std'
        #plot_title='zbar [m]'
        plot_title=varname
        figure_single_default(data=tmp_data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

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
        # cmin cmax, if not defined determined based on data
        if ((param=="poros") | (param=="poros30")):
            cmin = 0
            cmax = 0.92
        else:
            cmin = None
            cmax = None
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)

def plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, exp, outpath):

    # Initiate dataframe to store metrics in.
    INDEX = wtd_obs.columns
    COL = ['bias', 'ubRMSD', 'Pearson_R', 'RMSD']
    df_metrics = pd.DataFrame(index=INDEX, columns=COL,dtype=float)

    for c,site in enumerate(wtd_obs.columns):

        df_tmp = pd.concat((wtd_obs[site],wtd_mod[site]),axis=1)
        df_tmp.columns = ['data_obs','data_mod']

        bias_site = metrics.bias(df_tmp) # Bias = bias_site[0]
        ubRMSD_site = metrics.ubRMSD(df_tmp) # ubRMSD = ubRMSD_site[0]
        pearson_R_site = metrics.Pearson_R(df_tmp) # Pearson_R = pearson_R_site[0]
        RMSD_site = (ubRMSD_site[0]**2 + bias_site[0]**2)**0.5

        # Save metrics in df_metrics.
        df_metrics.loc[site]['bias'] = bias_site[0]
        df_metrics.loc[site]['ubRMSD'] =  ubRMSD_site[0]
        df_metrics.loc[site]['Pearson_R'] = pearson_R_site[0]
        df_metrics.loc[site]['RMSD'] = RMSD_site

        # Create x-axis matching in situ data.
        x_start = df_tmp.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end = df_tmp.index[-1]   # End a-axis with the last day with an observed wtd value.
        Xlim = [x_start, x_end]

        # Calculate z-score for the time series.
        df_zscore = df_tmp.apply(zscore)

        plt.figure(figsize=(16, 6.5))
        fontsize = 12

        ax1 = plt.subplot(311)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.','-'], linewidth=2, xlim=Xlim)
        plt.ylabel('zbar [m]')

        Title = site + '\n' + ' bias = ' + str(bias_site[0]) + ', ubRMSD = ' + str(ubRMSD_site[0]) + ', Pearson_R = ' + str(pearson_R_site[0]) + ', RMSD = ' + str(RMSD_site)
        plt.title(Title)

        ax2 = plt.subplot(312)
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['.','-'], linewidth=2, xlim=Xlim)
        plt.ylabel('z-score')

        ax3 = plt.subplot(313)
        precip_obs[site].plot(ax=ax3, fontsize=fontsize, style=['.'], linewidth=2, xlim=Xlim)
        plt.ylabel('precipitation [mm/d]')

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

    # Plot boxplot for metrics
    plt.figure()
    df_metrics.boxplot()
    fname = 'metrics'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

def plot_skillmetrics_comparison_wtd_multimodel(wtd_obs, wtd_mod, precip_obs, exp, outpath):

    # Initiate dataframe to store metrics in.
    INDEX = wtd_obs.columns

    for c,site in enumerate(wtd_obs.columns):

        df_tmp = pd.concat((wtd_obs[site], wtd_mod[0][site],wtd_mod[1][site],wtd_mod[2][site]),axis=1)
        df_tmp.columns = ['In-situ','North','Tropics-N','Tropics-D']

        # Create x-axis matching in situ data.
        x_start = df_tmp.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end = df_tmp.index[-1]   # End a-axis with the last day with an observed wtd value.
        Xlim = [x_start, x_end]

        # Calculate z-score for the time series.
        df_zscore = df_tmp.dropna(axis=0).apply(zscore)

        plt.figure(figsize=(16, 6.5))
        fontsize = 12

        ax1 = plt.subplot(311)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.','-','-','-'], linewidth=2, xlim=Xlim)
        plt.ylabel('zbar [m]')

        plt.title(site)

        ax1 = plt.subplot(312)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.','-','-','-'], linewidth=2, xlim=[df_zscore.index[0],df_zscore.index[-1]])
        plt.ylabel('zbar [m]')

        ax2 = plt.subplot(313)
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['.','-','-','-'], linewidth=2, xlim=[df_zscore.index[0],df_zscore.index[-1]])
        plt.ylabel('z-score')

        plt.legend()

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '_multimodel.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

    # Plot boxplot for metrics
    plt.figure()
    df_metrics.boxplot()
    fname = 'metrics'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

def plot_delta_spinup(exp1, exp2, domain, root, outpath):

    # Hugo Rudebeck:
    # Funtion to plot the difference between two runs. Takes the difference between the all the sites for the two runs and
    # saves the maximum difference for every time step. Does this for all variables.

    outpath = os.path.join(outpath, exp1, 'delta_spinup')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io_spinup1 = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    io_spinup2 = LDAS_io('daily', exp=exp2, domain=domain, root=root) # This path is not correct anymore, the long nc cubes for the 2nd spinup are now
    # stored in /ddn1/vol1/staging/leuven/stg_00024/OUTPUT/hugor/output/INDONESIA_M09_v01_spinup2/ as daily_images_2000-2019.nc and daily_timeseries_2000-2019.nc.
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io_spinup1)

    ntimes = len(io_spinup1.images['time'])
    delta_var = np.zeros([ntimes])

    for i,cvar in enumerate(io_spinup1.images.data_vars):   # Loop over all variables in cvar.

        unit = assign_units(cvar)   # Assign units to cvar.

        for t in range(ntimes):
            # logging.info('time %i of %i' % (t, ntimes))
            diff = io_spinup1.images[cvar][t,:,:] - io_spinup2.images[cvar][t,:,:]
            diff_1D = diff.values.ravel()   # Make a 1D array.
            diff_1D_noNaN = diff_1D[~np.isnan(diff_1D)] # Remove nan.
            diff_1D_sorted = np.sort(diff_1D_noNaN.__abs__()) # Sort the absolute values.
            delta_var[t] = diff_1D_sorted[-2]  # Take the second largest absolute value, due to problems with instability
            # in one grid cell causing abnormal values.

            # # searching for strange values evaporation
            # if cvar == 'evap' and A.data.item() > 10 and t > 2000:
            #     # print(t)
            #     print(np.where(np.abs(diff.values) > 10))




        # Plot the maximum difference between the two runs and save figure in 'outpath'.
        plt.plot(delta_var,linewidth=2)
        my_title = cvar
        plt.title(my_title, fontsize=14)
        my_xlabel = 'time step [days]'
        plt.xlabel(my_xlabel, fontsize=12)
        my_ylabel = 'difference' + ' ' + '$\mathregular{' + unit + '}$'
        plt.ylabel(my_ylabel, fontsize=12)
        fname = cvar
        fname_long = os.path.join(outpath, fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

def plot_timeseries_wtd_sfmc(exp, domain, root, outpath, lat=53, lon=25):

    outpath = os.path.join(outpath, exp, 'timeseries')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    fontsize = 12
    # get M09 rowcol with data
    col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
    # get poros for col row
    siteporos = poros[row, col]

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

    my_title = 'lon=%s, lat=%s poros=%.2f' % (lon, lat, siteporos)
    plt.title(my_title, fontsize=fontsize+2)
    plt.tight_layout()
    fname = 'wtd_sfmc_lon_%s_lat_%s' % (lon,lat)
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_peat_and_sites(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps','catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
    #RTMparams = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    # land fraction
    frac_cell = io.grid.tilecoord.frac_cell.values
    param='poros'
    # fraction of peatlands with less than 5 % open water
    frac_peatland = np.nansum(np.all(np.vstack((frac_cell>0.5,params['poros'].values>0.8)),axis=0))/np.nansum(params['poros'].values>0.01)
    frac_peatland_less5 = np.nansum(np.all(np.vstack((frac_cell>0.95,params['poros'].values>0.8)),axis=0))/np.nansum(params['poros'].values>0.8)
    # set more than 5% open water grid cells to 1.
    params[param].values[np.all(np.vstack((frac_cell<0.95,params['poros'].values>0.8)),axis=0)] = 1.
    params[param].values[params['poros'].values<0.8] = np.nan
    params[param].values[np.all(np.vstack((params['poros'].values<0.95,params['poros'].values>0.8)),axis=0)] = 0.0
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params[param].values
    data = np.ma.masked_invalid(img)
    fname='01b_'+param
    cmin = 0
    cmax = 1
    title='Peatland distribution'
    # open figure
    figsize = (0.85*13, 0.85*10)
    fontsize = 13
    f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    cmap = matplotlib.colors.ListedColormap([[255./255,193./255,7./255],[30./255,136./255,229./255]])
    #cmap = 'jet'
    cbrange = (cmin, cmax)
    ax=plt.subplot(3,1,3)
    plt_img=np.ma.masked_invalid(data)
    m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='l')
    #m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=-170.,urcrnrlon=urcrnrlon,resolution='l')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    parallels=np.arange(-80.0,81,5.)
    m.drawparallels(parallels,linewidth=0.5,labels=[True,False,False,False])
    meridians=np.arange(0.,351.,20.)
    m.drawmeridians(meridians,linewidth=0.5,labels=[False,False,False,True])
    m.readshapefile('/data/leuven/317/vsc31786/gis/permafrost/permafrost_boundary', 'permafrost_boundary',linewidth=1.3,color=(0./255,0./255,0./255))
    # load peatland sites
    sites = pd.read_csv('/data/leuven/317/vsc31786/FIG_tmp/00DA/20190228_M09/cluster_radius_2_bog.txt',sep=',')
    #http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
    #lat=48.
    #lon=51.0
    x,y=m(sites['Lon'].values,sites['Lat'].values)
    if np.mean(lats)>40:
        m.plot(x,y,'.',color=(0./255,0./255,0./255),markersize=12,markeredgewidth=1.5,mfc='none')
    else:
        insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
        mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
        filenames = find_files(insitu_path, mastertable_filename)
        if isinstance(find_files(insitu_path, mastertable_filename),str):
            master_table = pd.read_csv(filenames, sep =';')
        else:
            for filename in filenames:
                if filename.endswith('csv'):
                    master_table = pd.read_csv(filename, sep =';')
                    cond = master_table['comparison_yes']==1
                    continue
                else:
                    logging.warning("some files, maybe swp files, exist that start with master table searchstring !")
        x,y=m(master_table['lon'].values[cond],master_table['lat'].values[cond])
        m.plot(x,y,'.',color=(0./255,0./255,0./255),markersize=12,markeredgewidth=1.5,mfc='none')

    im=m.pcolormesh(lons,lats,plt_img,cmap=cmap,latlon=True)
    im.set_clim(vmin=cbrange[0],vmax=cbrange[1])
    #cb=m.colorbar(im,"bottom",size="7%",pad="22%",shrink=0.5)
    #cb=matplotlib.pyplot.colorbar(im)
    im_ratio=np.shape(data)[0]/np.shape(data)[1]
    cb=matplotlib.pyplot.colorbar(im,fraction=0.13*im_ratio,pad=0.02)
    #ticklabs=cb.ax.get_yticklabels()
    #cb.ax.set_yticklabels(ticklabs,ha='right')
    #cb.ax.yaxis.set_tick_params(pad=45)#yournumbermayvary
    #labelsize
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
        t.set_horizontalalignment('right')
        t.set_x(9.0)
    tit=plt.title(title,fontsize=fontsize)
    #matplotlib.pyplot.text(1.0,1.0,mstats[i],horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,fontsize=fontsize)
    fname_long = os.path.join(outpath, fname+'.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
# DA stuff, to be cleaned up after Remote Sensing of Environment paper acceptance

def plot_all_variables_trends(exp, domain, root, outpath):

    # plot linear trends of variables
    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    latmin=40.
    latmax=70.
    lonmin=-170.
    lonmax=-50.
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)


    # mean
    for varname, da in io.images.data_vars.items():
        if varname not in ['evap','Tair','Rainf','zbar']:
            continue
        tmp_data = da

        #JJA_annual = tmp_data.where((tmp_data['time.season'] == 'JJA')).groupby('time.year').max(dim='time')
        JJA_annual = tmp_data.where((tmp_data['time.month'] >= 7) & (tmp_data['time.month'] <= 9)).groupby('time.year').max(dim='time')
        #JJA_annual = tmp_data.where((tmp_data['time.month'] >= 1) & (tmp_data['time.month'] <= 12)).groupby('time.year').max(dim='time')
        vals = JJA_annual.variable.values
        years = JJA_annual.year
        #col, row = io.grid.lonlat2colrow(-107.5, 54.8, domain=True)
        #plt.figure(figsize=(8, 8.5))
        #plt.plot(years,vals[:,row,col],'.')
        #plt.savefig(outpath+'/'+varname+'_x.png',dpi=150)
        vals2 = vals.reshape(len(years), -1)
        mask1 = (np.isnan(vals2)).any(axis=0)==False
        # Do a first-degree polyfit
        idx = np.where((np.isnan(vals2)).any(axis=0)==False)[0]
        #stats.mstats.theilslopes(vals2[:,idx],years)
        regressions_tmp = np.polyfit(years, vals2[:,idx], 1)
        regressions_tmp_lo = np.polyfit(years, vals2[:,idx], 1)
        regressions_tmp_up = np.polyfit(years, vals2[:,idx], 1)
        for counter, value in enumerate(idx):
            ctheil= stats.mstats.theilslopes(vals2[:,value],years, alpha=0.90)
            regressions_tmp_lo[0,counter] = ctheil[2]
            regressions_tmp_up[0,counter] = ctheil[3]
            result = mk.original_test(vals2[:,value])
            #if (regressions_tmp_lo[0,counter] * regressions_tmp_up[0,counter]) >= 0:
            if result[0]!='no trend':
                regressions_tmp[0,counter] = ctheil[0]
            else:
                regressions_tmp[0,counter] = 0.
        regressions = np.zeros([2,vals2.shape[1]])
        regressions[:] = np.nan
        regressions[:,idx]=regressions_tmp
        # Get the coefficients back
        trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
        cmap = plt.get_cmap('PiYG')
        cmin = np.min([np.nanmin(trends),-np.nanmax(trends)])
        cmax = -cmin
        plot_title=varname
        #if varname=='zbar':
        #    cmin=-0.6
        #    cmax=-0.1
        #if varname=='runoff':
        #    cmin=0
        #    cmax=5
        #if varname=='evap':
        #    cmin=0
        #    cmax=5
        plot_title=varname+" "+assign_units(varname)
        fname=varname+'_trend'
        #plot_title='zbar [m]'
        figure_single_default(data=trends,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=plot_title)

def plot_skillmetrics_comparison_wtd_DA(wtd_obs, wtd_mod, precip_obs, exp, outpath):

    # Initiate dataframe to store metrics in.
    INDEX = wtd_obs.columns

    for c,site in enumerate(wtd_obs.columns):

        df_tmp = pd.concat((wtd_obs[site], wtd_mod[0][site],wtd_mod[1][site]),axis=1)
        df_tmp.columns = ['In-situ','OL','DA']

        # Create x-axis matching in situ data.
        x_start = df_tmp.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end = df_tmp.index[-1]   # End a-axis with the last day with an observed wtd value.
        Xlim = [x_start, x_end]

        # Calculate z-score for the time series.
        df_zscore = df_tmp.dropna(axis=0).apply(zscore)

        plt.figure(figsize=(16, 6.5))
        fontsize = 12

        ax1 = plt.subplot(311)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.','-','-'], linewidth=2, xlim=Xlim)
        plt.ylabel('zbar [m]')

        plt.title(site)

        ax1 = plt.subplot(312)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.','-','-'], linewidth=2, xlim=[df_zscore.index[0],df_zscore.index[-1]])
        plt.ylabel('zbar [m]')

        ax2 = plt.subplot(313)
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['.','-','-'], linewidth=2, xlim=[df_zscore.index[0],df_zscore.index[-1]])
        plt.ylabel('z-score')

        plt.legend()

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '_DA.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()


def plot_anomaly_JulyAugust_zbar(exp, domain, root, outpath):

    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    fname='01_GWanomaly_2012'
    io = LDAS_io('daily',exp, domain, root)
    # reference month
    #gs = (pd.to_datetime(io.timeseries['time'].values).month > 6) & (pd.to_datetime(io.timeseries['time'].values).month < 8)
    # get poros grid and bf1 and bf2 parameters
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

    #tmp = io.timeseries['zbar'][gs,:,:].values
    #data = io.timeseries['zbar'][cday,:,:].values - tmp.mean(axis=0)
    ## load anomaly from nc file
    ds = xr.open_dataset("/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/anomaly_JulyAugust_zbar.nc")
    data = ds['anomaly_zbar_JulyAugust'].values
    np.place(data,poros<0.8,np.nan)
    cmap = 'jet_r'
    cmin = -0.5
    cmax = 0.5
    # open figure
    figsize = (10,10)
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=200, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    #m = Basemap(projection='lcc', llcrnrlat=50, urcrnrlat=70, llcrnrlon=60,urcrnrlon=80, resolution='l')
    m = Basemap(projection='laea', width=2000000,height=1500000, lat_ts=50,lat_0=61,lon_0=75., resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    #ign = m.readshapefile('/data/leuven/317/vsc31786/gis/PEATFIRE_Ignitions_Siberia_2012_export', 'PEATFIRE_Ignitions_Siberia_2012_export')
    m.readshapefile('/data/leuven/317/vsc31786/gis/PEATFIRE_Siberia_2012_export', 'PEATFIRE_Siberia_2012_export',linewidth=1.0,color='k')
    #m.plot()
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
    cb.set_label("groundwater anomaly (m), 21 July 2012", fontsize=14)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    #plt.title("groundwater table anomaly (July 2012) and  peatland wildfires", fontsize=fontsize)
    #plt.tight_layout()
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

def plot_RTMparams_old(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
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

def plot_RTMparams_filled(exp, domain, root, outpath):

    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')

    #params[['bh','bv','omega','rgh_hmin','rgh_hmax']].fillna(-9999)
    params.fillna(-9999,inplace=True)
    cond = params.groupby(['bh','bv','omega','rgh_hmin','rgh_hmax']).vegcls.transform(len) > 70
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[cond==False] = np.nan
    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param+'_filled'
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)

def plot_RTMparams_delta(exp1, exp2, domain, root, outpath):

    # plot RTM parameters for LDAS output
    outpath = os.path.join(outpath, exp1, 'maps', 'rtmparams','compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    # setup grid for plots
    io = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    # set up grid
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    # read parameters
    #params_exp1 = LDAS_io(exp=exp1, domain=domain, root='/staging/leuven/stg_00024/OUTPUT/michelb').read_params('RTMparam')
    params_exp1 = LDAS_io(exp=exp1, domain=domain, root=root).read_params('RTMparam')
    params_exp1.fillna(-9999,inplace=True)
    params_exp2 = LDAS_io(exp=exp2, domain=domain, root=root).read_params('RTMparam')
    params_exp2.fillna(-9999,inplace=True)

    for param in params_exp1:
        #if ['bh','bv','omega','rgh_hmin','rgh_hmax'].count(param)==0:
        #    continue
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params_exp2[param].values - params_exp1[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        #param = param+' (CLSM, recal_NORTH + SMOS_v620Tb) - '+param+" (CLSM, L4SM_v001_Lit4_CalD0)"
        param = param+' '+exp2+' - '+param+' '+exp1
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,plot_title=param)


def plot_RTMparams(exp, domain, root, outpath):

    # plot RTM parameters for LDAS output
    # + make plots of calib and filled grid cells as well as give fraction of calibrated cells.
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    # setup grid for plots
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
        # set up grid
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    # read parameters
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params.fillna(-9999,inplace=True)

    # if more than four (random choice, could be one also) 36 km grid cells (4 x 16 (M09) grid cells --> 64) (have exactly the same parameters, this set must be 'filled' one.
    cond_f = params.groupby(['bh','bv','omega','rgh_hmin','rgh_hmax']).vegcls.transform(len) > 16
    cond_filled = cond_f & (params.bh!=-9999) & (params.poros>0.05)
    cond_calib = (cond_f == False) & (params.bh!=-9999) & (params.poros>0.05)
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[(cond_filled==False) & (cond_calib==False)] = np.nan
    for param in params:
        if ['bh','bv','omega','rgh_hmin','rgh_hmax'].count(param)==0:
            continue
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)
    # histograms
    f = plt.figure(num=None, figsize=(10,10), dpi=300, facecolor='w', edgecolor='k')
    i=0
    for param in params:
        if ['bh','bv','omega','rgh_hmin','rgh_hmax'].count(param)==0:
            continue
        i=i+1
        plt.subplot(3,2,i)
        plt.hist(params[param][cond_calib].values)
        plt.title(param)
    fname = os.path.join(outpath, 'histogram_RTM_params.png')
    plt.savefig(fname,dpi=300)

    params.bh[cond_filled] = 0
    params.bh[cond_calib] = 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cond_filled
    cond_filled_2D = img==1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cond_calib
    cond_calib_2D = img==1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params['bh'].values
    data = np.ma.masked_invalid(img)
    fname='00_calib_filled'
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title='filled=0, calib=1',cmap='winter')

    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ############## n_valid_innov
    n_valid_innov = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    cond_valid = n_valid_innov != 0
    n_valid_innov = n_valid_innov.where(cond_valid)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    data_valid = np.copy(data)
    data_valid[cond_valid == False] = np.nan
    data_valid = obs_M09_to_M36(data_valid)
    fname='00_calib_filled_valid'
    figure_single_default(data=data_valid,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title='filled=0, calib=1',cmap='winter')

    print('N(calib) / N(all): %.4f' % (np.size(np.where(cond_calib==True)[0])/(np.size(np.where(cond_calib==True)[0])+np.size(np.where(cond_filled==True)[0]))))
    print('N(calib, valid) / N(all, valid): %.4f' % (np.size(np.where(cond_calib_2D & cond_valid.values)[0])/(np.size(np.where(cond_calib_2D & cond_valid.values)[0])+np.size(np.where(cond_filled_2D & cond_valid.values)[0]))))

    ### same for peat
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')

    #params[['bh','bv','omega','rgh_hmin','rgh_hmax']].fillna(-9999)
    params.fillna(-9999,inplace=True)
    cond_f = params.groupby(['bh','bv','omega','rgh_hmin','rgh_hmax']).vegcls.transform(len) > 16
    cond_filled_peat = cond_f & (params.bh!=-9999) & (params.poros>0.65)
    cond_calib_peat = (cond_f==False) & (params.bh!=-9999) & (params.poros>0.65)
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[(cond_filled_peat==False) & (cond_calib_peat==False)] = np.nan
    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname=param+'_peat'
        figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=param)

    params.bh[cond_filled_peat] = 0
    params.bh[cond_calib_peat] = 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params['bh'].values
    data = np.ma.masked_invalid(img)
    fname='00_peat_calib_filled'
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title='filled=0, calib=1 (peat distribution)',cmap='winter')

    print('N(calib, peat) / N(peat): %.4f' % (np.size(np.where(cond_calib==True)[0])/(np.size(np.where(cond_calib==True)[0])+np.size(np.where(cond_filled==True)[0]))))
    print('N(calib, peat, valid) / N(peat, valid): %.4f' % (np.size(np.where(cond_calib_2D & cond_valid.values)[0])/(np.size(np.where(cond_calib_2D & cond_valid.values)[0])+np.size(np.where(cond_filled_2D & cond_valid.values)[0]))))

    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ############## n_valid_innov
    n_valid_innov = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    cond_valid = n_valid_innov != 0
    n_valid_innov = n_valid_innov.where(n_valid_innov != 0)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    data_valid = np.copy(data)
    data_valid[cond_valid == False] = np.nan
    data_valid = obs_M09_to_M36(data_valid)
    fname='00_peat_calib_filled_valid'
    figure_single_default(data=data_valid,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title='filled=0, calib=1 (peat distribution)',cmap='winter')


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
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    n_valid_incr = ds['n_valid_incr'][:,:].values
    np.place(n_valid_incr,n_valid_incr==0,np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ############## n_valid_innov
    data = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data)))

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



    ############## n_valid_innov_quatro
    data = ds['n_valid_innov'][0,:,:]/N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][1,:,:]/N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][2,:,:]/N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][3,:,:]/N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov_quatro'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))


    ########## innov mean #############
    data = ds['innov_mean'][0,:,:].values
    data0 = obs_M09_to_M36(data)
    data = ds['innov_mean'][1,:,:].values
    data1 = obs_M09_to_M36(data)
    data = ds['innov_mean'][2,:,:].values
    data2 = obs_M09_to_M36(data)
    data = ds['innov_mean'][3,:,:].values
    data3 = obs_M09_to_M36(data)
    cmin = -3.
    cmax = 3.
    fname='innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['innov_mean_H_Asc', 'innov_mean_H_Des', 'innov_mean_V_Asc', 'innov_mean_V_Des']))

    ########## innov std #############
    data = ds['innov_var'][0,:,:].values**0.5
    data0 = obs_M09_to_M36(data)
    data = ds['innov_var'][1,:,:].values**0.5
    data1 = obs_M09_to_M36(data)
    data = ds['innov_var'][2,:,:].values**0.5
    data2 = obs_M09_to_M36(data)
    data = ds['innov_var'][3,:,:].values**0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## innov mean #############
    data = ds['norm_innov_mean'][0,:,:].values
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][1,:,:].values
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][2,:,:].values
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][3,:,:].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='norm_innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['norm_innov_mean_H_Asc [K]', 'norm_innov_mean_H_Des [K]', 'norm_innov_mean_V_Asc [K]', 'norm_innov_mean_V_Des [K]']))

    ########## innov std #############
    data = ds['norm_innov_var'][0,:,:].values**0.5
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][1,:,:].values**0.5
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][2,:,:].values**0.5
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][3,:,:].values**0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='norm_innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['norm_innov_std_H_Asc [K]', 'norm_innov_std_H_Des [K]', 'norm_innov_std_V_Asc [K]', 'norm_innov_std_V_Des [K]']))

    ########## incr std #############
    data = ds['incr_srfexc_var'].values**0.5
    cmin = 0
    cmax = None
    fname='incr_srfexc_std'
    my_title='std(\u0394srfexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_rzexc_var'].values**0.5
    cmin = 0
    cmax = None
    fname='incr_rzexc_std'
    my_title='std(\u0394rzexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_catdef_var'].values**0.5
    cmin = 0
    cmax = 20
    fname='incr_catdef_std'
    my_title='std(\u0394catdef): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

def plot_ensstd_insitu_error(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    n_valid_incr = ds['n_valid_incr'][:,:].values
    np.place(n_valid_incr,n_valid_incr==0,np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ds_ensstd = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    if exp.startswith('CO') or exp.startswith('IN'):
        ncpath_OL = io.paths.root +'/' + exp + '/output_postprocessed/'
        ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')
        ds_ensstd_OL_CLSM = xr.open_dataset(io.paths.root +'/' + 'CONGO_M09_CLSM_v01_SMOSfw_OL' + '/output_postprocessed/ensstd_mean.nc')
        ds_ensstd_DA_CLSM = xr.open_dataset(io.paths.root +'/' + 'CONGO_M09_CLSM_v01_SMOSfw_DA' + '/output_postprocessed/ensstd_mean.nc')
    else:
        ncpath_OL = io.paths.root +'/' + exp[:-3] + '/output_postprocessed/'
        ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')
        ds_ensstd_OL_CLSM = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw/output_postprocessed/ensstd_mean.nc')
        ds_ensstd_DA_CLSM = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA/output_postprocessed/ensstd_mean.nc')
    ### ensstd vs ubRMSD plot
    ensstd_vs_ubRMSD_plot=0
    if ensstd_vs_ubRMSD_plot==1:
        master_table = pd.read_csv('/vsc-hard-mounts/leuven-data/317/vsc31786/FIG_tmp/00DA/20190228_M09/wtd_stats.txt',sep=',')
        if exp.startswith('CO'):
            insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
            mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
            filenames = find_files(insitu_path, mastertable_filename)
            if isinstance(find_files(insitu_path, mastertable_filename),str):
                master_table = pd.read_csv(filenames, sep =';')
            else:
                for filename in filenames:
                    if filename.endswith('csv'):
                        master_table = pd.read_csv(filename, sep =';')
                        continue
                    else:
                        logging.warning("some files, maybe swp files, exist that start with master table searchstring !")
        ensstd_OL = np.zeros([59])
        ensstd_OL_CLSM = np.zeros([59])
        ensstd_DA = np.zeros([59])
        ensstd_DA_CLSM = np.zeros([59])

        for i,site_ID in enumerate(master_table.iloc[:,0]):

            #if not site_ID.startswith('IN') and not site_ID.startswith('BR'):
            #    # Only use sites in Indonesia and Brunei.
            #    continue

            #if blacklist.iloc[:, 0].str.contains(site_ID).any() or master_table.iloc[i,4] == 0:
            #    # If site is on the blacklist (contains bad data), or if "comparison" =  0, then don include that site in the dataframe.
            #    continue

            # Get lat lon from master table for site.
            lon = master_table.iloc[i,3]
            lat = master_table.iloc[i,2]
            if exp.startswith('CO'):
                lon = master_table['lon'][i]
                lat = master_table['lat'][i]
            # Get poros for col row. + Check whether site is in domain, if not 'continue' with next site
            try:
                col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
                siteporos = poros[row, col]
            except:
                print(site_ID + " not in domain.")
                continue

            # Get porosity for site lon lat.
            # Get M09 rowcol with data.
            scol, srow = io.grid.lonlat2colrow(lon, lat, domain=True)
            crow = np.array([1, -1, 1, -1, 0, 0,  2, -2, 2, -2, 2, -2, 0,  0,  3, -3, 3, -3, 3, -3, 3, -3, 0, 0,  4, -4, 4, -4, 4, -4, 4, -4, 4, -4, 0,0])
            ccol = np.array([0, 0,  1, -1, 1, -1, 0, 0,  1, -1, 2, -2, -2, 2,  1, -1, 2, -2, 3, -3, 0,  0, -3, 3, 1, -1, 2, -2, 3, -3, 4, -4, 0, 0, -4,4])
            row = srow
            col = scol
            print("i: "+str(i))
            continue_yes=0
            dp=0
            for p in range(0,70):
                if p==35:
                    crow = crow*(-1)
                    dp = -35
                print("p: "+str(p))
                # Get poros for col row.
                siteporos = poros[row, col]
                print(siteporos)
                if siteporos <= 0.7:
                    row = srow + crow[p+dp]
                    col = scol + ccol[p+dp]
                else:
                    ensstd_OL[i] = ds_ensstd_OL['zbar'].values[row,col]
                    ensstd_DA[i] = ds_ensstd['zbar'].values[row,col]
                    ensstd_OL_CLSM[i] = ds_ensstd_OL_CLSM['zbar'].values[row,col]
                    ensstd_DA_CLSM[i] = ds_ensstd_DA_CLSM['zbar'].values[row,col]
                    break

        ubRMSD_OL = master_table.iloc[:,14]
        ubRMSD_DA = master_table.iloc[:,15]
        ubRMSD_OL_CLSM = master_table.iloc[:,12]
        ubRMSD_DA_CLSM = master_table.iloc[:,13]
        # open figure
        figsize = (5, 5)
        fontsize = 10
        f = plt.figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
        plt.plot(ubRMSD_OL, ensstd_OL, 'o', markersize=7, markeredgecolor='royalblue', markerfacecolor='whitesmoke', label='OL')
        plt.plot(ubRMSD_DA, ensstd_DA, 'o', markersize=7, markeredgecolor='lawngreen', markerfacecolor='whitesmoke', label='DA')
        plt.plot(np.array([0,0.25]),np.array([0,0.25]),'k-')
        plt.xlabel('ubRMSD (m)')
        plt.ylabel('<ensstd> (m)')
        plt.legend()
        fname_long = os.path.join(outpath, 'ubRMSD_ensstd_PEATCLSM.png')
        plt.tight_layout()
        plt.savefig(fname_long, dpi=f.dpi)
        plt.close()
        # open figure
        figsize = (5, 5)
        fontsize = 10
        f = plt.figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
        NSE_OL = 1 - np.sum((ensstd_OL-ubRMSD_OL)**2.0)/np.sum((ubRMSD_OL-np.mean(ubRMSD_OL))**2.0)
        NSE_DA = 1 - np.sum((ensstd_DA-ubRMSD_DA)**2.0)/np.sum((ubRMSD_DA-np.mean(ubRMSD_DA))**2.0)
        NSE_OL_formatted = '%.4f' % (NSE_OL)
        NSE_DA_formatted = '%.4f' % (NSE_DA)
        RMSD_OL = np.nanmean((ensstd_OL-ubRMSD_OL)**2.0)**0.5
        RMSD_DA = np.nanmean((ensstd_DA-ubRMSD_DA)**2.0)**0.5
        RMSD_OL_formatted = '%.4f' % (RMSD_OL)
        RMSD_DA_formatted = '%.4f' % (RMSD_DA)
        RMSD_OL_CLSM = np.nanmean((ensstd_OL_CLSM-ubRMSD_OL_CLSM)**2.0)**0.5
        RMSD_DA_CLSM = np.nanmean((ensstd_DA_CLSM-ubRMSD_DA_CLSM)**2.0)**0.5
        RMSD_OL_CLSM_formatted = '%.4f' % (RMSD_OL_CLSM)
        RMSD_DA_CLSM_formatted = '%.4f' % (RMSD_DA_CLSM)
        #NSE_OL = 1 - np.sum((ensstd_OL-ubRMSD_OL)**2.0)/np.sum((ubRMSD_OL-np.mean(ubRMSD_OL))**2.0)
        #NSE_DA = 1 - np.sum((ensstd_DA-ubRMSD_DA)**2.0)/np.sum((ubRMSD_DA-np.mean(ubRMSD_DA))**2.0)
        plt.plot(ubRMSD_OL, ensstd_OL, 'o', markersize=7, markeredgecolor='royalblue', markerfacecolor='whitesmoke', label='OL (PEATCLSM); ' + 'RMSD = ' + RMSD_OL_formatted)
        plt.plot(ubRMSD_DA, ensstd_DA, 'o', markersize=7, markeredgecolor='lawngreen', markerfacecolor='whitesmoke', label='DA (PEATCLSM); ' + 'RMSD = ' + RMSD_DA_formatted)
        plt.plot(ubRMSD_OL_CLSM, ensstd_OL_CLSM, 'D', markersize=7, markeredgecolor='royalblue', markerfacecolor='whitesmoke', label='OL (CLSM); ' + 'RMSD = ' + RMSD_OL_CLSM_formatted)
        plt.plot(ubRMSD_DA_CLSM, ensstd_DA_CLSM, 'D', markersize=7, markeredgecolor='lawngreen', markerfacecolor='whitesmoke', label='DA (CLSM); ' + 'RMSD = ' + RMSD_DA_CLSM_formatted)
        plt.plot(np.array([0,1.1]),np.array([0,1.1]),'k-')
        plt.xlabel('ubRMSD (m)')
        plt.ylabel('<ensstd> (m)')
        plt.legend()
        fname_long = os.path.join(outpath, 'ubRMSD_ensstd_both.png')
        plt.tight_layout()
        plt.savefig(fname_long, dpi=f.dpi)
        plt.close()

    ########## ensstd OL DA mean #############
    data = ds_ensstd_OL['zbar'][:, :].values
    np.place(data,n_valid_incr<100,np.nan)
    data_p1 = 1000*data
    data = ds_ensstd['zbar'][:, :].values
    np.place(data,n_valid_incr<1,np.nan)
    data_p2 = 1000*data
    data = data_p2-data_p1
    np.place(data,n_valid_incr<1,np.nan)
    data_p3 = data
    np.place(data_p3,(n_valid_incr<1) | (np.isnan(n_valid_incr) | (poros<0.7)),np.nan)
    cmin = ([None,None,-40])
    cmax = ([None,None,40])
    fname='04b_delta_OL_DA_ensstd_zbar_mean_triple'
    #my_title='srfexc: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=([r'$<ensstd(\overline{z}_{WT})>\/(OL\/PEATCLSM)\/[m]$', r'$<ensstd(zbar)>\/(DA\/PEATCLSM)\/[m]$', \
                                       r'$<\!ensstd(\overline{z}_{WT,DA})\!> - <\!ensstd(\overline{z}_{WT,OL})\!>\enspace(mm)$']),mstats=mstats)

    # hist figure
    figsize = (7.0, 6)
    plt.rcParams.update({'font.size': 33})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot( data_p1[(poros>0.7) & (np.isnan(data_p1)==False)],hist=False, \
                           kde_kws={"color": (0.42745098, 0.71372549, 1. ), "lw": 6, "label": "OL"}).set(xlim=(22, 90.))
    h2 = seaborn.distplot(data_p2[(poros>0.7) & (np.isnan(data_p2)==False)],hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "DA"}).set(xlim=(22, 90.))
    plt.ylim(0., 0.10)
    plt.yticks((0.0,0.1))
    plt.xlabel('$ensstd(\overline{z}_{WT})\enspace(mm)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_ensstd_zWT'
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    #plt.savefig(fname_long)

def plot_filter_diagnostics_gs(exp, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics_gs')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    io_daily = LDAS_io('daily', exp=exp, domain=domain, root=root)
    #N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    gs_daily = (pd.to_datetime(io_daily.timeseries['time'].values).month > 7) & (pd.to_datetime(io_daily.timeseries['time'].values).month < 10)
    N_days = io_daily.timeseries['time'][gs_daily].size
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics_gs.nc')
    ds_ensstd = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    ncpath_OL = io.paths.root +'/' + exp[:-3] + '/output_postprocessed/'
    ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')

    n_valid_incr = ds['n_valid_incr'][:,:].values
    np.place(n_valid_incr,n_valid_incr==0,np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ############## n_valid_innov
    data = ds['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0/N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov'
    figure_single_default(data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data),np.nanstd(data)))

    ############## n_valid_innov_quatro
    data = ds['n_valid_innov'][0,:,:]/N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][1,:,:]/N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][2,:,:]/N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][3,:,:]/N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname='n_valid_innov_quatro'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))

    ########## innov mean #############
    data = ds['innov_mean'][0,:,:].values
    data0 = obs_M09_to_M36(data)
    data = ds['innov_mean'][1,:,:].values
    data1 = obs_M09_to_M36(data)
    data = ds['innov_mean'][2,:,:].values
    data2 = obs_M09_to_M36(data)
    data = ds['innov_mean'][3,:,:].values
    data3 = obs_M09_to_M36(data)
    cmin = -3.
    cmax = 3.
    fname='innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['innov_mean_H_Asc', 'innov_mean_H_Des', 'innov_mean_V_Asc', 'innov_mean_V_Des']))

    ########## innov std #############
    data = ds['innov_var'][0,:,:].values**0.5
    data0 = obs_M09_to_M36(data)
    data = ds['innov_var'][1,:,:].values**0.5
    data1 = obs_M09_to_M36(data)
    data = ds['innov_var'][2,:,:].values**0.5
    data2 = obs_M09_to_M36(data)
    data = ds['innov_var'][3,:,:].values**0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## innov mean #############
    data = ds['norm_innov_mean'][0,:,:].values
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][1,:,:].values
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][2,:,:].values
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][3,:,:].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='norm_innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['norm_innov_mean_H_Asc [K]', 'norm_innov_mean_H_Des [K]', 'norm_innov_mean_V_Asc [K]', 'norm_innov_mean_V_Des [K]']))

    ########## innov std #############
    data = ds['norm_innov_var'][0,:,:].values**0.5
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][1,:,:].values**0.5
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][2,:,:].values**0.5
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][3,:,:].values**0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='norm_innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=(['norm_innov_std_H_Asc [K]', 'norm_innov_std_H_Des [K]', 'norm_innov_std_V_Asc [K]', 'norm_innov_std_V_Des [K]']))

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
    data = ds['incr_srfexc_var'].values**0.5
    cmin = 0
    cmax = None
    fname='incr_srfexc_std'
    my_title='std(\u0394srfexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_rzexc_var'].values**0.5
    cmin = 0
    cmax = None
    fname='incr_rzexc_std'
    my_title='std(\u0394rzexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)

    ########## incr std #############
    data = ds['incr_catdef_var'].values**0.5
    cmin = 0
    cmax = 20
    fname='incr_catdef_std'
    my_title='std(\u0394catdef): m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,plot_title=my_title)


def plot_scaling_delta(exp1, exp2, domain, root, outpath):
    exp = exp1
    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    io = LDAS_io('ObsFcstAna',exp,domain,root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # get scaling nc
    ncpath = io.paths.root +'/' + exp1 + '/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'scaling.nc')
    ncpath = io.paths.root +'/' + exp2 + '/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'scaling.nc')

    AscDes = ['A','D']
    data0 = ds1['m_mod_H_%2i'%angle][:,:,:,0].mean(axis=2,skipna=True) - ds2['m_mod_H_%2i'%angle][:,:,:,0].mean(axis=2,skipna=True)
    data1 = ds1['m_mod_H_%2i'%angle][:,:,:,1].mean(axis=2,skipna=True) - ds2['m_mod_H_%2i'%angle][:,:,:,1].mean(axis=2,skipna=True)
    data2 = ds1['m_mod_V_%2i'%angle][:,:,:,0].mean(axis=2,skipna=True) - ds2['m_mod_V_%2i'%angle][:,:,:,0].mean(axis=2,skipna=True)
    data3 = ds1['m_mod_V_%2i'%angle][:,:,:,1].mean(axis=2,skipna=True) - ds2['m_mod_V_%2i'%angle][:,:,:,1].mean(axis=2,skipna=True)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname='delta_mean_scaling'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['H-Asc', 'H-Des','V-Asc', 'V-Des']))



def plot_filter_diagnostics_delta(exp1, exp2, domain, root, outpath):
    #H-Asc
    #H-Des
    #V-Asc
    #V-Des
    outpath = os.path.join(outpath, exp1, 'maps', 'compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    # set up grid
    io = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp1 + '/output_postprocessed/'
    #ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA_old_scaling/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    #ds1e = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    ncpath = io.paths.root +'/' + exp2 + '/output_postprocessed/'
    #ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    #ds2e = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    
    n_valid_innov = np.nansum(ds1['n_valid_innov'][:,:,:].values,axis=0)/2.0
    np.place(n_valid_innov,n_valid_innov==0,np.nan)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    n_valid_incr = ds1['n_valid_incr'][:,:].values
    np.place(n_valid_incr,n_valid_incr==0,np.nan)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    
    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values, ds1['innov_var'][1,:, :].values, ds1['innov_var'][2,:, :].values, ds1['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<1,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values, ds2['innov_var'][1,:, :].values, ds2['innov_var'][2,:, :].values, ds2['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<1,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<1) | (poros<0.7) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([0,0,-15])
    if np.mean(lats)>40:
        cmax = ([270,270,15])
    else:
        cmax = ([30,30,15])
    fname='02b_delta_innov_var_avg_triple'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros>0.7]),np.nanstd(data_p3[poros>0.7]))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[K^2]$', r'$var(O-F)\/(PEATCLSM)\/[K^2]$', \
                                       r'$var(O-F)_{PEATCLSM} - var(O-F)_{CLSM}\enspace(K^2)$']),mstats=mstats)
    # cross plot
    figsize = (7.0, 6)
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    plt.plot(data_p1[poros>0.7],data_p2[poros>0.7],'.')
    plt.xlabel('CLSM')
    plt.ylabel('PEATCLSM')
    plt.xlim(0,50)
    plt.ylim(0,50)
    fname = 'cross_var_O-F'
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats)>40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros>0.7) & (np.isnan(data_p1)==False)],hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1. ), "lw": 6, "label": "CLSM", 'clip': (0.0, 200.0)}).set(xlim=(0, 22.))
    h2 = seaborn.distplot(data_p2[(poros>0.7) & (np.isnan(data_p2)==False)],hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM", 'clip': (0.0, 200.0)}).set(xlim=(0, 22.))
    plt.ylim(0., 0.28)
    plt.xlabel('$var(O-F)\enspace(K^2)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    #plt.show()
    #plt.legend((h1,h2),('CLSM', 'PEATCLSM'))

    #plt.vlines(x = np.nanmean(data_p1[poros>0.7]), ymin=0.0, ymax=0.1,
    #           color = 'red', linestyle = 'dotted', linewidth = 2)

    #plt.vlines(x = np.nanmean(data_p2[poros>0.7]), ymin=0.0, ymax=0.1,
    #            color = 'blue', linestyle = 'dotted', linewidth = 2)
    fname = 'hist_var_O-F'
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    #plt.savefig(fname_long)

    ########## catdef var pct #############
    data = (ds1['incr_catdef_var'][:, :].values +  ds1['incr_rzexc_var'][:, :].values +  ds1['incr_srfexc_var'][:, :].values)**0.5
    np.place(data,n_valid_incr<1,np.nan)
    data_p1 = data
    data = (ds2['incr_catdef_var'][:, :].values +  ds2['incr_rzexc_var'][:, :].values +  ds2['incr_srfexc_var'][:, :].values)**0.5
    np.place(data,n_valid_incr<1,np.nan)
    data_p2 = data
    data = (data_p2 - data_p1) \
           / (data_p1 + data_p2)
    data_p3 = 100*data
    data = (data_p2 - data_p1)
    data_p3 = data
    np.place(data_p3,(n_valid_incr<1) | np.isnan(n_valid_incr) | (poros<0.7),np.nan)
    cmin = ([None,None,-10])
    cmax = ([None,None,10])
    fname='03b_delta_std_incr_catdef_triple'
    #my_title='total_water: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                  'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                  'm = %.1f, s = %.1f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$<std(catdef)\/(CLSM)\/[mm]$',r'$<std(catdef)\/(PEATCLSM)\/[mm]$', \
                                       r'$std(\Delta wtot_{PEATCLSM}) - std(\Delta wtot_{CLSM})\enspace(mm)$']),mstats=mstats)

    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats)>40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros>0.7) & (np.isnan(data_p1)==False)],hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1. ), "lw": 6, "label": "CLSM", 'clip': (0.0, 20.0)}).set(xlim=(0, 16.))
    h2 = seaborn.distplot(data_p2[(poros>0.7) & (np.isnan(data_p2)==False)],hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM", 'clip': (0.0, 20.0)}).set(xlim=(0, 16.))
    plt.ylim(0., 0.33)
    plt.xlabel('$std(\Delta wtot)\enspace(mm)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_std_wtot'
    fname_long = os.path.join(outpath, fname+'.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    #plt.savefig(fname_long)

    ########## n valid innov #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values, ds1['innov_var'][1,:, :].values, ds1['innov_var'][2,:, :].values, ds1['innov_var'][3,:, :].values)),axis=2)
    data = data**0.5
    np.place(data,n_valid_innov<1,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values, ds2['innov_var'][1,:, :].values, ds2['innov_var'][2,:, :].values, ds2['innov_var'][3,:, :].values)),axis=2)
    data = data**0.5
    np.place(data,n_valid_innov<1,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    data_p3 = data_p3**0.5
    np.place(data_p3,(n_valid_innov<1) | (poros<0.7) | (np.isnan(n_valid_innov)),np.nan)

    data = ds2['n_valid_innov'][:, :, :].sum(dim='species',skipna=True)/2.0
    data = data.where(data != 0)
    data_p3 = obs_M09_to_M36(data)

    cmin = ([0,0,0])
    cmax = ([15,15,1000])
    fname='01_n_valid_innov'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[K]$', r'$std(O-F)\/(PEATCLSM)\/[K]$', \
                                       r'Number of assimilated observations (-)']),mstats=mstats)

    ########## innov var #############
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values) \
           / (ds1['innov_var'][0,:, :].values)
    np.place(data,ds1['n_valid_innov'][0, :,:]<1,np.nan)
    data0 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values) \
           / (ds1['innov_var'][1,:, :].values)
    np.place(data,ds1['n_valid_innov'][1,:,:]<1,np.nan)
    data1 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values) \
           / (ds1['innov_var'][2,:, :].values)
    np.place(data,ds1['n_valid_innov'][2,:,:]<1,np.nan)
    data2 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values) \
           / (ds1['innov_var'][3,:, :].values)
    np.place(data,ds1['n_valid_innov'][3,:,:]<1,np.nan)
    data3 = obs_M09_to_M36(data)

    data = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    # mean over peatlands
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    mean_peat = np.nanmean(data[poros>0.7])
    mean_peat_SI = np.nanmean(data[(poros>0.7)&(lons>63)&(lons<87)&(lats>56)&(lats<67)])
    mean_peat_HU = np.nanmean(data[(poros>0.7)&(lons>-95)&(lons<-80)&(lats>50)&(lats<60)])
    print('mean SI and HU')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)
    
    worse = np.where(data[poros>0.7] > 0.0)[0]
    better = np.where(data[poros>0.7] < 0.0)[0]
    np.size(better)/(np.size(better)+np.size(worse))
    cmin = -50
    cmax = 50
    fname='delta_innov_var_pct'
    data = data*100.
    my_title='delta var(O-F): m (peat) = %.1f, s (peat) = %.1f (pct)' % (np.nanmean(data[poros>0.7]),np.nanstd(data[poros>0.7]))
    figure_single_default(data=100*data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=my_title)

    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values, ds1['innov_var'][1,:, :].values, ds1['innov_var'][2,:, :].values, ds1['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values, ds2['innov_var'][1,:, :].values, ds2['innov_var'][2,:, :].values, ds2['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<1) | (np.isnan(n_valid_innov)),np.nan)
    mean_peat = np.nanmean(data_p3[poros>0.7])
    mean_peat_SI = np.nanmean(data_p3[(poros>0.7)&(lons>63)&(lons<87)&(lats>56)&(lats<67)])
    mean_peat_HU = np.nanmean(data_p3[(poros>0.7)&(lons>-95)&(lons<-80)&(lats>50)&(lats<60)])
    print('mean SI and HU')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)
    cmin = ([0,0,None])
    cmax = ([270,270,None])
    fname='02_delta_innov_var_avg_triple'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[K^2]$', r'$var(O-F)\/(PEATCLSM)\/[K^2]$', r'$var(O-F)\/(PEATCLSM) - var(O-F)\/(CLSM)\/[K^2]$']),mstats=mstats)

    ########## norm innov std triple #############
    data = np.nanmean(np.dstack((ds1['norm_innov_var'][0,:, :].values, ds1['norm_innov_var'][1,:, :].values, ds1['norm_innov_var'][2,:, :].values, ds1['norm_innov_var'][3,:, :].values)),axis=2)
    np.place(data,(n_valid_innov<1) | (np.isnan(n_valid_innov)),np.nan)
    data_p1 = obs_M09_to_M36(data)**0.5
    data = np.nanmean(np.dstack((ds2['norm_innov_var'][0,:, :].values, ds2['norm_innov_var'][1,:, :].values, ds2['norm_innov_var'][2,:, :].values, ds2['norm_innov_var'][3,:, :].values)),axis=2)
    np.place(data,(n_valid_innov<1) | (np.isnan(n_valid_innov)),np.nan)
    data_p2 = obs_M09_to_M36(data)**0.5
    data_p3 = (data_p2-1.) - (data_p1-1.)
    np.place(data_p3,(n_valid_innov<1) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([0,0,None])
    cmax = ([3.0,3.0,None])
    fname='delta_norm_innov_std_avg_triple'
    #my_title='norm_innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data_p1[poros>0.7]),np.nanstd(data_p1[poros>0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p2[poros>0.7]),np.nanstd(data_p2[poros>0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p3[poros>0.7]),np.nanstd(data_p3[poros>0.7]))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$stdnorm(O-F)\/(CLSM)\/[K/K]$', r'$stdnorm(O-F)\/(PEATCLSM)\/[K/K]$', r'$(stdnorm(O-F)\/(PEATCLSM) - 1) - (stdnorm(O-F)\/(CLSM) - 1)\/[K/K]$']),mstats=mstats)


    ########## innov std triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values**0.5, ds1['innov_var'][1,:, :].values**0.5, ds1['innov_var'][2,:, :].values**0.5, ds1['innov_var'][3,:, :].values**0.5)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values**0.5, ds2['innov_var'][1,:, :].values**0.5, ds2['innov_var'][2,:, :].values**0.5, ds2['innov_var'][3,:, :].values**0.5)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][0, :, :].values**0.5 - ds1['innov_var'][0,:, :].values**0.5
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values**0.5 - ds1['innov_var'][1,:, :].values**0.5
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values**0.5 - ds1['innov_var'][2,:, :].values**0.5
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values**0.5 - ds1['innov_var'][3,:, :].values**0.5
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<100) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([0,0,None])
    cmax = ([10,10,None])
    fname='delta_innov_std_avg_triple'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[K]$', r'$std(O-F)\/(PEATCLSM)\/[K]$', r'$std(O-F)\/(PEATCLSM) - std(O-F)\/(CLSM)\/[K]$']),mstats=mstats)

    ########## ensstd sfmc mean triple #############
    #data = ds1e['sfmc'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p1 = data
    #data = ds2e['sfmc'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p2 = data
    #data = ds2e['sfmc'][:, :].values - ds1e['sfmc'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p3 = data
    #np.place(data_p3,(n_valid_incr<100) | (np.isnan(n_valid_incr)),np.nan)
    #cmin = ([None,None,None])
    #cmax = ([None,None,None])
    #fname='delta_sfmc_ensstd_mean_triple'
    ##my_title='srfexc: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    #data_all = list([data_p1,data_p2,data_p3])
    #mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    #figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$<ensstd(sfmc)>)\/(CLSM)\/[mm]$', r'$<ensstd(sfmc)>)\/(PEATCLSM)\/[mm]$', r'$<ensstd(sfmc)>)\/(PEATCLSM) - <ensstd(sfmc)>)\/(CLSM)\/[mm]$']), mstats=mstats)

    ########## ensstd total water mean triple #############
    #data = ds1e['total_water'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p1 = data
    #data = ds2e['total_water'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p2 = data
    #data = ds2e['total_water'][:, :].values - ds1e['total_water'][:, :].values
    #np.place(data,n_valid_incr<100,np.nan)
    #data_p3 = data
    #np.place(data_p3,(n_valid_incr<100) | (np.isnan(n_valid_incr)),np.nan)
    #cmin = ([None,None,None])
    #cmax = ([None,None,None])
    #fname='delta_total_water_ensstd_mean_triple'
    ##my_title='total_water: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    #data_all = list([data_p1,data_p2,data_p3])
    #mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    #figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$<ensstd(total\/water)>)\/(CLSM)\/[mm]$', r'$<ensstd(total\/water)>)\/(PEATCLSM)\/[mm]$', r'$<ensstd(total\/water)>)\/(PEATCLSM) - <ensstd(total\/water)>)\/(CLSM)\/[mm]$']),mstats=mstats)



    ########## norm innov mean triple #############
    data = np.nanmean(np.dstack((ds1['norm_innov_mean'][0,:, :].values, ds1['norm_innov_mean'][1,:, :].values, ds1['norm_innov_mean'][2,:, :].values, ds1['norm_innov_mean'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['norm_innov_mean'][0,:, :].values, ds2['norm_innov_mean'][1,:, :].values, ds2['norm_innov_mean'][2,:, :].values, ds2['norm_innov_mean'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][0, :, :].values - ds1['norm_innov_mean'][0,:, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][1, :, :].values - ds1['norm_innov_mean'][1,:, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][2, :, :].values - ds1['norm_innov_mean'][2,:, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][3, :, :].values - ds1['norm_innov_mean'][3,:, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<100) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([-1.5,-1.5,None])
    cmax = ([1.5,1.5,None])
    fname='delta_norm_innov_mean_avg_triple'
    #my_title='norm_innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1[poros>0.7]),np.nanstd(data_p1[poros>0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2[poros>0.7]),np.nanstd(data_p2[poros>0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3[poros>0.7]),np.nanstd(data_p3[poros>0.7]))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$<norm(O-F)>\/(CLSM)\/[K]$', r'$<norm(O-F)>\/(PEATCLSM)\/[K]$', r'$<norm(O-F)>\/(PEATCLSM) - <norm(O-F)>\/(CLSM)\/[K]$']),mstats=mstats)


    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values, ds1['innov_var'][1,:, :].values, ds1['innov_var'][2,:, :].values, ds1['innov_var'][3,:, :].values)),axis=2)
    data = data**0.5
    np.place(data,n_valid_innov<1,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values, ds2['innov_var'][1,:, :].values, ds2['innov_var'][2,:, :].values, ds2['innov_var'][3,:, :].values)),axis=2)
    data = data**0.5
    np.place(data,n_valid_innov<1,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    data_p3 = data_p3**0.5
    np.place(data_p3,(n_valid_innov<1) | (poros<0.7) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([0,0,-4])
    cmax = ([15,15,4])
    fname='02_delta_innov_std_avg_triple'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros>0.7]),np.nanstd(data_p3[poros>0.7]))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[K]$', r'$std(O-F)\/(PEATCLSM)\/[K]$', \
                                       r'$std(O-F)_{PEATCLSM} - std(O-F)_{CLSM}\enspace(K)$']),mstats=mstats)


    ########## innov var triple pct #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0,:, :].values, ds1['innov_var'][1,:, :].values, ds1['innov_var'][2,:, :].values, ds1['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<1,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0,:, :].values, ds2['innov_var'][1,:, :].values, ds2['innov_var'][2,:, :].values, ds2['innov_var'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<1,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0,:, :].values) \
           / (ds1['innov_var'][0,:, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1,:, :].values) \
           / (ds1['innov_var'][1,:, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2,:, :].values) \
           / (ds1['innov_var'][2,:, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3,:, :].values) \
           / (ds1['innov_var'][3,:, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = 100.*np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<1) | (poros<0.7) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([0,0,-60])
    cmax = ([270,270,60])
    fname='02_delta_pct_innov_var_avg_triple'
    #my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros>0.7]),np.nanstd(data_p3[poros>0.7]))])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[K^2]$', r'$var(O-F)\/(PEATCLSM)\/[K^2]$', \
                                       r'$(var(O-F)_{PEATCLSM} - var(O-F)_{CLSM})/var(O-F)_{CLSM}\enspace\times 100\enspace(\%)$']),mstats=mstats)



    ########## innov mean triple #############
    data = np.nanmean(np.dstack((ds1['innov_mean'][0,:, :].values, ds1['innov_mean'][1,:, :].values, ds1['innov_mean'][2,:, :].values, ds1['innov_mean'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_mean'][0,:, :].values, ds2['innov_mean'][1,:, :].values, ds2['innov_mean'][2,:, :].values, ds2['innov_mean'][3,:, :].values)),axis=2)
    np.place(data,n_valid_innov<100,np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][0, :, :].values - ds1['innov_mean'][0,:, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][1, :, :].values - ds1['innov_mean'][1,:, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][2, :, :].values - ds1['innov_mean'][2,:, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][3, :, :].values - ds1['innov_mean'][3,:, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    np.place(data_p3,(n_valid_innov<100) | (np.isnan(n_valid_innov)),np.nan)
    cmin = ([-5,-5,None])
    cmax = ([5,5,None])
    fname='delta_innov_mean_avg_triple'
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1,data_p2,data_p3])
    figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=([r'$<(O-F)>\/(CLSM)\/[K]$', r'$<(O-F)>\/(PEATCLSM)\/[K]$', r'$<(O-F)>\/(PEATCLSM) - <(O-F)>\/(CLSM)\/[K]$']),mstats=mstats)


    ########## n_innov #############
    data = ds2['n_valid_innov'][0, :, :].values - ds1['n_valid_innov'][0,:, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][1, :, :].values - ds1['n_valid_innov'][1,:, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][2, :, :].values - ds1['n_valid_innov'][2,:, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][3, :, :].values - ds1['n_valid_innov'][3,:, :].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_n_valid_innov'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['H_Asc', 'H_Des', 'V_Asc', 'V_Des']))

    ########## innov mean #############
    data = ds2['innov_mean'][0, :, :].values - ds1['innov_mean'][0,:, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][1, :, :].values - ds1['innov_mean'][1,:, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][2, :, :].values - ds1['innov_mean'][2,:, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][3, :, :].values - ds1['innov_mean'][3,:, :].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_innov_mean'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['H_Asc', 'H_Des', 'V_Asc', 'V_Des']))


    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values**0.5 - ds1['innov_var'][0,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][0, :,:]<100,np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values**0.5 - ds1['innov_var'][1,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][1,:,:]<100,np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values**0.5 - ds1['innov_var'][2,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][2,:,:]<100,np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values**0.5 - ds1['innov_var'][3,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][3,:,:]<100,np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))


    ########## catdef std #############
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<0,np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<200,np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<500,np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<1000,np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_catdef_incr_std_quatro'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['n_valid_incr>0', 'n_valid_incr>200', 'n_valid_incr>500', 'n_valid_incr>1000']))

    ########## catdef std #############
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    cmin = None
    cmax = None
    fname='delta_catdef_incr_std'
    my_title='mean delta std: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=my_title)

    ########## catdef std pct #############
    data = 100 * ((ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5) \
           / (ds1['incr_catdef_var'][:, :].values**0.5))
           #/ (0.5*(ds2['incr_catdef_var'][:, :].values**0.5 + ds1['incr_catdef_var'][:, :].values**0.5))
    cmin = -200
    cmax = 200
    fname='delta_catdef_incr_std_pct'
    my_title='mean delta std pct: m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=my_title)

    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values**0.5 - ds1['innov_var'][0,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][0, :,:]<100,np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values**0.5 - ds1['innov_var'][1,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][1,:,:]<100,np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values**0.5 - ds1['innov_var'][2,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][2,:,:]<100,np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values**0.5 - ds1['innov_var'][3,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][3,:,:]<100,np.nan)
    data3 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((data0,data1,data2,data3)),axis=2)
    cmin = None
    cmax = None
    fname='delta_innov_std'
    my_title='delta innov std: m = %.2f, s = %.2f [mm]' % (np.nanmean(data),np.nanstd(data))
    figure_single_default(data=data,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=my_title)




def plot_daily_delta(exp1, exp2, domain, root, outpath):
    outpath = os.path.join(outpath, exp1, 'maps', 'compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root +'/' + exp1 + '/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ncpath = io.paths.root +'/' + exp2 + '/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values**0.5 - ds1['innov_var'][0,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][0, :,:]<100,np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values**0.5 - ds1['innov_var'][1,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][1,:,:]<100,np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values**0.5 - ds1['innov_var'][2,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][2,:,:]<100,np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values**0.5 - ds1['innov_var'][3,:, :].values**0.5
    np.place(data,ds1['n_valid_innov'][3,:,:]<100,np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_innov_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))


    ########## innov std #############
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<0,np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<200,np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<500,np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values**0.5 - ds1['incr_catdef_var'][:, :].values**0.5
    np.place(data,ds1['n_valid_incr'][:,:]<1000,np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname='delta_catdef_incr_std'
    #my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0,data1,data2,data3])
    figure_quatro_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
                          plot_title=(['delta_catdef_incr_std', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))



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
        meridians = np.arange(0.,351.,20.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        #         http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
        #lat=48.
        #lon=51.0
        #x,y = m(lon, lat)
        #m.plot(x, y, 'ko', markersize=6,mfc='none')
        if fname == '':
            bounds = np.array([-2., -1., -0.1, 0.1, 1., 2.])
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = m.pcolormesh(lons, lats, plt_img, norm=norm, cmap=cmap, latlon=True)
            cb = m.colorbar(im, "bottom", size="7%", pad="15%", extend='both')
        else:
            # color bar
            im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
            im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
            cb = m.colorbar(im, "bottom", size="7%", pad="22%")
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

def figure_triple_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                          llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title,mstats):
    # open figure
    if np.mean(lats)>40:
        figsize = (0.85*13, 0.85*10)
    else:
        figsize = (0.85*7, 0.85*16)
    fontsize = 13
    f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    for i in np.arange(0,3):
        if np.size(cmin)>1:
            cmin_ = cmin[i]
        else:
            cmin_ = cmin
        if np.size(cmax)>1:
            cmax_ = cmax[i]
        else:
            cmax_ = cmax
        cmap = 'jet'
        if (cmin_ == None) | (cmin_ == -9999):
            cmin_ = np.nanmin(data[i])
        if (cmax_ == None) | (cmax_ == -9999):
            cmax_ = np.nanmax(data[i])
        if cmin_ < 0.0:
            cmax_ = np.max([-cmin_,cmax_])
            cmin_ = -cmax_
            cmap = 'seismic'
        cbrange = (cmin_, cmax_)
        ax = plt.subplot(3,1,i+1)
        plt_img = np.ma.masked_invalid(data[i])
        #m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=-170.,urcrnrlon=urcrnrlon,resolution='l')
        m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='l')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        parallels=np.arange(-80.0,81,5.)
        m.drawparallels(parallels,linewidth=0.5,labels=[True,False,False,False])
        meridians=np.arange(0.,351.,20.)
        m.drawmeridians(meridians,linewidth=0.5,labels=[False,False,False,True])
        #         http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
        #lat=48.
        #lon=51.0
        #x,y = m(lon, lat)
        #m.plot(x, y, 'ko', markersize=6,mfc='none')
        if fname == '':
            bounds = np.array([-2., -1., -0.1, 0.1, 1., 2.])
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = m.pcolormesh(lons, lats, plt_img, norm=norm, cmap=cmap, latlon=True)
            cb = m.colorbar(im, "bottom", size="7%", pad="15%", extend='both')
        else:
            # color bar
            im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
            im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
            #cb = m.colorbar(im, "bottom", size="7%", pad="22%", shrink=0.5)
            #cb = matplotlib.pyplot.colorbar(im)
            if np.mean(lats)>40:
                im_ratio = np.shape(data)[1]/np.shape(data)[2]
                cb = matplotlib.pyplot.colorbar(im,fraction=0.13*im_ratio, pad=0.02)
            else:
                cb = matplotlib.pyplot.colorbar(im)
            #ticklabs = cb.ax.get_yticklabels()
            #cb.ax.set_yticklabels(ticklabs,ha='right')
            #cb.ax.yaxis.set_tick_params(pad=45)  # your number may vary
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
            t.set_horizontalalignment('right')
            if np.mean(lats)>40:
                t.set_x(9.0)
            else:
                t.set_x(4.0)
        tit = plt.title(plot_title[i], fontsize=fontsize)
        if np.mean(lats)>40:
            matplotlib.pyplot.text(1.0, 1.0, mstats[i], horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
        else:
            matplotlib.pyplot.text(1.0, 0.2, mstats[i], bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
    fname_long = os.path.join(outpath, fname+'.png')
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

def figure_single_default(data,lons,lats,cmin,cmax,llcrnrlat, urcrnrlat,
                              llcrnrlon,urcrnrlon,outpath,exp,fname,plot_title,cmap='seismic'):
    #if plot_title.startswith('zbar'):
    #    cmap = 'jet_r'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < 0.0:
        cmax = np.max([-cmin,cmax])
        cmin = -cmax
        cmap = 'seismic'
    # open figure
    # Norther peatland:
    if np.mean(lats)>30:
        fig_aspect_ratio = (0.1*(np.max(lons)-np.min(lons)))/(0.18*(np.max(lats)-np.min(lats)))
        figsize = (fig_aspect_ratio+10,10)
        parallels = np.arange(-80.0,81,5.)
        meridians = np.arange(0.,351.,20.)
    else:
        figsize = (10,10)
        parallels = np.arange(-80.0,81,np.round((np.max(lats)-np.min(lats))/4))
        meridians = np.arange(0.,351.,np.round((np.max(lons)-np.min(lons))/4))
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(parallels,labels=[False,True,True,False])
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    #lat=53.
    #lon=38.5
    #x,y = m(lon, lat)
    #m.plot(x, y, 'ko', markersize=6,mfc='none')
    #cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "bottom", size="6%", pad="22%",shrink=0.5)
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

def plot_timeseries(exp1, exp2, domain, root, outpath, lat=53, lon=25):
    exp = exp1
    io_daily_PCLSM = LDAS_io('daily', exp=exp, domain=domain, root=root)
    io_daily_CLSM = LDAS_io('daily', exp=exp2, domain=domain, root=root)
    root2='/staging/leuven/stg_00024/OUTPUT/michelb'
    root2=root
    io_daily_PCLSM_OL = LDAS_io('daily', exp=exp[0:-3], domain=domain, root=root2)
    io_daily_CLSM_OL = LDAS_io('daily', exp=exp2[0:-3], domain=domain, root=root2)

    #io = LDAS_io('ObsFcstAna', exp='SMAP_EASEv2_M09_SMOSfw', domain=domain, root=root)
    #io_CLSM = LDAS_io('ObsFcstAna', exp='SMAP_EASEv2_M09_CLSM_SMOSfw', domain=domain, root=root)
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    io_CLSM = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    io_incr = LDAS_io('incr', exp=exp, domain=domain, root=root)
    io_incr_CLSM = LDAS_io('incr', exp=exp2, domain=domain, root=root)
    #io_ens = LDAS_io('ensstd', exp=exp, domain=domain, root=root)
    #io_ens_CLSM = LDAS_io('ensstd', exp=exp2, domain=domain, root=root)
    [lons,lats,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon] = setup_grid_grid_for_plot(io)
    [col, row] = get_M09_ObsFcstAna(io,lon,lat)

    dcols = [-2,-1,0,1,2]
    drows = [-2,-1,0,1,2]
    col_ori = col
    row_ori = row
    for icol,dcol in enumerate(dcols):
        for icol,drow in enumerate(drows):
            col = col_ori+dcol*4
            row = row_ori+drow*4
            # ObsFcstAna PCLSM
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

            ts_ana_1 = io.read_ts('obs_ana', col, row, species=1, lonlat=False)
            ts_ana_1.name = 'Tb ana (species=1)'
            ts_ana_2 = io.read_ts('obs_ana', col, row, species=2, lonlat=False)
            ts_ana_2.name = 'Tb ana (species=2)'
            ts_ana_3 = io.read_ts('obs_ana', col, row, species=3, lonlat=False)
            ts_ana_3.name = 'Tb ana (species=3)'
            ts_ana_4 = io.read_ts('obs_ana', col, row, species=4, lonlat=False)
            ts_ana_4.name = 'Tb ana (species=4)'

            ts_assim_1 = io.read_ts('obs_assim', col, row, species=1, lonlat=False)
            ts_assim_1.name = 'a1'
            ts_assim_2 = io.read_ts('obs_assim', col, row, species=2, lonlat=False)
            ts_assim_2.name = 'a2'
            ts_assim_3 = io.read_ts('obs_assim', col, row, species=3, lonlat=False)
            ts_assim_3.name = 'a3'
            ts_assim_4 = io.read_ts('obs_assim', col, row, species=4, lonlat=False)
            ts_assim_4.name = 'a4'

            ts_innov_1 = ts_obs_1 - ts_fcst_1
            ts_innov_1[ts_assim_1!=-1] = np.nan
            ts_innov_1.name = 'innov (species=1,PCLSM)'
            ts_innov_2 = ts_obs_2 - ts_fcst_2
            ts_innov_2[ts_assim_2!=-1] = np.nan
            ts_innov_2.name = 'innov (species=2,PCLSM)'
            ts_innov_3 = ts_obs_3 - ts_fcst_3
            ts_innov_3[ts_assim_3!=-1] = np.nan
            ts_innov_3.name = 'innov (species=3,PCLSM)'
            ts_innov_4 = ts_obs_4 - ts_fcst_4
            ts_innov_4[ts_assim_4!=-1] = np.nan
            ts_innov_4.name = 'innov (species=4,PCLSM)'

            # incr PCLSM
            ts_incr_catdef = io_incr.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            ts_incr_catdef.name = 'catdef incr'
            ts_incr_srfexc = io_incr.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            ts_incr_srfexc.name = 'srfexc incr'

            # ensstd PCLSM
            #ts_ens_catdef = io_ens.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            #ts_ens_catdef.name = 'catdef'
            #ts_ens_srfexc = io_ens.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            #ts_ens_srfexc.name = 'srfexc'

            # daily PCLSM
            ts_catdef_PCLSM = io_daily.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_PCLSM.name = 'analysis (PCLSM)'
            ts_catdef_PCLSM_OL = io_daily_OL.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_PCLSM_OL.name = 'open loop (PCLSM)'

            # ObsFcstAna CLSM
            ts_CLSM_obs_1 = io_CLSM.read_ts('obs_obs', col, row, species=1, lonlat=False)
            ts_CLSM_obs_1.name = 'Tb obs (species=1,CLSM)'
            ts_CLSM_obs_2 = io_CLSM.read_ts('obs_obs', col, row, species=2, lonlat=False)
            ts_CLSM_obs_2.name = 'Tb obs (species=2,CLSM)'
            ts_CLSM_obs_3 = io_CLSM.read_ts('obs_obs', col, row, species=3, lonlat=False)
            ts_CLSM_obs_3.name = 'Tb obs (species=3,CLSM)'
            ts_CLSM_obs_4 = io_CLSM.read_ts('obs_obs', col, row, species=4, lonlat=False)
            ts_CLSM_obs_4.name = 'Tb obs (species=4,CLSM)'

            ts_CLSM_fcst_1 = io_CLSM.read_ts('obs_fcst', col, row, species=1, lonlat=False)
            ts_CLSM_fcst_1.name = 'Tb fcst (species=1,CLSM)'
            ts_CLSM_fcst_2 = io_CLSM.read_ts('obs_fcst', col, row, species=2, lonlat=False)
            ts_CLSM_fcst_2.name = 'Tb fcst (species=2,CLSM)'
            ts_CLSM_fcst_3 = io_CLSM.read_ts('obs_fcst', col, row, species=3, lonlat=False)
            ts_CLSM_fcst_3.name = 'Tb fcst (species=3,CLSM)'
            ts_CLSM_fcst_4 = io_CLSM.read_ts('obs_fcst', col, row, species=4, lonlat=False)
            ts_CLSM_fcst_4.name = 'Tb fcst (species=4,CLSM)'

            ts_CLSM_ana_1 = io_CLSM.read_ts('obs_ana', col, row, species=1, lonlat=False)
            ts_CLSM_ana_1.name = 'Tb ana (species=1,CLSM)'
            ts_CLSM_ana_2 = io_CLSM.read_ts('obs_ana', col, row, species=2, lonlat=False)
            ts_CLSM_ana_2.name = 'Tb ana (species=2,CLSM)'
            ts_CLSM_ana_3 = io_CLSM.read_ts('obs_ana', col, row, species=3, lonlat=False)
            ts_CLSM_ana_3.name = 'Tb ana (species=3,CLSM)'
            ts_CLSM_ana_4 = io_CLSM.read_ts('obs_ana', col, row, species=4, lonlat=False)
            ts_CLSM_ana_4.name = 'Tb ana (species=4,CLSM)'

            ts_CLSM_assim_1 = io_CLSM.read_ts('obs_assim', col, row, species=1, lonlat=False)
            ts_CLSM_assim_1.name = 'a1 (CLSM)'
            ts_CLSM_assim_2 = io_CLSM.read_ts('obs_assim', col, row, species=2, lonlat=False)
            ts_CLSM_assim_2.name = 'a2 (CLSM)'
            ts_CLSM_assim_3 = io_CLSM.read_ts('obs_assim', col, row, species=3, lonlat=False)
            ts_CLSM_assim_3.name = 'a3 (CLSM)'
            ts_CLSM_assim_4 = io_CLSM.read_ts('obs_assim', col, row, species=4, lonlat=False)
            ts_CLSM_assim_4.name = 'a4 (CLSM)'

            ts_CLSM_innov_1 = ts_CLSM_obs_1 - ts_CLSM_fcst_1
            ts_CLSM_innov_1[ts_CLSM_assim_1!=-1] = np.nan
            ts_CLSM_innov_1.name = 'innov (species=1,CLSM)'
            ts_CLSM_innov_2 = ts_CLSM_obs_2 - ts_CLSM_fcst_2
            ts_CLSM_innov_2[ts_CLSM_assim_2!=-1] = np.nan
            ts_CLSM_innov_2.name = 'innov (species=2,CLSM)'
            ts_CLSM_innov_3 = ts_CLSM_obs_3 - ts_CLSM_fcst_3
            ts_CLSM_innov_3[ts_CLSM_assim_3!=-1] = np.nan
            ts_CLSM_innov_3.name = 'innov (species=3,CLSM)'
            ts_CLSM_innov_4 = ts_CLSM_obs_4 - ts_CLSM_fcst_4
            ts_CLSM_innov_4[ts_CLSM_assim_4!=-1] = np.nan
            ts_CLSM_innov_4.name = 'innov (species=4,CLSM)'

            # incr CLSM
            ts_incr_CLSM_catdef = io_incr_CLSM.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            ts_incr_CLSM_catdef.name = 'catdef incr (CLSM)'
            ts_incr_CLSM_srfexc = io_incr_CLSM.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            ts_incr_CLSM_srfexc.name = 'srfexc incr (CLSM)'

            # ensstd PCLSM
            #ts_ens_CLSM_catdef = io_ens_CLSM.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            #ts_ens_CLSM_catdef.name = 'catdef (CLSM)'
            #ts_ens_CLSM_srfexc = io_ens_CLSM.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            #ts_ens_CLSM_srfexc.name = 'srfexc (CLSM)'

            # daily CLSM
            ts_catdef_CLSM = io_daily_CLSM.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_CLSM.name = 'analysis (CLSM)'
            ts_catdef_CLSM_OL = io_daily_CLSM_OL.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_CLSM_OL.name = 'open loop (CLSM)'

            df = pd.concat((ts_obs_1, ts_CLSM_obs_1, ts_innov_1, ts_innov_2, ts_innov_3, ts_innov_4, ts_incr_catdef,
                            ts_ana_1, ts_ana_2, ts_ana_3, ts_ana_4,ts_assim_1,ts_assim_2,ts_assim_3,ts_assim_4,
                            ts_catdef_PCLSM, ts_catdef_PCLSM_OL,
                            ts_CLSM_innov_1, ts_CLSM_innov_2, ts_CLSM_innov_3, ts_CLSM_innov_4, ts_incr_CLSM_catdef,
                            ts_CLSM_ana_1, ts_CLSM_ana_2, ts_CLSM_ana_3, ts_CLSM_ana_4,ts_CLSM_assim_1,ts_CLSM_assim_2,ts_CLSM_assim_3,ts_CLSM_assim_4,
                            ts_catdef_CLSM, ts_catdef_CLSM_OL),axis=1)
            #df = pd.concat((ts_obs_1, ts_obs_2, ts_obs_3, ts_obs_4, ts_fcst_1, ts_fcst_2, ts_fcst_3, ts_fcst_4, ts_incr_catdef),axis=1).dropna()
            #mask = (df.index > '2016-05-01 00:00:00') & (df.index <= '2016-06-01 00:00:00')
            #df = df.loc[mask]
            plt.figure(figsize=(19,7))
            fontsize = 12

            ax1 = plt.subplot(311)
            #df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            #df[['innov (species=1,PCLSM)','innov (species=2,PCLSM)','innov (species=3,PCLSM)','innov (species=4,PCLSM)',
            #    'innov (species=1,CLSM)','innov (species=2,CLSM)','innov (species=3,CLSM)','innov (species=4,CLSM)',]].plot(ax=ax1,
            #    fontsize=fontsize, style=['o','o','o','o','x','x','x','x'], linewidth=2)
            df[['Tb obs (species=1)','Tb obs (species=1,CLSM)']].plot(ax=ax1,
                fontsize=fontsize, style=['o','x'], linewidth=2)
            plt.ylabel('innov (O-F) [K]')

            ax2 = plt.subplot(312)
            df[['catdef incr']].plot(ax=ax2, fontsize=fontsize, style=['o:'], linewidth=2)
            plt.ylabel('incr [mm]')
            my_title = 'lon=%s, lat=%s' % (lon,lat)
            plt.title(my_title, fontsize=fontsize+2)

            #ax3 = plt.subplot(413)
            #df[['a1','a2','a3','a4']].plot(ax=ax3, fontsize=fontsize, style=['.-','.-','.-','.-'], linewidth=2)
            #plt.ylabel('Tb [K]')

            ax4 = plt.subplot(313)
            #df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            df[['analysis (PCLSM)','analysis (CLSM)','open loop (PCLSM)','open loop (CLSM)']].plot(ax=ax4, fontsize=fontsize, style=['.-','.-','.-','.-'], linewidth=2)
            plt.ylabel('catdef [mm]')

            #ax5 = plt.subplot(313)
            ##df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            #df[['srfexc','srfexc (CLSM)']].plot(ax=ax5, fontsize=fontsize, style=['.-','x-'], linewidth=2)
            #plt.ylim(bottom=0)
            #plt.ylabel('ensstd [mm]')

            plt.tight_layout()
            fname = 'innov_incr_col_%s_row_%s' % (col,row)
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

    #### mean_obs
    data0 = ds['m_obs_H_%2i'%angle][:,:,:,0].mean(axis=2)
    data1 = ds['m_obs_H_%2i'%angle][:,:,:,1].mean(axis=2)
    data2 = ds['m_obs_V_%2i'%angle][:,:,:,0].mean(axis=2)
    data3 = ds['m_obs_V_%2i'%angle][:,:,:,1].mean(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 240
    cmax = 285
    fname='mean_obs'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['H-Asc', 'H-Des','V-Asc', 'V-Des']))

    #### mean_mod
    data0 = ds['m_mod_H_%2i'%angle][:,:,:,0].mean(axis=2)
    data1 = ds['m_mod_H_%2i'%angle][:,:,:,1].mean(axis=2)
    data2 = ds['m_mod_V_%2i'%angle][:,:,:,0].mean(axis=2)
    data3 = ds['m_mod_V_%2i'%angle][:,:,:,1].mean(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 240
    cmax = 285
    fname='mean_mod'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['H-Asc', 'H-Des','V-Asc', 'V-Des']))

    #### mean_mod - mean_obs
    data0 = ds['m_mod_H_%2i'%angle][:,:,:,0].mean(axis=2) - ds['m_obs_H_%2i'%angle][:,:,:,0].mean(axis=2)
    data1 = ds['m_mod_H_%2i'%angle][:,:,:,1].mean(axis=2) - ds['m_obs_H_%2i'%angle][:,:,:,1].mean(axis=2)
    data2 = ds['m_mod_V_%2i'%angle][:,:,:,0].mean(axis=2) - ds['m_obs_V_%2i'%angle][:,:,:,0].mean(axis=2)
    data3 = ds['m_mod_V_%2i'%angle][:,:,:,1].mean(axis=2) - ds['m_obs_V_%2i'%angle][:,:,:,1].mean(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname='mean_mod-mean_obs'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['H-Asc', 'H-Des','V-Asc', 'V-Des']))

    #### mean_mod - mean_obs
    data0 = (ds['m_mod_H_%2i'%angle][:,:,:,0] - ds['m_obs_H_%2i'%angle][:,:,:,0]).mean(axis=2)
    data1 = (ds['m_mod_H_%2i'%angle][:,:,:,1] - ds['m_obs_H_%2i'%angle][:,:,:,1]).mean(axis=2)
    data2 = (ds['m_mod_V_%2i'%angle][:,:,:,0] - ds['m_obs_V_%2i'%angle][:,:,:,0]).mean(axis=2)
    data3 = (ds['m_mod_V_%2i'%angle][:,:,:,1] - ds['m_obs_V_%2i'%angle][:,:,:,1]).mean(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname='mean_mod-obs'
    data_all = list([data0,data1,data2,data3])
    figure_quatro_scaling(data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp,fname=fname,
                          plot_title=list(['H-Asc', 'H-Des','V-Asc', 'V-Des']))

    AscDes = ['A','D']
    data0 = ds['m_obs_H_%2i'%angle][:,:,:,0].count(axis=2)
    data1 = ds['m_obs_H_%2i'%angle][:,:,:,1].count(axis=2)
    data2 = ds['m_obs_V_%2i'%angle][:,:,:,0].count(axis=2)
    data3 = ds['m_obs_V_%2i'%angle][:,:,:,1].count(axis=2)
    data0 = data0.where(data0!=0)
    data1 = data1.where(data1!=0)
    data2 = data2.where(data2!=0)
    data3 = data3.where(data3!=0)
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

if __name__=='__main__':
    plot_ismn_statistics()
