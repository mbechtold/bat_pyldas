import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')
    matplotlib.use('QT4Agg')
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from bat_pyldas.functions import *
from scipy.stats import zscore
import scipy.interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
from scipy.interpolate import interp2d
from validation_good_practice.ancillary import metrics
import sys
from scipy import stats
import pymannkendall as mk
import copy
from sympy import symbols, diff
import statsmodels.api as sm


# gdal, gdalconst, osr

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
        unit = ['not_known']
    return (unit)


def plot_all_variables_temporal_moments(exp, domain, root, outpath, param='daily'):
    # plot temporal mean and standard deviation of variables
    outpath = os.path.join(outpath, exp, 'maps', 'stats')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io = LDAS_io(param, exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    try:
        m1 = xr.open_dataset(os.path.join(root, exp, 'output_postprocessed/', param + '_mean.nc'))
    except:
        m1 = io.timeseries.mean(axis=0)
    try:
        m2 = xr.open_dataset(os.path.join(root, exp, 'output_postprocessed/', param + '_std.nc'))
    except:
        m2 = io.timeseries.std(axis=0)
    # mean
    # m1 = io.timeseries.mean(axis=0)
    for varname, da in m1.data_vars.items():
        tmp_data = da
        # cmin = 0
        # cmax = 0.7
        cmin = None
        cmax = None
        plot_title = varname
        if varname == 'zbar':
            cmin = -1.2
            cmax = -0.05
        if varname == 'runoff':
            cmin = 0
            cmax = 5
        if varname == 'evap':
            cmin = 0
            cmax = 5
        # if varname=='tp1':
        #    cmin=267
        #    cmax=295
        # if varname=='fsw_change':
        #    tmp_data = tmp_data*365*6*24*60*60
        plot_title = varname
        # if varname=='fsw_change':
        #    plot_title='cumulative sum of fsw_change [mm]'
        fname = varname + '_mean'
        # plot_title='zbar [m]'
        figure_single_default(data=tmp_data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=plot_title)
    # m2 = io.timeseries.std(axis=0)
    for varname, da in m2.data_vars.items():
        tmp_data = da
        # cmin = 0
        # cmax = 0.7
        cmin = None
        cmax = None
        fname = varname + '_std'
        # plot_title='zbar [m]'
        plot_title = varname
        figure_single_default(data=tmp_data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=plot_title)


def plot_all_temporal_maps(exp, domain, root, outpath, param='daily'):
    # plot temporal mean and standard deviation of the paper variables
    def create_figure_maps(exp, domain, root, outpath):
        outpath = os.path.join(outpath, 'maps')
        if not os.path.exists(outpath):
            os.makedirs(outpath, exist_ok=True)

        io = LDAS_io(param, exp=exp, domain=domain, root=root)
        [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

        # m1 is the mean nc file
        try:
            m1 = xr.open_dataset(os.path.join(root, exp, 'output_postprocessed/', param + '_mean.nc'))
        except:
            m1 = io.timeseries.mean(axis=0)
        # m2 is the std nc files
        try:
            m2 = xr.open_dataset(os.path.join(root, exp, 'output_postprocessed/', param + '_std.nc'))
        except:
            m2 = io.timeseries.std(axis=0)

        # read in the catparam to use poros as a filter and creat masked data array of poros
        params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
        # land fraction, if more than 0
        frac_cell = io.grid.tilecoord.frac_cell.values
        par = 'poros'
        tc = io.grid.tilecoord
        tg = io.grid.tilegrids

        params[par].values[np.all(np.vstack((frac_cell < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0
        params[par].values[params['poros'].values < 0.65] = np.nan
        params[par].values[
            np.all(np.vstack((params['poros'].values < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[par].values
        data = np.ma.masked_invalid(img)
        maskt = np.ma.getmask(data)

        if exp[:2] == 'IN':
            figsize = (30, 20)
        else:
            figsize = (25, 25)

        runoff = m1.data_vars['runoff']
        Rainf = m1.data_vars['Rainf']
        lhflux = m1.data_vars['lhflux']
        LWdown = m1.data_vars['LWdown']
        SWdown = m1.data_vars['SWdown']
        lwup = m1.data_vars['lwup']
        swup = m1.data_vars['swup']
        shflux = m1.data_vars['shflux']
        Rnet = (LWdown + SWdown) - (swup + lwup)
        RP_eff = runoff / Rainf
        EReff = lhflux / Rnet
        BR = shflux / lhflux
        zbar = m1.data_vars['zbar']
        sfmc = m1.data_vars['sfmc']

        variables = ['zbar', 'sfmc', 'RP_eff', 'EReff', 'BR', 'runoff', 'Rainf', 'Rnet', 'lhflux']

        # plots for zbar and
        # variables = ['zbar', 'sfmc']
        for i in variables:
            da = eval(i)
            varname = i
            tmp_data = da

            masked_var = np.ma.masked_array(da, mask=maskt)
            mean = np.mean(masked_var)
            mean = np.round(mean, 2)
            sd = np.std(masked_var)
            sd = np.round(sd, 2)

            plt_img = np.ma.masked_invalid(masked_var)
            fname = exp + '_' + varname + '_mean'
            plot_title = varname + '_mean, m = ' + str(mean) + ', sd = ' + str(sd)
            outpath_mean = outpath + '_mean'

            if varname == 'zbar':
                if ('TN' in exp) or ('TD' in exp):
                    cmin = -1.2
                    cmax = 0.0
                else:
                    cmin = -5.0
                    cmax = 1.0
            elif varname == 'sfmc':
                if ('TN' in exp) or ('TD' in exp):
                    cmin = 0.4
                    cmax = 0.8
                else:
                    cmin = 0.4
                    cmax = 0.8
            elif varname == 'RP_eff':
                cmin = 0.0
                cmax = 0.7
            elif varname == 'EReff':
                cmin = 0.4
                cmax = 1.0
            elif varname == 'BR':
                cmin = 0.0
                cmax = 0.6
            else:
                cmin = None
                cmax = None

            if varname == 'BR':
                figure_single_default(data=plt_img, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                                      urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                                      outpath=outpath_mean, exp=exp, fname=fname, plot_title=plot_title,
                                      cmap='coolwarm')
            else:
                figure_single_default(data=plt_img, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                                      urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                                      outpath=outpath_mean, exp=exp, fname=fname, plot_title=plot_title)
        runoff = m2.data_vars['runoff']
        Rainf = m2.data_vars['Rainf']
        evap = m2.data_vars['evap']
        LWdown = m2.data_vars['LWdown']
        SWdown = m2.data_vars['SWdown']
        lwup = m2.data_vars['lwup']
        swup = m2.data_vars['swup']
        shflux = m2.data_vars['shflux']
        Rnet = (LWdown + SWdown) - (swup + lwup)
        RP_eff = runoff / Rainf
        EReff = evap / Rnet
        BR = shflux / evap
        zbar = m2.data_vars['zbar']
        sfmc = m2.data_vars['sfmc']

        variables = ['zbar', 'sfmc']

        # standard deviation plots only for sfmc and WTD
        for i in variables:
            da = eval(i)
            varname = i
            tmp_data = da

            masked_var = np.ma.masked_array(da, mask=maskt)
            mean = np.mean(masked_var)
            mean = np.round(mean, 2)
            sd = np.std(masked_var)
            sd = np.round(sd, 2)

            plt_img = np.ma.masked_invalid(masked_var)
            fname = exp + '_' + varname + '_std'
            plot_title = varname + '_std, m = ' + str(mean) + ', sd = ' + str(sd)
            outpath_sd = outpath + '_sd'

            if varname == 'zbar':
                if ('TN' in exp) or ('TD' in exp):
                    cmin = 0.1
                    cmax = 0.6
                else:
                    cmin = 0.5
                    cmax = 2.0
            elif varname == 'sfmc':
                if ('TN' in exp) or ('TD' in exp):
                    cmin = 0.00
                    cmax = 0.14
                else:
                    cmin = 0.00
                    cmax = 0.14

            figure_single_default(data=plt_img, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                                  urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath_sd,
                                  exp=exp, fname=fname, plot_title=plot_title, cmap='coolwarm')

    # SA TN
    root = '/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'SAMERICA_M09_PEATCLSMTN_v01'
    create_figure_maps(exp, domain, root, outpath)

    # SA CLSM
    exp = 'SAMERICA_M09_CLSM_v01'
    create_figure_maps(exp, domain, root, outpath)

    # CO
    exp = 'CONGO_M09_PEATCLSMTN_v01'
    create_figure_maps(exp, domain, root, outpath)

    # CO
    exp = 'CONGO_M09_CLSM_v01'
    create_figure_maps(exp, domain, root, outpath)

    # IN TN
    exp = 'INDONESIA_M09_PEATCLSMTN_v01'
    create_figure_maps(exp, domain, root, outpath)

    # IN TD
    exp = 'INDONESIA_M09_PEATCLSMTD_v01'
    create_figure_maps(exp, domain, root, outpath)

    # IN CLSM
    exp = 'INDONESIA_M09_CLSM_v01'
    create_figure_maps(exp, domain, root, outpath)


def plot_catparams(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp, 'maps', 'catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')

    for param in params:

        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname = param
        # cmin cmax, if not defined determined based on data
        if ((param == "poros") | (param == "poros30")):
            cmin = 0
            cmax = 0.92
        else:
            cmin = None
            cmax = None
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=param)


def ToGeoTiff_catparams(exp, domain, root, outpath):
    import pyproj
    import gdal
    from osgeo import osr
    outpath = os.path.join(outpath, exp, 'maps', 'catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')

    ease = pyproj.Proj(("+proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 "
                        "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m"))
    x_min, y_max = ease(io.grid.tilegrids['ll_lon']['domain'], io.grid.tilegrids['ur_lat']['domain'])
    x_min = x_min
    y_max = y_max
    for param in params:

        if param == "poros":
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = params[param].values
            data = np.ma.masked_invalid(img)
            drv = gdal.GetDriverByName("GTiff")
            ds = drv.Create(outpath + "/" + param + ".tif", data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
            x_resolution = 9008.055210146
            y_resolution = -9008.055210146
            x_skew = 0
            y_skew = 0
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(6933)
            ds.SetProjection(srs.ExportToWkt())
            # The order of input arguments in this tuple is weired, but this is correct.
            ds.SetGeoTransform((x_min, x_resolution, x_skew, y_max, y_skew, y_resolution))
            ds.GetRasterBand(1).WriteArray(data)
            # output_raster = outpath+"/"+param+"_WGS84.tif"
            # srs2 = osr.SpatialReference()
            # srs2.ImportFromEPSG(4326)
            # gdal.Warp(output_raster,ds,dstSRS=srs2)


def plot_skillmetrics_comparison_et(et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod,
                                    le_obs, le_mod, zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod,
                                    Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc, rzmc, srfexc, rzexc, catdef,
                                    Qair, vpd_obs, Wind, exp, outpath):
    INDEX = et_obs.columns
    COL = ['bias (m)', 'ubRMSD (m)', 'Pearson_R (-)', 'RMSD (m)', 'abs_bias (m)']
    df_metrics = pd.DataFrame(index=INDEX, columns=COL, dtype=float)

    for c, site in enumerate(et_obs.columns):

        df_tmp_et = pd.concat((et_obs[site], et_mod[site]), axis=1)
        df_tmp_et.columns = ['data_obs', 'data_mod']

        # this is done again to use in the skill metrics calculation, else transformations occur in original data
        df_tmp2_et = pd.concat((et_obs[site], et_mod[site]), axis=1)
        df_tmp2_et.columns = ['data_obs', 'data_mod']

        df_tmp_ee = pd.concat((ee_obs[site], ee_mod[site]), axis=1)
        df_tmp_ee.columns = ['data_obs', 'data_mod']

        df_tmp_br = pd.concat((br_obs[site], br_mod[site]), axis=1)
        df_tmp_br.columns = ['data_obs', 'data_mod']

        df_tmp_rn = pd.concat((rn_obs[site], rn_mod[site]), axis=1)
        df_tmp_rn.columns = ['data_obs', 'data_mod']

        df_tmp2_rn = pd.concat((rn_obs[site], rn_mod[site]), axis=1)
        df_tmp2_rn.columns = ['data_obs', 'data_mod']

        df_tmp_sh = pd.concat((sh_obs[site], sh_mod[site]), axis=1)
        df_tmp_sh.columns = ['data_obs', 'data_mod']

        df_tmp_le = pd.concat((le_obs[site], le_mod[site]), axis=1)
        df_tmp_le.columns = ['data_obs', 'data_mod']

        """metric calculation overall"""
        # metric calculation with back-up df
        bias_site = metrics.bias(df_tmp2_et)  # Bias = bias_site[0]
        ubRMSD_site = metrics.ubRMSD(df_tmp2_et)  # ubRMSD = ubRMSD_site[0]
        pearson_R_site = metrics.Pearson_R(df_tmp2_et)  # Pearson_R = pearson_R_site[0]
        RMSD_site = (ubRMSD_site[0] ** 2 + bias_site[0] ** 2) ** 0.5
        abs_bias_site = bias_site.abs()
        # abs_bias_site_value = abs(abs_bias_site)
        # abs_bias_site = abs_bias_site('bias').update(abs_bias_site_value('bias'))

        # Save metrics in df_metrics.
        df_metrics.loc[site]['bias (mm/day)'] = bias_site[0]
        df_metrics.loc[site]['ubRMSD (mm/day)'] = ubRMSD_site[0]
        df_metrics.loc[site]['Pearson_R (-)'] = pearson_R_site[0]
        df_metrics.loc[site]['RMSD (mm/day)'] = RMSD_site
        df_metrics.loc[site]['abs_bias (mm/day)'] = abs_bias_site[0]

        '''plotting et, zscore, ee, br and seperate plot for overall skillmetrics'''
        # Create x-axis matching in situ data, for plotting time series
        x_start_et = df_tmp2_et.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end_et = df_tmp2_et.index[-1]  # End a-axis with the last day with an observed wtd value.
        Xlim_wtd = [x_start_et, x_end_et]
        Xlim_wtd_e = [x_start_et, df_tmp2_et.index[-191]]
        # xlim to check only the first year of data
        # Xlim_wtd = [df_tmp2_et.index[365+365], df_tmp2_et.index[365+365+364]]

        # Calculate z-score for the time series.
        df_zscore = df_tmp2_et.apply(zscore)

        # define color based on where it is stored/which model run it uses
        if '/Northern/' in outpath:
            color = ['darkorange', '#1f77b4']
        elif '/CLSM/' in outpath:
            color = ['dimgray', '#1f77b4']
        elif '/Drained' in outpath:
            color = ['m', '#1f77b4']
        elif '/Natural' in outpath:
            color = ['g', '#1f77b4']
        elif 'CLSMTN' in exp:
            color = ['g', '#1f77b4']
        elif 'CLSMTD' in exp:
            color = ['m', '#1f77b4']
        else:
            color = ['#1f77b4', '#1f77b4']

        # figures for the paper
        fontsize = 24
        df_tmp_et = df_tmp_et[['data_mod', 'data_obs']]
        df_tmp_et.plot(figsize=(20, 5), fontsize=fontsize, style=['-', '.'], color=color, linewidth=2.5, markersize=6.5,
                       xlim=Xlim_wtd, legend=False)
        plt.ylabel('ET (mm/day)', fontsize=fontsize)
        plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)  # minor xticks
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True)  # minor xticks
        plt.ylim([0, 8])
        plt.tight_layout()
        fname = site + '_' + exp
        if 'TD' in exp:
            fname_long = os.path.join(
                '/data/leuven/324/vsc32460/FIG/in_situ_comparison/paper/timeseries_ET/' + fname + '_TD' + '.png')
        elif 'TN' in exp:
            fname_long = os.path.join(
                '/data/leuven/324/vsc32460/FIG/in_situ_comparison/paper/timeseries_ET/' + fname + '.png')
        else:
            fname_long = os.path.join(
                '/data/leuven/324/vsc32460/FIG/in_situ_comparison/paper/timeseries_ET/' + fname + '_CLSM' + '.png')
        plt.savefig(fname_long, dpi=350)
        plt.close()

        # fig1
        fig1 = plt.figure(figsize=(16, 8.5))
        fontsize = 12

        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, fig=None)
        df_tmp_et = df_tmp_et[['data_mod', 'data_obs']]
        df_tmp_et.plot(ax=ax1, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3,
                       xlim=Xlim_wtd, ylim=[0, 7.5])

        plt.ylabel('ET (mm/day)')

        Title = site + '\n' + ' bias = ' + str(bias_site[0]) + ' (mm/day), ubRMSD = ' + str(
            ubRMSD_site[0]) + '(mm/day), Pearson_R = ' + str(pearson_R_site[0]) + '(-), RMSD = ' + str(
            RMSD_site) + '(mm/day), abs_bias = ' + str(abs_bias_site[0]) + ' (mm/day)'
        plt.title(Title)

        ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, fig=None)
        df_zscore = df_zscore[['data_mod', 'data_obs']]
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['-', '-'], color=color, linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('z-score')

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/et' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig2
        fig2 = plt.figure(figsize=(16, 8.5))
        fontsize = 12

        ax3 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, fig=None)
        df_tmp_ee = df_tmp_ee[['data_mod', 'data_obs']]
        df_tmp_ee.plot(ax=ax3, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd)
        plt.ylabel('Evapotranspiration Efficiency (-)')

        ax4 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, fig=None)
        df_tmp_le = df_tmp_le[['data_mod', 'data_obs']]
        df_tmp_le.plot(ax=ax4, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd)
        plt.ylabel('Latent heat (W/m$^{2}$)')

        ax5 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, fig=None)
        df_tmp_rn = df_tmp_rn[['data_mod', 'data_obs']]
        df_tmp_rn.plot(ax=ax5, fontsize=fontsize, style=['-', '-'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd)
        plt.ylabel('Rn (W/m$^{2}$)')

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/ee' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig3
        fig3 = plt.figure(figsize=(16, 8.5))
        fontsize = 12

        ax6 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, fig=None)
        df_tmp_br = df_tmp_br[['data_mod', 'data_obs']]
        df_tmp_br.plot(ax=ax6, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd, ylim=[-0.2, 1.5])
        plt.ylabel('Bowen Ratio (-)')

        ax8 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, fig=None)
        df_tmp_le = df_tmp_le[['data_mod', 'data_obs']]
        df_tmp_le.plot(ax=ax8, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd)
        plt.ylabel('Latent heat (W/m$^{2}$)')

        ax7 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, fig=None)
        df_tmp_sh = df_tmp_sh[['data_mod', 'data_obs']]
        df_tmp_sh.plot(ax=ax7, fontsize=fontsize, style=['-', '.'], color=color, linewidth=0.9, markersize=3.8,
                       xlim=Xlim_wtd)
        plt.ylabel('Sensible heat (W/m$^{2}$)')

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/br' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig4
        fig4 = plt.figure(figsize=(20, 7.5))
        fontsize = 12

        # xlim to check only one specific year or period
        # Xlim_wtd = [df_tmp2_et.index[365], df_tmp2_et.index[365+365+365]]

        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1, fig=None)
        df_tmp_ratios = df_tmp_ee.merge(df_tmp_br, left_index=True, right_index=True)
        df_tmp_ratios.plot(ax=ax1, y=['data_mod_x', 'data_mod_y'], fontsize=fontsize, style=['-', '-'],
                           color=['darkorange', 'darkmagenta'], linewidth=0.9,
                           xlim=Xlim_wtd, label=['Evapotranspiration Efficiency ', 'Bowen Ratio'])
        plt.ylabel('(-)')

        ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1, fig=None)
        df_tmp_hf = df_tmp_le.merge(df_tmp_sh, left_index=True, right_index=True)
        df_tmp_hf = df_tmp_hf.merge(df_tmp_rn, left_index=True, right_index=True)
        df_tmp_hf.plot(ax=ax2, y=['data_mod', 'data_mod_x', 'data_mod_y'], fontsize=fontsize, style=['-', '-', '-'],
                       color=['gold', 'crimson', 'mediumblue'], linewidth=0.9,
                       xlim=Xlim_wtd, label=['Net Radiation', 'Latent Heat', 'Sensible Heat'])
        plt.ylabel('(W/m$^{2}$)')
        plt.legend()

        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1, fig=None)
        df_tmp_ratios.plot(ax=ax3, y=['data_obs_x', 'data_obs_y'], fontsize=fontsize, style=['-', '-'],
                           color=['darkorange', 'darkmagenta'], linewidth=0.9,
                           xlim=Xlim_wtd, label=['Evapotranspiration Efficiency ', 'Bowen Ratio'])
        plt.ylabel('Evapotranspiration Efficiency (-)')

        ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, fig=None)
        df_tmp_hf = df_tmp_le.merge(df_tmp_sh, left_index=True, right_index=True)
        df_tmp_hf = df_tmp_hf.merge(df_tmp_rn, left_index=True, right_index=True)
        df_tmp_hf.plot(ax=ax4, y=['data_obs', 'data_obs_x', 'data_obs_y'], fontsize=fontsize, style=['-', '-', '-'],
                       color=['gold', 'crimson', 'mediumblue'], linewidth=0.9,
                       xlim=Xlim_wtd, label=['Net Radiation', 'Latent Heat', 'Sensible Heat'])
        plt.ylabel('(W/m$^{2}$)')
        plt.legend()

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/heat_fluxes' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig5
        fig5 = plt.figure(figsize=(26, 12))
        fontsize = 12

        df_dataframe = pd.concat(
            (et_mod[site], eveg_mod[site], esoi_mod[site], eint_mod[site], zbar_mod[site], et_obs[site], wtd_obs[site]),
            axis=1, join='inner')
        df_dataframe.columns = ['evap', 'eveg', 'esoi', 'eint', 'zbar', 'et_obs', 'wtd_obs']

        ax0 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax0, y='et_obs', x='wtd_obs', fontsize=fontsize, style=['.'], color=['#1f77b4'],
                          markersize=3.5)
        plt.ylabel('In situ evapotranspiration (mm/day)', fontsize=22)
        plt.xlabel('In situ water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.ylim(1, 8)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax10 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax10, y='et_obs', x='zbar', fontsize=fontsize, style=['.'], color=['r'], markersize=3.5)
        plt.ylabel('In situ evapotranspiration (mm/day)', fontsize=22)
        plt.xlabel('Water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.ylim(1, 8)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax1 = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax1, y='evap', x='zbar', fontsize=fontsize, style=['.'], color=color, markersize=3.5)
        plt.ylabel('Evapotranspiration (mm/day)', fontsize=22)
        plt.xlabel('Water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.ylim(1, 8)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax2, y='eveg', x='zbar', fontsize=fontsize, style=['.'], color=color, markersize=3.5)
        plt.ylabel('Plant Transpiration (mm/day)', fontsize=22)
        plt.xlabel('Water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax3, y='esoi', x='zbar', fontsize=fontsize, style=['.'], color=color, markersize=3.5)
        plt.ylabel('Soil Evaporation (mm/day)', fontsize=22)
        plt.xlabel('Water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax4 = plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1, fig=None)
        df_dataframe.plot(ax=ax4, y='eint', x='zbar', fontsize=fontsize, style=['.'], color=color, markersize=3.5)
        plt.ylabel('Interception Evaporation (mm/day)', fontsize=22)
        plt.xlabel('Water table depth (m)', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim((-2.3, 0.1))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/ETcomponents_wtd' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # calculation of ETpot and normalization plot
        # calculation of ETpot according to Maes et al.2017
        # Change Tair and rn from Tair_mod to Tair_obs and rn_mod to rn_obs
        # Model ETpot!!!!
        PM = 'False'
        if 'True' in PM:  # Penman Monteith based ETpot calculation
            Tair_mod[site] = (Tair_mod[
                site]) - 273.15  # to calculate everything as °C instead of K, how they do it in Allen et al. 1998 (=FAO)
            gc_PM = 42.0  # mm/s
            rc_PM = 1000 / gc_PM  # s/m
            raH = 208 / Wind[site]  # s/m because windspeed is m/s
            psi = 0.655 * (10e-6) * Psurf_mod[site]
            rohcp = (psi * 0.622 * 2.4298) / (1.01 * (Tair_mod[site] + 273) * 0.287)
            # VPD_mod calc
            SVP = (610 * (10 ** ((7.5 * Tair_mod[site]) / (237.3 + Tair_mod[site]))))
            RH = (Qair[site] * Psurf_mod[site]) / (0.622 * SVP) * 100
            VPD_mod = ((100 - RH) / 100) * SVP
            VPD_mod = pd.Series.to_frame(VPD_mod)

            s_teller = (0.6108) * np.exp(((17.27 * Tair_mod[site]) / (Tair_mod[site] + 237.3)))
            s_noemer = (Tair_mod[site] + 237.3) ** 2
            s = 4098 * (s_teller / s_noemer)
            ETpot_mod = (((s * (rn_mod[site] - ghflux_mod[site])) + (rohcp * VPD_mod[site]) / raH) / (
                    s + psi + (psi * rc_PM / raH))) / 7.3992

            # In situ ETpot!!!
            Tair_obs[site] = (
                Tair_obs[
                    site])  # to calculate everything as °C instead of K, how they do it in Allen et al. 1998 (=FAO)
            psi = 0.655 * (10e-6) * Psurf_mod[site]
            rohcp = (psi * 0.622 * 2.4298) / (1.01 * (Tair_obs[site] + 273) * 0.287)
            s_teller = (0.6108) * np.exp(((17.27 * Tair_obs[site]) / (Tair_obs[site] + 237.3)))
            s_noemer = (Tair_obs[site] + 237.3) ** 2
            s = 4098 * (s_teller / s_noemer)
            ETpot_obs = (((s * (rn_obs[site] - ghflux_mod[site])) + (rohcp * vpd_obs[site]) / raH) / (
                    s + psi + (psi * rc_PM / raH))) / 7.3992

        else:  # Priestley and Taylor based ETpot calculation
            # PT parameters
            Tair_mod[site] = (Tair_mod[
                site]) - 273.15  # to calculate everything as °C instead of K, how they do it in Allen et al. 1998 (=FAO)
            alpha_PT = 1.09
            psi = 0.655 * (10e-6) * Psurf_mod[site]

            # model ETpot
            s_teller = (0.6108) * np.exp(((17.27 * Tair_mod[site]) / (Tair_mod[site] + 237.3)))
            s_noemer = (Tair_mod[site] + 237.3) ** 2
            s = 4098 * (s_teller / s_noemer)
            ETpot_mod = (alpha_PT * ((s * (rn_mod[site] - ghflux_mod[site])) / (s + psi))) / 7.3992

            # In situ ETpot!
            Tair_obs[site] = (
                Tair_obs[
                    site])  # to calculate everything as °C instead of K, how they do it in Allen et al. 1998 (=FAO)
            s_teller = (0.6108) * np.exp(((17.27 * Tair_obs[site]) / (Tair_obs[site] + 237.3)))
            s_noemer = (Tair_obs[site] + 237.3) ** 2
            s = 4098 * (s_teller / s_noemer)
            ETpot_obs = (alpha_PT * ((s * (rn_obs[site] - ghflux_mod[site])) / (s + psi))) / 7.3992

        # other df calculations
        et_obs_used = et_obs[site]
        et_mod_used = et_mod[site]

        norm_et_mod = et_obs_used / ETpot_mod
        norm_et_obs = et_obs_used / ETpot_obs
        norm_et_mod_mod = et_mod_used / ETpot_mod
        norm_et_mod = pd.Series.to_frame(norm_et_mod)
        norm_et_obs = pd.Series.to_frame(norm_et_obs)
        norm_et_mod_mod = pd.Series.to_frame(norm_et_mod_mod)
        norm_et_wtd_mod = pd.concat([norm_et_mod, wtd_obs[site]], axis=1)
        norm_et_wtd_mod.columns = ['et_norm_mod', 'wtd_obs']
        norm_et_wtd_mod = norm_et_wtd_mod[
            norm_et_wtd_mod['wtd_obs'].notna()]  # removes all rows for each column with nan-values in column: wtd_obs
        norm_et_wtd_obs = pd.concat([norm_et_obs, wtd_obs[site]], axis=1)
        norm_et_wtd_obs.columns = ['et_norm_obs', 'wtd_obs']
        norm_et_wtd_obs = norm_et_wtd_obs[
            norm_et_wtd_obs['wtd_obs'].notna()]  # removes all rows for each column with nan-values in column: wtd_obs
        norm_et_wtd_mod_mod = pd.concat([norm_et_mod_mod, zbar_mod[site]], axis=1)
        norm_et_wtd_mod_mod.columns = ['et_norm_mod_mod', 'wtd_mod']
        norm_et_wtd_mod_mod = norm_et_wtd_mod_mod[
            norm_et_wtd_mod_mod['wtd_mod'].notna()]
        pot_et_wtd_mod = pd.concat([ETpot_mod, wtd_obs[site]], axis=1)
        pot_et_wtd_mod.columns = ['pot_et_mod', 'wtd_obs']
        pot_et_wtd_mod = pot_et_wtd_mod[
            pot_et_wtd_mod['wtd_obs'].notna()]  # removes all rows for each column with nan-values in column: wtd_obs
        pot_et_wtd_obs = pd.concat([ETpot_obs, wtd_obs[site]], axis=1)
        pot_et_wtd_obs.columns = ['pot_et_obs', 'wtd_obs']
        pot_et_wtd_obs = pot_et_wtd_obs[
            pot_et_wtd_obs['wtd_obs'].notna()]  # removes all rows for each column with nan-values in column: wtd_obs

        et_obs_used = pd.Series.to_frame(et_obs_used)
        et_mod_used = pd.Series.to_frame(et_mod_used)
        ETpot_mod = pd.Series.to_frame(ETpot_mod)
        ETpot_obs = pd.Series.to_frame(ETpot_obs)

        # if site is 'UndrainedPSF' or 'DrainedPSF':
        #    norm_et_wtd_obs.to_csv(
        #        r'/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Natural/ET/ETpot_WTD_obs' + site + '.csv',
        #        index=True, header=True)
        #    norm_et_wtd_mod.to_csv(
        #        r'/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Natural/ET/ETpot_WTD_mod' + site + '.csv',
        #        index=True, header=True)
        # else:
        #    print('Not DrainedPSF or UndrainedPSF')

        # fig6
        fig6 = plt.figure(figsize=(20, 12))
        fontsize = 12

        # xlim to check only one specific year or period
        # Xlim_wtd = [df_tmp2_et.index[365], df_tmp2_et.index[365+365+365]]

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1, fig=None)
        ETpot_mod.plot(ax=ax1, fontsize=fontsize, style=['-'], color='darkorange', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('Model potential ET (mm/day)', fontsize=16)
        plt.legend()
        plt.ylim([2, 8])

        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1, fig=None)
        et_obs_used.plot(ax=ax2, fontsize=fontsize, style=['-'], color='#1f77b4', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('In situ ET \n (mm/day)', fontsize=16)
        plt.legend()

        ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1, fig=None)
        norm_et_mod.plot(ax=ax3, fontsize=fontsize, style=['-'], color='darkolivegreen', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('Normalized ET with ETpot \n from model (mm/day)', fontsize=16)
        plt.legend()
        plt.ylim([0.25, 2])

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/ET_mod_timeseries' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig7
        fig7 = plt.figure(figsize=(20, 12))
        fontsize = 12

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1, fig=None)
        ETpot_obs.plot(ax=ax1, fontsize=fontsize, style=['-'], color='sandybrown', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('In situ potential ET (mm/day)', fontsize=16)
        plt.legend()
        plt.ylim([2, 8])

        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1, fig=None)
        et_obs_used.plot(ax=ax2, fontsize=fontsize, style=['-'], color='#1f77b4', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('In situ ET \n (mm/day)', fontsize=16)
        plt.legend()

        ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1, fig=None)
        norm_et_obs.plot(ax=ax3, fontsize=fontsize, style=['-'], color='limegreen', linewidth=0.9, xlim=Xlim_wtd)
        plt.ylabel('Normalized ET with ETpot \n from in situ (mm/day)', fontsize=16)
        plt.legend()
        plt.ylim([0.25, 2])

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/ET_obs_timeseries' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig8
        fig8 = plt.figure(figsize=(22, 13))
        fontsize = 12

        # remove haze and concat modeled to timeframe of observed
        exclusion_dates = pd.date_range(start='2006/09/17', end='2006/12/17')
        norm_et_wtd_obs = norm_et_wtd_obs.loc[~norm_et_wtd_obs.index.isin(exclusion_dates)]
        overlap_check = pd.concat((norm_et_wtd_mod_mod, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        norm_et_wtd_mod_mod = overlap_check[overlap_check['et_norm_obs'].notna()]
        norm_et_wtd_mod_mod = norm_et_wtd_mod_mod.drop(columns=["et_norm_obs"])
        overlap_check = pd.concat((pot_et_wtd_mod, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        pot_et_wtd_mod = overlap_check[overlap_check['et_norm_obs'].notna()]
        pot_et_wtd_mod = pot_et_wtd_mod.drop(columns=["et_norm_obs"])
        overlap_check = pd.concat((pot_et_wtd_obs, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        pot_et_wtd_obs = overlap_check[overlap_check['et_norm_obs'].notna()]
        pot_et_wtd_obs = pot_et_wtd_obs.drop(columns=["et_norm_obs"])
        overlap_check = pd.concat((norm_et_wtd_mod, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        norm_et_wtd_mod = overlap_check[overlap_check['et_norm_obs'].notna()]
        norm_et_wtd_mod = norm_et_wtd_mod.drop(columns=["et_norm_obs"])

        ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=1, fig=None)
        pot_et_wtd_mod.plot(ax=ax1, y='pot_et_mod', x='wtd_obs', fontsize=fontsize, style=['.'], color='darkorange',
                            markersize=4.5)
        plt.ylabel('Model potential \n ET (mm/day)', fontsize=20)
        plt.xlabel('In situ water table depth (m)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1, fig=None)
        pot_et_wtd_obs.plot(ax=ax2, y='pot_et_obs', x='wtd_obs', fontsize=fontsize, style=['.'], color='sandybrown',
                            markersize=4.5)
        plt.ylabel('In situ potential\n  ET (mm/day)', fontsize=20)
        plt.xlabel('In situ water table depth (m)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1, fig=None)
        norm_et_wtd_mod.plot(ax=ax3, y='et_norm_mod', x='wtd_obs', fontsize=fontsize, style=['.'],
                             color='darkolivegreen', markersize=4.5)
        plt.ylabel('In situ ET normalized with \n model ETpot (mm/day)', fontsize=20)
        plt.xlabel('In situ water table depth (m)', fontsize=20)
        plt.ylim([0.2, 2])
        plt.xticks(fontsize=18)
        # plt.xlim([-1.4, 0.2])
        plt.yticks(fontsize=18)

        ax4 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1, fig=None)
        norm_et_wtd_obs.plot(ax=ax4, y='et_norm_obs', x='wtd_obs', fontsize=fontsize, style=['.'], color='limegreen',
                             markersize=4.3)
        plt.ylabel('In situ ET normalized with \n in situ ETpot (mm/day)', fontsize=20)
        plt.xlabel('In situ water table depth (m)', fontsize=20)
        plt.ylim([0.2, 2])
        # plt.xlim([-1.35, -0.2])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax5 = plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1, fig=None)
        norm_et_wtd_mod_mod.plot(ax=ax5, y='et_norm_mod_mod', x='wtd_mod', fontsize=fontsize - 4, style=['.'],
                                 color='limegreen',
                                 markersize=4.5)
        plt.ylabel('Model ET normalized with \n model ETpot (mm/day)', fontsize=20)
        plt.xlabel('Model water table depth (m)', fontsize=20)
        plt.ylim([0.2, 2])
        # plt.xlim([-1.4, 0.2])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/potential_ET_WTD' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # figextra
        rn_wtd = pd.concat([rn_obs[site], wtd_obs[site]],
                           axis=1, join='inner')
        RN_wtd = pd.concat([rn_mod[site], wtd_obs[site]],
                           axis=1, join='inner')
        rn_wtd.columns = ['rn_obs', 'wtd_obs']
        RN_wtd.columns = ['rn_mod', 'wtd_obs']
        RN_wtd = RN_wtd[RN_wtd['wtd_obs'].notna()]
        rn_wtd = rn_wtd[rn_wtd['wtd_obs'].notna()]

        # remove haze period in rn data based on the period assigned above
        overlap_check = pd.concat((rn_wtd, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        rn_wtd = overlap_check[overlap_check['et_norm_obs'].notna()]
        rn_wtd = rn_wtd.drop(columns=["et_norm_obs"])
        overlap_check = pd.concat((RN_wtd, norm_et_wtd_obs["et_norm_obs"]), axis=1)
        RN_wtd = overlap_check[overlap_check['et_norm_obs'].notna()]
        RN_wtd = RN_wtd.drop(columns=["et_norm_obs"])

        figextra = plt.figure(figsize=(20, 20))
        fontsize = 12

        # calculate lowess fit and confidence interval 95%
        def smooth(x, y, xgrid):
            samples = np.random.choice(len(x), 1002, replace=True)
            y_s = y[samples]
            x_s = x[samples]
            y_sm = sm_lowess(y_s, x_s, it=0, missing='drop', return_sorted=False)
            # regularly sample it onto the grid
            y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
            return y_grid

        xgrid = np.linspace(pot_et_wtd_mod['wtd_obs'].min(), pot_et_wtd_mod['wtd_obs'].max())
        K = 1000

        ax1 = plt.subplot2grid((2, 2), (0, 0), fig=None)
        pot_et_wtd_obs.plot(ax=ax1, y='pot_et_obs', x='wtd_obs', fontsize=fontsize + 4, style=['.'], color='#1f77b4',
                            markersize=4)
        smooths = np.stack([smooth(pot_et_wtd_obs['wtd_obs'], pot_et_wtd_obs['pot_et_obs'], xgrid) for k in range(K)]).T
        mean = np.nanmean(smooths, axis=1)
        stderr = scipy.stats.sem(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        plt.plot(xgrid, mean, color='k', linewidth=3.5)
        plt.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
        plt.ylabel('In situ ET$\mathrm{_{pot}}$ (mm/day)', fontsize=fontsize + 24)
        plt.ylim([1, 8.6])
        plt.xlabel('')

        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1, fig=None)
        pot_et_wtd_mod.plot(ax=ax2, y='pot_et_mod', x='wtd_obs', fontsize=fontsize + 4, style=['.'], color=color,
                            markersize=4)
        smooths = np.stack([smooth(pot_et_wtd_mod['wtd_obs'], pot_et_wtd_mod['pot_et_mod'], xgrid) for k in range(K)]).T
        mean = np.nanmean(smooths, axis=1)
        stderr = scipy.stats.sem(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        plt.plot(xgrid, mean, color='k', linewidth=3.5)
        plt.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
        plt.ylabel('Model ET$\mathrm{_{pot}}$ (mm/day)', fontsize=fontsize + 24)
        plt.ylim([1, 8.6])
        plt.xlabel('')

        ax3 = plt.subplot2grid((2, 2), (1, 0), fig=None)
        rn_wtd.plot(ax=ax3, y='rn_obs', x='wtd_obs', fontsize=fontsize + 4, style=['.'], color='#1f77b4', markersize=4)
        smooths = np.stack([smooth(rn_wtd['wtd_obs'], rn_wtd['rn_obs'], xgrid) for k in range(K)]).T
        mean = np.nanmean(smooths, axis=1)
        stderr = scipy.stats.sem(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        plt.plot(xgrid, mean, color='k', linewidth=3.5)
        plt.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
        plt.ylabel('In situ R$\mathrm{_{net}}$ (W/m²)', fontsize=fontsize + 24)
        plt.xlabel('In situ WTD (m)', fontsize=fontsize + 24)
        plt.ylim([30, 260])

        ax4 = plt.subplot2grid((2, 2), (1, 1), fig=None)
        RN_wtd.plot(ax=ax4, y='rn_mod', x='wtd_obs', fontsize=fontsize + 4, style=['.'], color=color, markersize=4)
        smooths = np.stack([smooth(RN_wtd['wtd_obs'], RN_wtd['rn_mod'], xgrid) for k in range(K)]).T
        mean = np.nanmean(smooths, axis=1)
        stderr = scipy.stats.sem(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        plt.plot(xgrid, mean, color='k', linewidth=3.5)
        plt.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
        plt.ylabel('Model R$\mathrm{_{net}}$ (W/m²)', fontsize=fontsize + 24)
        plt.xlabel('In situ WTD (m)', fontsize=fontsize + 24)
        plt.ylim([30, 260])
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        ax4.get_legend().remove()
        ax1.tick_params(axis='both', which='major', labelsize=fontsize + 18)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize + 18)
        ax3.tick_params(axis='both', which='major', labelsize=fontsize + 18)
        ax4.tick_params(axis='both', which='major', labelsize=fontsize + 18)
        ax1.set_xticks([])
        ax2.set_xticks([])

        plt.tight_layout(w_pad=6, h_pad=4)
        fname = site
        fname_long = os.path.join(outpath + '/RnvsWTD' + fname + '.png')
        plt.savefig(fname_long, dpi=300)
        plt.close()

        # fig9
        RN = pd.concat([rn_mod[site], rn_obs[site]], axis=1)
        RN.columns = ['rn_mod', 'rn_obs']
        RN = RN[RN['rn_obs'].notna()]  # removes all rows for each column with nan-values in column: rn_obs
        TAIR = pd.concat([Tair_mod[site], Tair_obs[site]], axis=1)
        TAIR.columns = ['tair_mod', 'tair_obs']
        TAIR = TAIR[TAIR['tair_obs'].notna()]  # removes all rows for each column with nan-values in column: rn_obs
        RN2 = copy.deepcopy(RN)
        TAIR2 = copy.deepcopy(TAIR)

        SVP = (610 * (10 ** ((7.5 * Tair_mod[site]) / (237.3 + Tair_mod[site]))))
        RH = (Qair[site] * Psurf_mod[site]) / (0.622 * SVP) * 100
        VPD_mod = ((100 - RH) / 100) * SVP
        VPD_mod = pd.Series.to_frame(VPD_mod)
        VPD = pd.concat([VPD_mod[site], vpd_obs[site]], axis=1)
        VPD.columns = ['vpd_mod', 'vpd_obs']
        VPD2 = copy.deepcopy(VPD)

        """metric calculation overall"""
        R_rn = metrics.Pearson_R(RN2)
        bias_rn = metrics.bias(RN2)
        abias_rn = bias_rn.abs()
        R_ta = metrics.Pearson_R(TAIR2)
        bias_ta = metrics.bias(TAIR2)
        abias_ta = bias_ta.abs()
        R_vpd = metrics.Pearson_R(VPD2)
        bias_vpd = metrics.bias(VPD2)
        abias_vpd = bias_vpd.abs()

        fig9 = plt.figure(figsize=(17, 10))
        fontsize = 12

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, fig=None)
        RN.plot(ax=ax1, fontsize=fontsize, style=['-', '-'], color=color, linewidth=0.9, markersize=3,
                xlim=Xlim_wtd)
        plt.ylabel('Rnet (W/m²)')

        Title = site + '\n' + 'R (Rn) = ' + str(R_rn[0]) + ',  abs_bias (Rn) = ' + str(
            abias_rn[0]) + ',  bias (Rn) = ' + str(bias_rn[0])
        plt.title(Title)

        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, fig=None)
        VPD.plot(ax=ax2, fontsize=fontsize, style=['-', '-'], color=color, linewidth=0.9, markersize=3, xlim=Xlim_wtd)
        plt.ylabel('VPD (Pa)')

        Title = site + '\n' + 'R (vpd) = ' + str(R_vpd[0]) + ',  abs_bias (vpd) = ' + str(
            abias_vpd[0]) + ',  bias (vpd) = ' + str(bias_vpd[0])
        plt.title(Title)

        ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, fig=None)
        TAIR.plot(ax=ax3, fontsize=fontsize, style=['-', '-'], color=color, linewidth=0.9, markersize=3, xlim=Xlim_wtd)
        plt.ylabel('Tair (°C) ')

        Title = site + '\n' + 'R (Tair) = ' + str(R_ta[0]) + ',  abs_bias (Tair) = ' + str(
            abias_ta[0]) + ',  bias (Tair) = ' + str(bias_ta[0])
        plt.title(Title)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/rn-tair-vpd' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig10
        Figure1 = plt.figure(figsize=(20, 7.5))
        fontsize = 12
        zbar = zbar_mod[site]
        AR_all = pd.concat([AR1[site], AR2[site], AR4[site], zbar], axis=1)
        AR_all.columns = ['AR1', 'AR2', 'AR4', 'zbar']

        ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1, fig=None)
        AR_all.plot(ax=ax1, y='AR1', x='zbar', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('AR1 [-]', fontsize=20)
        plt.xlabel('Modeled water table depth (m)', fontsize=20)
        plt.ylim([0, 1])
        plt.xlim([-2.5, 0])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, fig=None)
        AR_all.plot(ax=ax2, y='AR2', x='zbar', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('AR2 [-]', fontsize=20)
        plt.xlabel('Modeled water table depth (m)', fontsize=20)
        plt.ylim([0, 1])
        plt.xlim([-2.5, 0])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, fig=None)
        AR_all.plot(ax=ax3, y='AR4', x='zbar', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('AR4 [-]', fontsize=20)
        plt.xlabel('Modeled water table depth (m)', fontsize=20)
        plt.ylim([0, 1])
        plt.xlim([-2.5, 0])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/ARvsWTD' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig11
        Figure2 = plt.figure(figsize=(30, 20))
        fontsize = 12
        sfmc_all = pd.concat((et_mod[site], eveg_mod[site], esoi_mod[site], zbar_mod[site], sfmc[site]), axis=1,
                             join='inner')
        sfmc_all.columns = ['evap', 'eveg', 'esoi', 'zbar', 'sfmc']

        ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1, fig=None)
        sfmc_all.plot(ax=ax1, y='sfmc', x='evap', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('sfmc', fontsize=20)
        plt.xlabel('evap', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1, fig=None)
        sfmc_all.plot(ax=ax2, y='sfmc', x='eveg', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('sfmc', fontsize=20)
        plt.xlabel('eveg', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, fig=None)
        sfmc_all.plot(ax=ax3, y='sfmc', x='esoi', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('sfmc', fontsize=20)
        plt.xlabel('esoi', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, fig=None)
        sfmc_all.plot(ax=ax4, y='sfmc', x='zbar', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('sfmc', fontsize=20)
        plt.xlabel('zbar', fontsize=20)
        plt.ylim([0, 1])
        plt.xlim([-2, 0])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax5 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2, fig=None)
        sfmc[site].plot(ax=ax5, fontsize=fontsize, style=['.'], color='k', markersize=4, xlim=Xlim_wtd)
        plt.ylabel('sfmc', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/sfmcvsall' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig12
        Figure3 = plt.figure(figsize=(30, 20))
        fontsize = 12
        rzmc_all = pd.concat((et_mod[site], eveg_mod[site], esoi_mod[site], zbar_mod[site], rzmc[site]), axis=1,
                             join='inner')
        rzmc_all.columns = ['evap', 'eveg', 'esoi', 'zbar', 'rzmc']

        ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1, fig=None)
        rzmc_all.plot(ax=ax1, y='rzmc', x='evap', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('rzmc', fontsize=20)
        plt.xlabel('evap', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1, fig=None)
        rzmc_all.plot(ax=ax2, y='rzmc', x='eveg', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('rzmc', fontsize=20)
        plt.xlabel('eveg', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, fig=None)
        rzmc_all.plot(ax=ax3, y='rzmc', x='esoi', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('rzmc', fontsize=20)
        plt.xlabel('esoi', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, fig=None)
        rzmc_all.plot(ax=ax4, y='rzmc', x='zbar', fontsize=fontsize, style=['.'], color='k', markersize=4)
        plt.ylabel('rzmc', fontsize=20)
        plt.xlabel('zbar', fontsize=20)
        plt.ylim([0, 1])
        plt.xlim([-2, 0])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax5 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2, fig=None)
        rzmc[site].plot(ax=ax5, fontsize=fontsize, style=['.'], color='k', markersize=4, xlim=Xlim_wtd)
        plt.ylabel('rzmc', fontsize=20)
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/rzmcvsall' + fname + '.png')
        plt.savefig(fname_long, dpi=200)
        plt.close()


def plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc, rzexc,
                                     catdef, exp, outpath):
    # Initiate dataframe to store metrics in of wtd
    INDEX = wtd_obs.columns
    COL = ['bias (m)', 'RMSD (m)', 'ubRMSD (m)', 'R (-)', 'anomR (-)']
    df_metrics = pd.DataFrame(index=INDEX, columns=COL, dtype=float)

    INDEX = wtd_obs.columns
    COL = ['bias_l', 'bias_u', 'RMSD_l', 'RMSD_u', 'ubRMSD_l', 'ubRMSD_u', 'R_l', 'R_u', 'anomR_l', 'anomR_u']
    df_metrics_CI = pd.DataFrame(index=INDEX, columns=COL, dtype=float)

    # Initiate dataframe to store metrics in of precipitation
    INDEX = precip_obs.columns
    COL = ['bias_P (m)', 'ubRMSD_P (m)', 'R_P (-)', 'RMSD_P (m)']
    df_metrics_P = pd.DataFrame(index=INDEX, columns=COL, dtype=float)

    for c, site in enumerate(wtd_obs.columns):

        df_tmp_wtd = pd.concat((wtd_obs[site], wtd_mod[site]), axis=1)
        df_tmp_wtd.columns = ['data_obs', 'data_mod']

        df_tmp_precip = pd.concat((precip_obs[site], precip_mod[site]), axis=1)
        df_tmp_precip.columns = ['data_obs', 'data_mod']

        df_tmp2_wtd = copy.deepcopy(df_tmp_wtd)
        df_tmp2_wtd = df_tmp2_wtd[['data_mod', 'data_obs']]

        df_tmp2_precip = copy.deepcopy(df_tmp_precip)
        df_tmp2_precip = df_tmp2_precip[['data_mod', 'data_obs']]

        # to remove the flooding events in SA sites, both for the skill metrics as well for the plots
        # if (('SAM_01' in site) or ('QT-2010-1' in site)):
        #    df_tmp_wtd = df_tmp_wtd.drop(df_tmp_wtd[df_tmp_wtd['data_obs'] > 0.02].index)
        #    df_tmp2_wtd = df_tmp2_wtd.drop(df_tmp2_wtd[df_tmp2_wtd['data_obs'] > 0.02].index)
        # else:
        #    print('ok')

        bias_site = metrics.bias(df_tmp2_wtd)  # Bias = bias_site[0]
        ubRMSD_site = metrics.ubRMSD(df_tmp2_wtd)  # ubRMSD = ubRMSD_site[0]
        pearson_R_site = metrics.Pearson_R(df_tmp2_wtd)  # Pearson_R = pearson_R_site[0]
        RMSD_site = copy.deepcopy(ubRMSD_site)
        RMSD_site[0] = (ubRMSD_site[0] ** 2 + bias_site[0] ** 2) ** 0.5

        x = ubRMSD_site[0]  # so x is ubRMSD_site[0]
        y = bias_site[0]  # so y is bias_site[0]
        # x =symbols('x')
        # y=symbols('y')
        # = ((x**2) + (y**2)) **0.5
        # derivative1 = diff(f,y)
        # derivative2 =diff(f,x)
        error_ubRMSD = (ubRMSD_site[6] - ubRMSD_site[5]) / 2
        error_bias = (bias_site[6] - bias_site[5]) / 2
        error_RMSD = ((1.0 * x * (x ** 2 + y ** 2) ** (-0.5)) ** 2 * (error_ubRMSD ** 2)) + (
                (1.0 * x * (x ** 2 + y ** 2) ** (-0.5)) ** 2 * (error_bias ** 2)) ** 0.5
        RMSD_site[5] = RMSD_site[0] - error_RMSD
        RMSD_site[6] = RMSD_site[0] + error_RMSD

        # calculate anomalies only of 3 years or more of total data
        threeYears = df_tmp_wtd['data_obs'].count()
        if threeYears >= 1095:
            anomaly_obs = calc_anomaly(df_tmp2_wtd['data_obs'])
            anomaly_mod = calc_anomaly(df_tmp2_wtd['data_mod'])
            anomalies = pd.concat([anomaly_mod, anomaly_obs], axis=1)
            anomR_site = metrics.Pearson_R(anomalies)
        else:
            anomR_site = np.array(['nan'])
            anomR_site = pd.Series(anomR_site, index=['R'])

        # calculate confidence intervals for each of the metrics
        # def pearsonr_ci(x, y, alpha=0.05):
        #    r, p = stats.pearsonr(x, y)
        #    r_z = np.arctanh(r)
        #    se = 1 / np.sqrt(x.size - 3)
        #    z = stats.norm.ppf(1 - alpha / 2)
        #    lo_z, hi_z = r_z - z * se, r_z + z * se
        #   lo, hi = np.tanh((lo_z, hi_z))
        #    return lo, hi

        # Save metrics in df_metrics.
        df_metrics.loc[site]['bias (m)'] = bias_site[0]
        df_metrics.loc[site]['ubRMSD (m)'] = ubRMSD_site[0]
        df_metrics.loc[site]['R (-)'] = pearson_R_site[0]
        df_metrics.loc[site]['RMSD (m)'] = RMSD_site[0]
        df_metrics.loc[site]['anomR (-)'] = anomR_site[0]

        df_metrics_CI.loc[site]['bias_l'] = bias_site[5]
        df_metrics_CI.loc[site]['bias_u'] = bias_site[6]
        df_metrics_CI.loc[site]['ubRMSD_l'] = ubRMSD_site[5]
        df_metrics_CI.loc[site]['ubRMSD_u'] = ubRMSD_site[6]
        df_metrics_CI.loc[site]['R_l'] = pearson_R_site[6]
        df_metrics_CI.loc[site]['R_u'] = pearson_R_site[7]
        df_metrics_CI.loc[site]['RMSD_l'] = RMSD_site[5]
        df_metrics_CI.loc[site]['RMSD_u'] = RMSD_site[6]
        try:
            df_metrics_CI.loc[site]['anomR_l'] = anomR_site[6]
            df_metrics_CI.loc[site]['anomR_u'] = anomR_site[7]
        except:
            df_metrics_CI.loc[site]['anomR_l'] = anomR_site[0]
            df_metrics_CI.loc[site]['anomR_u'] = anomR_site[0]

        if (-10) in set(df_tmp_precip['data_obs']):
            df_tmp_precip = df_tmp_precip
        else:
            bias_site_P = metrics.bias(df_tmp2_precip)  # Bias = bias_site[0]
            ubRMSD_site_P = metrics.ubRMSD(df_tmp2_precip)  # ubRMSD = ubRMSD_site[0]
            pearson_R_site_P = metrics.Pearson_R(df_tmp2_precip)  # Pearson_R = pearson_R_site[0]
            RMSD_site_P = (ubRMSD_site_P[0] ** 2 + bias_site_P[0] ** 2) ** 0.5
            abs_bias_site_P = bias_site_P.abs()

            # Save metrics in df_metrics.
            df_metrics_P.loc[site]['bias_P (m)'] = bias_site_P[0]
            df_metrics_P.loc[site]['ubRMSD_P (m)'] = ubRMSD_site_P[0]
            df_metrics_P.loc[site]['R_P (-)'] = pearson_R_site_P[0]
            df_metrics_P.loc[site]['RMSD_P (m)'] = RMSD_site_P

        # Create x-axis matching in situ data.
        x_start_wtd = df_tmp2_wtd.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end_wtd = df_tmp2_wtd.index[-1]  # End a-axis with the last day with an observed wtd value.
        Xlim_wtd = [x_start_wtd, x_end_wtd]

        # Create x-axis matching in situ data.
        x_start_precip = df_tmp_precip.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end_precip = df_tmp_precip.index[-1]  # End a-axis with the last day with an observed wtd value.
        Xlim_precip = [x_start_precip, x_end_precip]

        # Calculate z-score for the time series.

        df_zscore = df_tmp2_wtd.apply(zscore)
        df_zscore_P = df_tmp2_precip.apply(zscore)

        plt.figure(figsize=(16, 8.5))
        fontsize = 12

        # define color based on where it is stored/which model run it uses
        if '/Northern/' in outpath:
            color = ['darkorange', '#1f77b4']
        elif '/CLSM/' in outpath:
            color = ['dimgray', '#1f77b4']
        elif '/Drained' in outpath:
            color = ['m', '#1f77b4']
        elif '/Natural' in outpath:
            color = ['g', '#1f77b4']
        elif 'CLSMTN' in exp:
            color = ['g', '#1f77b4']
        elif 'CLSMTD' in exp:
            color = ['m', '#1f77b4']
        else:
            color = ['#1f77b4', '#1f77b4']

        if '/Northern/' in outpath:
            color_short = 'darkorange'
        elif '/CLSM/' in outpath:
            color_short = 'dimgray'
        elif '/Drained' in outpath:
            color_short = 'm'
        elif '/Natural' in outpath:
            color_short = 'g'
        elif 'CLSMTN' in exp:
            color_short = 'g'
        elif 'CLSMTD' in exp:
            color_short = 'm'
        else:
            color_short = '#1f77b4'

        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3, fig=None)
        ax1.set_ylabel('zbar [m]')
        df_tmp_wtd = df_tmp_wtd[['data_mod', 'data_obs']]
        df_tmp_wtd.plot(ax=ax1, fontsize=fontsize, style=['-', '.'], color=color, linewidth=2, markersize=4.5,
                        xlim=Xlim_wtd)
        ax5 = ax1.twinx()
        ax5.set_ylabel('Moisture content')
        rzmc_mod[site].plot(ax=ax5, fontsize=fontsize, style=['--'], color='cyan', linewidth=1.5,
                            xlim=Xlim_wtd)  # ifzbar is not showing add: (ax=ax5, x_compat=True,...
        sfmc_mod[site].plot(ax=ax5, fontsize=fontsize, style=['--'], color='palegreen', linewidth=1.5, xlim=Xlim_wtd)
        legendLines = [matplotlib.lines.Line2D([0], [0], color=color_short, linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='#1f77b4', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='cyan', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='palegreen', linewidth=2)]
        ax1.legend(legendLines, ['wtd_mod', 'wtd_obs', 'rzmc', 'sfmc'])
        #### this can be used to define the axes of certain sites. and replace the above line
        # if ('UndrainedPSF' in site ) or ('DrainedPSF' in site):
        #    df_tmp_wtd.plot(ax=ax1, fontsize=fontsize, style=['-','.'], color=color, linewidth=2, markersize=4.5, xlim=Xlim_wtd, ylim=[-2.0, 0.2])
        # else:
        #    df_tmp_wtd.plot(ax=ax1, fontsize=fontsize, style=['-','.'], color=color, linewidth=2, markersize=4.5, xlim=Xlim_wtd)
        # plt.ylabel('zbar [m]')

        Title = site + '\n' + ' bias = ' + str(bias_site[0]) + ', ubRMSD = ' + str(
            ubRMSD_site[0]) + ', R = ' + str(pearson_R_site[0]) + ', RMSD = ' + str(
            RMSD_site[0]) + ' anomR = ' + str(anomR_site[0])
        plt.title(Title)

        ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=3, fig=None)
        df_zscore = df_zscore[['data_mod', 'data_obs']]
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['-', '-'], color=color, linewidth=2, xlim=Xlim_wtd)
        plt.ylabel('z-score')
        plt.legend('P_mod', 'P_obs')

        if ((-10) in set(df_tmp_precip['data_obs'])):
            ax3 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3, fig=None)
            df_tmp_precip = df_tmp_precip[['data_mod', 'data_obs']]
            df_tmp_precip.plot(ax=ax3, fontsize=fontsize, style=['-', '.'], color=color, linewidth=2, markersize=4.5,
                               xlim=Xlim_wtd)
            plt.ylabel('precipitation [mm/d]')
        else:
            ax3 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3, fig=None)
            df_tmp_precip = df_tmp_precip[['data_mod', 'data_obs']]
            df_tmp_precip.plot(ax=ax3, fontsize=fontsize, style=['-', '.'], color=color, linewidth=2, markersize=4.5,
                               xlim=Xlim_wtd)
            plt.ylim([0, 120])
            plt.ylabel('precipitation [mm/d]')

            # Sebastian added #to plot the obs vs meas rainfall and 1/1line
            # ax4 = plt.subplot2grid((3,3), (2,2), rowspan=1, colspan=1, fig=None)
            # df_tmp_precip.plot(ax=ax4, fontsize=fontsize, x='data_mod',y='data_obs', style=['.'],color='#1f77b4', markersize=2.5, label='Precipitation (mm/day)')
            # plt.plot([0,250],[0,250], linestyle='solid', color='red')
            # plt.ylabel('data_obs')
            # plt.xlabel('data_mod')
            # plt.legend(fontsize=9)
            # plt.xticks(np.arange(0, 121, step=20))
            # plt.yticks(np.arange(0, 251, step=50))

            #cumulative rainfall plot (adjust the colspan to 2 in ax3)
            #ax4 = plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1, fig=None)
            #df_tmp_precip_sum = df_tmp_precip.dropna(axis=0, how='any')
            #df_tmp_precip_sum = df_tmp_precip_sum[['data_mod', 'data_obs']].cumsum(skipna=True)
            #df_tmp_precip_sum.plot(ax=ax4, fontsize=fontsize, style=['-', '-'], color=color, linewidth=2,
            #                      markersize=0.5, xlim=Xlim_wtd)
            #plt.ylabel('Cumulative precipitation [mm/d]')

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        if (site == 'Taufik_UpperSebangau_PSF_daily') or (site == 'Kalteng1') or (site == 'QT-2010-1') or (
                site == 'Itanga_avg'):
            fontsize = 24
            df_tmp_wtd = df_tmp_wtd[['data_mod', 'data_obs']]
            df_tmp_wtd.plot(figsize=(20, 5), fontsize=fontsize, style=['-', '.'], color=color, linewidth=3,
                            markersize=5.5, xlim=Xlim_wtd, legend=False)
            plt.ylabel('WTD (m)', fontsize=fontsize)
            plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)  # minor xticks
            plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True)  # minor xticks
            plt.axhline(y=0, xmin=0, xmax=1, color='grey', linestyle='--')
            plt.ylim([-4, 1])
            plt.tight_layout()
            fname = site + '_' + exp
            fname_long = os.path.join(
                '/data/leuven/324/vsc32460/FIG/in_situ_comparison/paper/timeseries_WTD/' + fname + '.png')
            plt.savefig(fname_long, dpi=350)
            plt.close()
        else:
            print('not sites for paper')

        # plot the anomalies
        if threeYears >= 1095:
            plt.figure(figsize=(20, 5))
            fontsize = 12
            anomalies.plot(fontsize=fontsize, style=['-', '.'], color=color, linewidth=2, markersize=4.5,
                           xlim=Xlim_wtd)
            plt.axhline(y=0, xmin=0, xmax=1, color='grey', linestyle='--')
            plt.ylabel('WTD anomalies (m)')
            plt.tight_layout()
            fname = site
            fname_long = os.path.join(outpath + '/anomaly_' + fname + '.png')
            plt.savefig(fname_long, dpi=150)
            plt.close()
        else:
            print('No anomaly calculation, too short timeseries')

        # fig2
        plt.figure(figsize=(16, 9))
        fontsize = 12
        moisture_all = pd.concat(
            (wtd_mod[site], sfmc_mod[site], rzmc_mod[site], srfexc[site], rzexc[site], catdef[site]), axis=1,
            join='inner')
        moisture_all.columns = ['zbar', 'sfmc', 'rzmc', 'srfexc', 'rzexc', 'catdef']

        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax1, y='zbar', x='catdef', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('zbar', fontsize=20)
        plt.xlabel('catdef', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax2, y='zbar', x='rzexc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('zbar', fontsize=20)
        plt.xlabel('rzexc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax3, y='zbar', x='rzmc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('zbar', fontsize=20)
        plt.xlabel('rzmc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax4 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax4, y='zbar', x='srfexc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('zbar', fontsize=20)
        plt.xlabel('srfexc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax5 = plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax5, y='zbar', x='sfmc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('zbar', fontsize=20)
        plt.xlabel('sfmc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax6 = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax6, y='catdef', x='srfexc', fontsize=fontsize, style=['.'], color=color_short,
                          markersize=4)
        plt.ylabel('catdef', fontsize=20)
        plt.xlabel('srfexc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax7 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax7, y='catdef', x='rzexc', fontsize=fontsize, style=['.'], color=color_short,
                          markersize=4)
        plt.ylabel('catdef', fontsize=20)
        plt.xlabel('rzexc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax8 = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax8, y='srfexc', x='sfmc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('srfexc', fontsize=20)
        plt.xlabel('sfmc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax9 = plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1, fig=None)
        moisture_all.plot(ax=ax9, y='rzexc', x='rzmc', fontsize=fontsize, style=['.'], color=color_short, markersize=4)
        plt.ylabel('rzexc', fontsize=20)
        plt.xlabel('rzmc', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/moistureVar' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # fig3
        plt.figure(figsize=(16, 8.5))
        fontsize = 12

        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3, fig=None)
        ax1.set_ylabel('catdef (mm)')
        catdef[site] = -catdef[site]
        catdef[site].plot(ax=ax1, fontsize=fontsize, style=['--'], color=color_short, linewidth=2, markersize=4.5,
                          xlim=Xlim_wtd)
        ax4 = ax1.twinx()
        ax4.set_ylabel('Moisture content (m³/m³)')
        rzmc_mod[site].plot(ax=ax4, fontsize=fontsize, style=['--'], color='cyan', linewidth=1.5,
                            xlim=Xlim_wtd)  # ifzbar is not showing add: (ax=ax5, x_compat=True,...
        sfmc_mod[site].plot(ax=ax4, fontsize=fontsize, style=['--'], color='palegreen', linewidth=1.5, xlim=Xlim_wtd)
        legendLines = [matplotlib.lines.Line2D([0], [0], color=color_short, linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='cyan', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='palegreen', linewidth=2)]
        ax1.legend(legendLines, ['catdef', 'rzmc', 'sfmc'])

        ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=3, fig=None)
        ax2.set_ylabel('Moisture content (m³/m³)')
        rzmc_mod[site].plot(ax=ax2, fontsize=fontsize, style=['--'], color='cyan', linewidth=1.5,
                            xlim=Xlim_wtd)  # ifzbar is not showing add: (ax=ax5, x_compat=True,...
        sfmc_mod[site].plot(ax=ax2, fontsize=fontsize, style=['--'], color='palegreen', linewidth=1.5, xlim=Xlim_wtd)
        ax5 = ax2.twinx()
        ax5.set_ylabel('Moisture transfer (mm)')
        srfexc[site].plot(ax=ax5, fontsize=fontsize, style=['--'], color='b', linewidth=1.5, xlim=Xlim_wtd)
        rzexc[site].plot(ax=ax5, fontsize=fontsize, style=['--'], color='g', linewidth=1.5, xlim=Xlim_wtd)
        legendLines = [matplotlib.lines.Line2D([0], [0], color='cyan', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='palegreen', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='b', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='g', linewidth=2)]
        ax2.legend(legendLines, ['rzmc', 'sfmc', 'srfexc', 'rzexc'])

        ax3 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3, fig=None)
        ax3.set_ylabel('catdef (mm)')
        catdef[site].plot(ax=ax3, fontsize=fontsize, style=['--'], color=color_short, linewidth=1.5, xlim=Xlim_wtd)
        ax6 = ax3.twinx()
        ax6.set_ylabel('Precipitation + RZEXC (mm/day)')
        precip_mod[site].plot(ax=ax6, fontsize=fontsize, style=['--'], color='gray', linewidth=1.5, xlim=Xlim_wtd)
        rzexc[site].plot(ax=ax6, fontsize=fontsize, style=['--'], color='g', linewidth=1.5, xlim=Xlim_wtd)
        plt.ylim([-100, 100])
        legendLines = [matplotlib.lines.Line2D([0], [0], color=color_short, linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='g', linewidth=2),
                       matplotlib.lines.Line2D([0], [0], color='gray', linewidth=2)]
        ax3.legend(legendLines, ['catdef', 'rzexc', 'precip'])

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/moistureTS' + fname + '.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # SA! added this temporarly for Dwi, needs this in fire simulations
        if (site == 'Dwi_1'):
            Dwi_wtd= wtd_mod[site]
            Dwi_rzmc= rzmc_mod[site]
            Dwi_sfmc= sfmc_mod[site]
            Dwi_precip= precip_mod[site]
            df_Dwi=pd.concat([Dwi_wtd, Dwi_rzmc, Dwi_sfmc,Dwi_precip],axis=1)
            df_Dwi = df_Dwi.loc['2000-1-1':'2020-12-31']
            print(df_Dwi)
            path_changing = os.path.join(outpath + '_Dwi.csv')
            df_Dwi.to_csv(path_changing, index=True, header=['WTD [m]', 'RZMC [m3/m3]','SFMC [m3/m3]','PRECIP [mm]'])
        else:
             print('ok')

    df_allmetrics = df_metrics.join(df_metrics_P)
    df_allmetrics_new = df_allmetrics[df_allmetrics['R_P (-)'] > 0.28].dropna()
    df_all_biasP = ((df_allmetrics_new['bias_P (m)']) - df_allmetrics_new['bias_P (m)'].min()) / (
            df_allmetrics_new['bias_P (m)'].max() - df_allmetrics_new['bias_P (m)'].min())
    df_all_RWTD = (df_allmetrics_new['R (-)'] - df_allmetrics_new['R (-)'].min()) / (
            df_allmetrics_new['R (-)'].max() - df_allmetrics_new['R (-)'].min())

    df_all_RWTD = df_allmetrics_new['R (-)']
    df_all_biasP = pd.DataFrame(df_all_biasP)
    df_all_RWTD = pd.DataFrame(df_all_RWTD)
    df_allWTDP = df_all_biasP.join(df_all_RWTD)

    # Plot boxplot for metrics of WTD only
    plt.figure()
    df_metrics.boxplot()
    fname = 'metrics'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    mean = df_metrics.mean()
    mean = mean.round(decimals=3)
    Title = 'mean bias (m) = ' + str(mean['bias (m)']) + ', mean ubRMSD (m) = ' + str(
        mean['ubRMSD (m)']) + ', mean R (-) = ' + str(mean['R (-)']) + '\n' + ' mean RMSD (m) = ' + str(
        mean['RMSD (m)']) + ', mean anomR (m) = ' + str(mean['anomR (-)'])
    plt.title(Title, fontsize=9)
    plt.savefig(fname_long, dpi=150)
    plt.close()

    # Plot boxplot for metrics of WTD only
    plt.figure()
    df_metrics_P.boxplot()
    mean = df_metrics_P.mean()
    mean = mean.round(decimals=2)
    Title = 'mean bias (m) = ' + str(mean['bias_P (m)']) + ', mean ubRMSD (m) = ' + str(
        mean['ubRMSD_P (m)']) + ', mean R (-) = ' + str(mean['R_P (-)']) + '\n' + ' mean RMSD (m) = ' + str(
        mean['RMSD_P (m)'])
    plt.title(Title, fontsize=9)
    plt.savefig('/data/leuven/324/vsc32460/FIG/in_situ_comparison/Data_assimilation/CO/Precipitation_boxplot_SEA.png', dpi=250)
    plt.close()

    # Sebastian added this to export the skillmetrics of each run so that it can be combined in 1 figure
    skillpath = '/data/leuven/324/vsc32460/FIG/'
    fname = 'skillmetrics_' + 'M09' + exp + '.csv'
    df_metrics.to_csv(skillpath + fname, index=True, header=True)
    fname = 'skillmetrics_' + 'M09' + exp + '_CI.csv'
    df_metrics_CI.to_csv(skillpath + fname, index=True, header=True)

    # comparison of abs bias P to WTD R
    if (exp == 'INDONESIA_M09_PEATCLSMTN_v01'):
        # linear regression for the data and correlation
        xtest = pd.DataFrame(df_allWTDP['bias_P (m)'])
        ytest = pd.DataFrame(df_allWTDP['R (-)'])
        reg = linear_model.LinearRegression().fit(xtest, ytest)
        reg.fit(xtest, ytest)
        m = reg.coef_[0]
        b = reg.intercept_
        print("formula: y ={0}x+{1}".format(m, b))
        Pearson_2 = df_allWTDP.corr(method='pearson')
        Pearson_2_value = Pearson_2.iloc[0]['R (-)']
        Pearson_2_value = Pearson_2_value.round(decimals=3)
        print(Pearson_2_value)

        plt.figure()
        fname = 'WTD_precip'
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
        df_allWTDP.plot(fontsize=fontsize, x='bias_P (m)', y='R (-)', style=['.'], color='r', markersize=4,
                        label='Pearson R')
        plt.plot([0, 4.2], [(b), (b + (m * (4.2)))], color='b')
        plt.ylabel('R (groundwater level)')
        plt.xlabel('Mean absolute precipitation error (cm)')
        plt.xlim(left=0, right=4)
        plt.legend(fontsize=fontsize)
        Title = ("y ={0}x+{1}".format(m, b) + ' , R =' + str(Pearson_2_value))
        plt.title(Title)
        plt.savefig(fname_long, dpi=150)
        plt.close()

        # boxplot for RWTD and ABSBP of good and bad
        df_allWTDP1 = df_allWTDP.copy()
        df_allWTDP1.insert(0, 'Group', ['B', 'B', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'A', 'A'], True)
        df_allWTDP1 = df_allWTDP1.drop(['bias_P (m)'], axis=1)
        plt.figure(figsize=(5.5, 5))
        ax = sns.boxplot(x="Group", y="R (-)", data=df_allWTDP1, palette=["#e74c3c", "#2ecc71"], width=0.7)
        fname = 'grouped_boxplot1'
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
        # Title =('Comparing the influence of the absolute bias between modeled'+ '\n' + ' and observed precipitation on the correlation coefficient' + '\n' + ' of the modeled and observed water level'+ '\n')
        # plt.title(Title, fontsize=9.5)
        # handles, _ = ax.get_legend_handles_labels()
        # ax.legend.remove()
        ax.set(xlabel=None,
               xticklabels=["High\n                          mean absolute error in precipitation", "Low \n"])
        plt.ylabel("R (water table depth)", fontsize=20)
        plt.xlabel("")
        plt.tick_params('both', labelsize=18.5)
        plt.savefig(fname_long, dpi=150, bbox_inches='tight')
        plt.close()

        # grouped boxplot for RWTD and ABSBP of good and bad NORMALIZED
        df_all_biasP1 = (df_allmetrics_new['abs_bias_P (m)'] - df_allmetrics_new['abs_bias_P (m)'].min()) / (
                df_allmetrics_new['abs_bias_P (m)'].max() - df_allmetrics_new['abs_bias_P (m)'].min())
        df_all_RWTD1 = (df_allmetrics_new['Pearson_R (-)'] - df_allmetrics_new['Pearson_R (-)'].min()) / (
                df_allmetrics_new['Pearson_R (-)'].max() - df_allmetrics_new['Pearson_R (-)'].min())
        df_all_biasP1 = pd.DataFrame(df_all_biasP1)
        df_all_RWTD1 = pd.DataFrame(df_all_RWTD1)
        df_allWTDP1 = df_all_biasP1.join(df_all_RWTD1)
        df_allWTDP1.insert(0, 'Group', ['B', 'B', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'A', 'A'], True)
        plt.figure()
        df_long = pd.melt(df_allWTDP1, "Group")
        sns.boxplot(x="variable", y="value", data=df_long, hue="Group", palette=["#e74c3c", "#2ecc71"])
        fname = 'grouped_boxplot_normalized'
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
        Title = 'Normalized Grouped boxplot'
        plt.title(Title, fontsize=9)
        plt.savefig(fname_long, dpi=150)
        plt.close()

    # sebastian added     # save skillmetrics in csvfile
    # path_changing = os.path.join(outpath + '/comparison_insitu_data/' + 'skillmetrics_parameters.csv')
    # skillmetrics_parameters_csv = df_metrics.to_csv (path_changing, index = True, header=True)


def plot_skillmetrics_comparison_wtd_multimodel(wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc,
                                                rzexc, catdef, exp, outpath):
    # Initiate dataframe to store metrics in.
    INDEX = wtd_obs.columns
    COL = ['bias', 'ubRMSD', 'Pearson_R', 'RMSD']
    df_metrics = pd.DataFrame(index=INDEX, columns=COL, dtype=object)

    for c, site in enumerate(wtd_obs.columns):
        df_tmp = pd.concat((wtd_obs[site], wtd_mod[1][site]))
        df_tmp.columns = ['In-situ', 'WHITELIST']
        try:
            df_tmp = pd.concat((df_tmp, wtd_mod[2][site]), axis=1)
            df_tmp.columns = ['In-situ', 'WHITELIST', '$PEATCLSM_{T,Natural}$ - IN']
        except:
            df_tmp = df_tmp

        try:
            df_tmp = pd.concat((df_tmp, wtd_mod[3][site]), axis=1)
            df_tmp.columns = ['In-situ', 'WHITELIST', '$PEATCLSM_{T,Natural}$ - CO']
        except:
            df_tmp = df_tmp

        try:
            df_tmp = pd.concat((df_tmp, wtd_mod[2][site], wtd_mod[3][site]), axis=1)
            df_tmp.columns = ['In-situ', 'PEATCLSM_{N}', '$PEATCLSM_{T,Natural}$ ', '$PEATCLSM_{T,Drained}$']
        except:
            df_tmp = df_tmp

        df_tmp2 = copy.deepcopy(df_tmp)

        bias_site = metrics.bias(df_tmp2)  # Bias = bias_site[0]
        ubRMSD_site = metrics.ubRMSD(df_tmp2)  # ubRMSD = ubRMSD_site[0]
        pearson_R_site = metrics.Pearson_R(df_tmp2)  # Pearson_R = pearson_R_site[0]
        # RMSD_site = (ubRMSD_site[0] ** 2 + bias_site[0] ** 2) ** 0.5

        # Save metrics in df_metrics.
        # df_metrics.loc[site]['bias'] = bias_site[0]
        # df_metrics.loc[site]['ubRMSD'] = ubRMSD_site[0]
        # df_metrics.loc[site]['Pearson_R'] = pearson_R_site[0]
        # df_metrics.loc[site]['RMSD'] = RMSD_site

        # Create x-axis matching in situ data.
        x_start = df_tmp2.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end = df_tmp2.index[-1]  # End a-axis with the last day with an observed wtd value.
        Xlim = [x_start, x_end]

        # Calculate z-score for the time series.
        df_zscore = df_tmp2.dropna(axis=0).apply(zscore)

        plt.figure(figsize=(16, 6.5))
        fontsize = 12

        ax1 = plt.subplot(311)
        df_tmp = df_tmp[['$PEATCLSM_N$', '$PEATCLSM_{T,Natural}$', '$PEATCLSM_{T,Drained}$', 'In-situ']]
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['-', '-', '-', '.'], color=['y', 'g', 'm', '#1f77b4'],
                    linewidth=1.7, xlim=Xlim)
        plt.ylabel('zbar [m]')

        plt.title(site)

        ax1 = plt.subplot(312)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['-', '-', '-', '.'], color=['y', 'g', 'm', '#1f77b4'],
                    linewidth=1.7, xlim=[df_zscore.index[0], df_zscore.index[-1]])
        plt.ylabel('zbar [m]')

        ax2 = plt.subplot(313)
        df_zscore = df_zscore[['$PEATCLSM_N$', '$PEATCLSM_{T,Natural}$', '$PEATCLSM_{T,Drained}$', 'In-situ']]
        df_zscore.plot(ax=ax2, fontsize=fontsize, style=['-', '-', '-', '.'], color=['y', 'g', 'm', '#1f77b4'],
                       linewidth=1.7, xlim=[df_zscore.index[0], df_zscore.index[-1]])
        plt.ylabel('z-score')

        plt.legend()

        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '_multimodel.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()

    # Plot boxplot for metrics
    # plt.figure()
    # df_metrics.boxplot()
    # fname = 'metrics'
    # fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    # plt.savefig(fname_long, dpi=150)
    # plt.close()


def plot_delta_spinup(exp1, exp2, domain, root, outpath):
    # Hugo Rudebeck:
    # Funtion to plot the difference between two runs. Takes the difference between the all the sites for the two runs and
    # saves the maximum difference for every time step. Does this for all variables.

    outpath = os.path.join(outpath, exp1, 'delta_spinup')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io_spinup1 = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    io_spinup2 = LDAS_io('daily', exp=exp2, domain=domain,
                         root=root)  # This path is not correct anymore, the long nc cubes for the 2nd spinup are now
    # stored in /ddn1/vol1/staging/leuven/stg_00024/OUTPUT/hugor/output/INDONESIA_M09_v01_spinup2/ as daily_images_2000-2019.nc and daily_timeseries_2000-2019.nc.
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io_spinup1)

    ntimes = len(io_spinup1.images['time'])
    delta_var = np.zeros([ntimes])

    for i, cvar in enumerate(io_spinup1.images.data_vars):  # Loop over all variables in cvar.

        unit = assign_units(cvar)  # Assign units to cvar.

        for t in range(ntimes):
            # logging.info('time %i of %i' % (t, ntimes))
            diff = io_spinup1.images[cvar][t, :, :] - io_spinup2.images[cvar][t, :, :]
            diff_1D = diff.values.ravel()  # Make a 1D array.
            diff_1D_noNaN = diff_1D[~np.isnan(diff_1D)]  # Remove nan.
            diff_1D_sorted = np.sort(diff_1D_noNaN.__abs__())  # Sort the absolute values.
            delta_var[t] = diff_1D_sorted[
                -2]  # Take the second largest absolute value, due to problems with instability
            # in one grid cell causing abCLSM_normal values.

            # # searching for strange values evaporation
            # if cvar == 'evap' and A.data.item() > 10 and t > 2000:
            #     # print(t)
            #     print(np.where(np.abs(diff.values) > 10))

        # Plot the maximum difference between the two runs and save figure in 'outpath'.
        plt.plot(delta_var, linewidth=2)
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
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

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

    df = pd.concat((ts_mod_wtd, ts_mod_sfmc), axis=1)

    plt.figure(figsize=(19, 8))

    ax1 = plt.subplot(211)
    df[['zbar (mod)']].plot(ax=ax1, fontsize=fontsize, style=['.-'], linewidth=2)
    plt.ylabel('zbar [m]')

    ax2 = plt.subplot(212)
    df[['sfmc (mod)']].plot(ax=ax2, fontsize=fontsize, style=['.-'], linewidth=2)
    plt.ylabel('sfmc $\mathregular{[m^3/m^3]}$')

    my_title = 'lon=%s, lat=%s poros=%.2f' % (lon, lat, siteporos)
    plt.title(my_title, fontsize=fontsize + 2)
    plt.tight_layout()
    fname = 'wtd_sfmc_lon_%s_lat_%s' % (lon, lat)
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_peat_and_sites(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp, 'maps', 'catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
    # RTMparams = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    # land fraction
    frac_cell = io.grid.tilecoord.frac_cell.values
    param = 'poros'
    # fraction of peatlands with less than 5 % open water
    # frac_peatland = np.nansum(np.all(np.vstack((frac_cell>0.5,params['poros'].values>0.65)),axis=0))/np.nansum(params['poros'].values>0.01)
    # frac_peatland_less5 = np.nansum(np.all(np.vstack((frac_cell>0.95,params['poros'].values>0.65)),axis=0))/np.nansum(params['poros'].values>0.65)
    # set more than 5% open water grid cells to 1.
    params[param].values[np.all(np.vstack((frac_cell < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0
    params[param].values[params['poros'].values < 0.65] = np.nan
    params[param].values[
        np.all(np.vstack((params['poros'].values < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params[param].values
    data = np.ma.masked_invalid(img)
    fname = 'global_peat_distribution_' + param
    cmin = 0
    cmax = 1
    # title='Peatland distribution and tropical in situ sites'
    # open figure
    figsize = (45, 20)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    cmap = matplotlib.colors.ListedColormap([85 / 255, 10 / 255, 10 / 255])
    plt_img = np.ma.masked_invalid(data)
    # for tropical latitudes and longitudes only, else comment line out and savefig without 'tropics'
    [llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = [-20, 20, -100, 170]
    m = Basemap(projection='cyl', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='i')  # cyl or cea projection for tropics
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries(linewidth=0.3)
    parallels = np.arange(-80.0, 81, 10.)
    m.drawparallels(parallels, linewidth=0.4, labels=[True, False, False, False], color='grey', fontsize=35)
    meridians = np.arange(0., 351., 20.)
    m.drawmeridians(meridians, linewidth=0.4, labels=[False, False, True, False], color='grey', fontsize=35)

    # draw 3 rectangles on map
    # function to draw a rectangle (or 3: IN, CO, SA)
    def draw_screen_poly(LON, LAT, m):
        x, y = m(LON, LAT)
        xy = zip(x, y)
        poly = Polygon(list(xy), facecolor='red', alpha=0.2, linestyle='-', edgecolor='k', linewidth=2.8)
        plt.gca().add_patch(poly)

    # CO
    LAT = [-3.5, 3.5, 3.5, -3.5]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [15.0, 15.0, 22.0, 22.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    draw_screen_poly(LON, LAT, m)
    # IN
    LAT = [-11.5, 7.5, 7.5, -11.5]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [94.0, 94.0, 155.0, 155.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    draw_screen_poly(LON, LAT, m)
    # SA
    LAT = [-8.0, 13.0, 13.0, -8.0]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [-87.0, -87.0, -45.0,
           -45.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    draw_screen_poly(LON, LAT, m)

    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)

    fname_long = os.path.join(outpath, fname + 'tropics.png')
    # plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()

    def zoom_ins(exp, domain, root, outpath):
        # read in the ldas data
        io = LDAS_io('catparam', exp=exp, domain=domain, root=root)
        [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
        tc = io.grid.tilecoord
        params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
        # RTMparams = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
        # land fraction
        frac_cell = io.grid.tilecoord.frac_cell.values
        param = 'poros'

        # set more than 5% open water grid cells to 1.
        params[param].values[np.all(np.vstack((frac_cell < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0
        params[param].values[params['poros'].values < 0.65] = np.nan
        params[param].values[
            np.all(np.vstack((params['poros'].values < 0.95, params['poros'].values > 0.65)), axis=0)] = 1.0

        # creat the image to plot the peatland distribution
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        plt_img = np.ma.masked_invalid(data)

        # open figure and set shapes and colors
        if exp[:2] == 'IN':
            figsize = (30, 20)
        else:
            figsize = (25, 25)
        fontsize = 14
        f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        cmap = matplotlib.colors.ListedColormap([85 / 255, 10 / 255, 10 / 255])
        # line below is to get the new views, if not wanted, comment out and savefig without 'tropics'
        [llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = [LAT[0], LAT[1], LON[0], LON[2]]
        m = Basemap(projection='cyl', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon, resolution='i')  # cyl or cea projection for tropics
        if exp[:2] == 'SA' or exp[:2] == 'IN':
            m.drawcoastlines(linewidth=1.8)
            m.drawcountries(linewidth=2)
        else:
            m.drawcountries(linewidth=3.2)

        if exp[:2] == 'SA':
            parallels = np.arange(-80.0, 81, 5)
            m.drawparallels(parallels, linewidth=1, labels=[True, False, False, False], color='grey',
                            fontsize=80)
            meridians = np.arange(0., 351., 10)
            m.drawmeridians(meridians, linewidth=1, labels=[False, False, False, True], color='grey',
                            fontsize=80)
        elif exp[:2] == 'IN':
            parallels = np.arange(-80.0, 81, 5)
            m.drawparallels(parallels, linewidth=1, labels=[True, False, False, False], color='grey',
                            fontsize=64)
            meridians = np.arange(0., 351., 10)
            m.drawmeridians(meridians, linewidth=1, labels=[False, False, False, True], color='grey',
                            fontsize=64)
        else:
            parallels = np.arange(-80.0, 81, 2)
            m.drawparallels(parallels, linewidth=1.8, labels=[True, False, False, False], color='grey',
                            fontsize=132)
            meridians = np.arange(0., 351., 2)
            m.drawmeridians(meridians, linewidth=1.8, labels=[False, False, False, True], color='grey',
                            fontsize=132)

        # load in situ coordinates of WTD
        insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
        mastertable_filename = 'WTD_TROPICS_MASTER_TABLE_ALLDorN.csv'
        filenames = find_files(insitu_path, mastertable_filename)
        if isinstance(find_files(insitu_path, mastertable_filename), str):
            master_table = pd.read_csv(filenames, sep=';')
        else:
            for filename in filenames:
                if filename.endswith('csv'):
                    master_table = pd.read_csv(filename, sep=';')
                    cond = master_table['comparison_yes'] == 1
                    continue
                else:
                    logging.warning("some files, maybe swp files, exist that start with master table searchstring !")

        # plot in situ coordinates
        master_table_N = master_table[master_table['comparison_yes'] == 1]
        master_table_SA = master_table_N[master_table_N['lon'] < -20]
        master_table_CO = master_table_N[(master_table_N['lon'] >= -20) & (master_table_N['lon'] < 60)]
        master_table_IN = master_table_N[master_table_N['lon'] >= 60]
        master_table_IN_D = master_table_IN[master_table_IN['drained_U=uncertain'] == 'D']
        master_table_IN_N = master_table_IN[master_table_IN['drained_U=uncertain'] == 'N']

        insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/ET'
        mastertable_filename = 'ET_TROPICS_MASTER_TABLE.csv'
        filenames = find_files(insitu_path, mastertable_filename)
        if isinstance(find_files(insitu_path, mastertable_filename), str):
            master_table_ET = pd.read_csv(filenames, sep=';')
        else:
            for filename in filenames:
                if filename.endswith('csv'):
                    master_table_ET = pd.read_csv(filename, sep=';')
                    cond = master_table_ET['comparison_yes'] == 1
                    continue
                else:
                    logging.warning("some files, maybe swp files, exist that start with master table searchstring !")

        # plot in situ coordinates
        master_table_N_ET = master_table_ET[master_table_ET['comparison_yes'] == 1]
        master_table_SA_ET = master_table_N_ET[master_table_N_ET['lon'] < -20]
        master_table_IN_ET = master_table_N_ET[master_table_N_ET['lon'] >= 60]

        # x, y = m(master_table['lon'].values[cond], master_table['lat'].values[cond])
        if exp[:2] == 'IN':
            m.plot(master_table_IN_D['lon'], master_table_IN_D['lat'], 'o', markerfacecolor='magenta',
                   markersize=fontsize + 4, markeredgecolor='k', markeredgewidth=1.2)
            m.plot(master_table_IN_N['lon'], master_table_IN_N['lat'], 'o', markerfacecolor='limegreen',
                   markeredgecolor='k', markersize=fontsize + 4, markeredgewidth=1.2)
            m.plot(master_table_IN_ET['lon'], master_table_IN_ET['lat'], 'o', markerfacecolor='None',
                   markeredgecolor='b', markersize=fontsize + 6.5, markeredgewidth=3.5)
        elif exp[:2] == 'SA':
            m.plot(master_table_SA['lon'], master_table_SA['lat'], 'o', markerfacecolor='limegreen',
                   markeredgecolor='k', markersize=fontsize + 14, markeredgewidth=1.8)
            m.plot(master_table_SA_ET['lon'], master_table_SA_ET['lat'], 'o', markerfacecolor='None',
                   markeredgecolor='b', markersize=fontsize + 19, markeredgewidth=8)
        else:
            m.plot(master_table_CO['lon'], master_table_CO['lat'], 'o', markerfacecolor='limegreen',
                   markersize=fontsize + 26)
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)

        # save the figure
        fname = exp[:2] + '_peat_sites_' + param
        fname_long = os.path.join(outpath, fname + 'tropics.png')
        plt.tight_layout()
        plt.savefig(fname_long, dpi=f.dpi)
        plt.close()

    # zoom-ins adjust experiment to the zoom in
    # CO
    root = '/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'CONGO_M09_PEATCLSMTN_v01'
    # to plot new, smaller views, else comment out
    LAT = [-3.5, 3.5, 3.5, -3.5]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [15.0, 15.0, 22.0, 22.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    zoom_ins(exp, domain, root, outpath)

    # IN
    root = '/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'INDONESIA_M09_PEATCLSMTN_v01'
    # to plot new, smaller views, else comment out
    LAT = [-11.5, 7.5, 7.5, -11.5]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [94.0, 94.0, 155.0, 155.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    zoom_ins(exp, domain, root, outpath)

    # SA
    root = '/staging/leuven/stg_00024/OUTPUT/sebastiana'
    exp = 'SAMERICA_M09_PEATCLSMTN_v01'
    # to plot new, smaller views, else comment out
    LAT = [-8.0, 13.0, 13.0, -8.0]  # minlat, maxlat, maxlat, minlat OR bottom left, top left, top right, bottom right
    LON = [-87.0, -87.0, -45.0,
           -45.0]  # minlon, minlon, maxlon, maxlon OR bottom left, top left, top right, bottom right
    zoom_ins(exp, domain, root, outpath)


def plot_peat_and_sites_NORTH_RSE_paper(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp, 'maps', 'catparam')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
    # RTMparams = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    # land fraction
    frac_cell = io.grid.tilecoord.frac_cell.values
    param = 'poros'
    # fraction of peatlands with less than 5 % open water
    frac_peatland = np.nansum(np.all(np.vstack((frac_cell > 0.5, params['poros'].values > 0.8)), axis=0)) / np.nansum(
        params['poros'].values > 0.01)
    frac_peatland_less5 = np.nansum(
        np.all(np.vstack((frac_cell > 0.95, params['poros'].values > 0.8)), axis=0)) / np.nansum(
        params['poros'].values > 0.8)
    # set more than 5% open water grid cells to 1.
    params[param].values[np.all(np.vstack((frac_cell < 0.95, params['poros'].values > 0.8)), axis=0)] = 1.
    params[param].values[params['poros'].values < 0.8] = 0.0
    params[param].values[np.all(np.vstack((params['poros'].values < 0.95, params['poros'].values > 0.8)), axis=0)] = 0.5
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params[param].values
    data = np.ma.masked_invalid(img)

    latmin = 45.
    latmax = 70.
    lonmin = -170.
    lonmax = 95.
    [data, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data, lons, lats, latmin, latmax,
                                                                                 lonmin, lonmax)
    fname = '01b_' + param
    cmin = 0
    cmax = 1
    title = 'Soil map and in-situ data'
    # open figure
    figsize = (0.85 * 13, 0.85 * 10)
    fontsize = 13
    f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    cmap = matplotlib.colors.ListedColormap(
        [[231. / 255., 247. / 255., 213. / 255.], [255. / 255, 193. / 255, 7. / 255],
         [30. / 255, 136. / 255, 229. / 255]])
    # cmap = matplotlib.colors.ListedColormap([[207./255.,207./255.,207./255.],[255./255,188./255,121./255],[162./255,200./255,236./255]])
    # cmap = 'jet'
    cbrange = (cmin, cmax)
    ax = plt.subplot(3, 1, 3)
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='l')
    # m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=-170.,urcrnrlon=urcrnrlon,resolution='l')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    parallels = np.arange(-80.0, 81, 5.)
    m.drawparallels(parallels, linewidth=0.5, labels=[True, False, False, False])
    meridians = np.arange(0., 351., 20.)
    m.drawmeridians(meridians, linewidth=0.5, labels=[False, False, False, True])
    m.readshapefile('/data/leuven/317/vsc31786/gis/permafrost/permafrost_boundary', 'permafrost_boundary',
                    linewidth=1.3, color=(0. / 255, 0. / 255, 0. / 255))
    # load peatland sites
    sites = pd.read_csv('/data/leuven/317/vsc31786/FIG_tmp/00DA/20190228_M09/cluster_radius_2_bog.txt', sep=',')
    # http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
    # lat=48.
    # lon=51.0
    x, y = m(sites['Lon'].values, sites['Lat'].values)
    if np.mean(lats) > 40:
        m.plot(x, y, '.', color=(181. / 255, 2. / 255, 31. / 255), markersize=12, markeredgewidth=1.5, mfc='none')
    else:
        insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
        mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
        filenames = find_files(insitu_path, mastertable_filename)
        if isinstance(find_files(insitu_path, mastertable_filename), str):
            master_table = pd.read_csv(filenames, sep=';')
        else:
            for filename in filenames:
                if filename.endswith('csv'):
                    master_table = pd.read_csv(filename, sep=';')
                    cond = master_table['comparison_yes'] == 1
                    continue
                else:
                    logging.warning("some files, maybe swp files, exist that start with master table searchstring !")
        x, y = m(master_table['lon'].values[cond], master_table['lat'].values[cond])
        m.plot(x, y, '.', color=(0. / 255, 0. / 255, 0. / 255), markersize=12, markeredgewidth=1.5, mfc='none')

    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    # cb=m.colorbar(im,"bottom",size="7%",pad="22%",shrink=0.5)
    # cb=matplotlib.pyplot.colorbar(im)
    im_ratio = np.shape(data)[0] / np.shape(data)[1]
    cb = matplotlib.pyplot.colorbar(im, fraction=0.13 * im_ratio, pad=0.02)
    # ticklabs=cb.ax.get_yticklabels()
    # cb.ax.set_yticklabels(ticklabs,ha='right')
    # cb.ax.yaxis.set_tick_params(pad=45)#yournumbermayvary
    # labelsize
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
        t.set_horizontalalignment('right')
        t.set_x(9.0)
    tit = plt.title(title, fontsize=fontsize)
    # matplotlib.pyplot.text(1.0,1.0,mstats[i],horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,fontsize=fontsize)
    fname_long = os.path.join(outpath, fname + '.png')
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
        os.makedirs(outpath, exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    latmin = 40.
    latmax = 70.
    lonmin = -170.
    lonmax = -50.
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # mean
    for varname, da in io.images.data_vars.items():
        if varname not in ['evap', 'Tair', 'Rainf', 'zbar']:
            continue
        tmp_data = da

        # JJA_annual = tmp_data.where((tmp_data['time.season'] == 'JJA')).groupby('time.year').max(dim='time')
        JJA_annual = tmp_data.where((tmp_data['time.month'] >= 7) & (tmp_data['time.month'] <= 9)).groupby(
            'time.year').max(dim='time')
        # JJA_annual = tmp_data.where((tmp_data['time.month'] >= 1) & (tmp_data['time.month'] <= 12)).groupby('time.year').max(dim='time')
        vals = JJA_annual.variable.values
        years = JJA_annual.year
        # col, row = io.grid.lonlat2colrow(-107.5, 54.8, domain=True)
        # plt.figure(figsize=(8, 8.5))
        # plt.plot(years,vals[:,row,col],'.')
        # plt.savefig(outpath+'/'+varname+'_x.png',dpi=150)
        vals2 = vals.reshape(len(years), -1)
        mask1 = (np.isnan(vals2)).any(axis=0) == False
        # Do a first-degree polyfit
        idx = np.where((np.isnan(vals2)).any(axis=0) == False)[0]
        # stats.mstats.theilslopes(vals2[:,idx],years)
        regressions_tmp = np.polyfit(years, vals2[:, idx], 1)
        regressions_tmp_lo = np.polyfit(years, vals2[:, idx], 1)
        regressions_tmp_up = np.polyfit(years, vals2[:, idx], 1)
        for counter, value in enumerate(idx):
            ctheil = stats.mstats.theilslopes(vals2[:, value], years, alpha=0.90)
            regressions_tmp_lo[0, counter] = ctheil[2]
            regressions_tmp_up[0, counter] = ctheil[3]
            result = mk.original_test(vals2[:, value])
            # if (regressions_tmp_lo[0,counter] * regressions_tmp_up[0,counter]) >= 0:
            if result[0] != 'no trend':
                regressions_tmp[0, counter] = ctheil[0]
            else:
                regressions_tmp[0, counter] = 0.
        regressions = np.zeros([2, vals2.shape[1]])
        regressions[:] = np.nan
        regressions[:, idx] = regressions_tmp
        # Get the coefficients back
        trends = regressions[0, :].reshape(vals.shape[1], vals.shape[2])
        cmap = plt.get_cmap('PiYG')
        cmin = np.min([np.nanmin(trends), -np.nanmax(trends)])
        cmax = -cmin
        plot_title = varname
        # if varname=='zbar':
        #    cmin=-0.6
        #    cmax=-0.1
        # if varname=='runoff':
        #    cmin=0
        #    cmax=5
        # if varname=='evap':
        #    cmin=0
        #    cmax=5
        plot_title = varname + " " + assign_units(varname)
        fname = varname + '_trend'
        # plot_title='zbar [m]'
        figure_single_default(data=trends, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=plot_title)


def plot_skillmetrics_comparison_wtd_DA(wtd_obs, wtd_mod, outpath, exp_DA, domain, root):
    # Initiate dataframe to store metrics in.
    INDEX = wtd_obs.columns
    COL = ['bias (m)', 'RMSD (m)', 'ubRMSD (m)', 'R (-)', 'anomR (-)']
    df_metrics_OL = pd.DataFrame(index=INDEX, columns=COL, dtype=float)
    df_metrics_DA = pd.DataFrame(index=INDEX, columns=COL, dtype=float)

    # to plot observation updates
    io = LDAS_io('incr', exp=exp_DA, domain=domain, root=root)
    tc = io.grid.tilecoord

    # to load in situ site coordinates
    insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
    mastertable_filename = 'WTD_TROPICS_MASTER_TABLE_ALLDorN.csv'
    filenames = find_files(insitu_path, mastertable_filename)
    master_table = pd.read_csv(filenames, sep=';')

    for c, site in enumerate(wtd_obs.columns):
        df_tmp = pd.concat((wtd_obs[site], wtd_mod[0][site], wtd_mod[1][site]), axis=1)
        df_tmp.columns = ['In-situ', 'OL', 'DA']

        df_tmp_OL = copy.deepcopy(df_tmp[['OL', 'In-situ']])
        df_tmp_OL = df_tmp_OL[['OL', 'In-situ']]
        df_tmp_DA = copy.deepcopy(df_tmp[['DA','In-situ']])
        df_tmp_DA = df_tmp_DA[['DA', 'In-situ']]

        #metric calculation
        bias_site_OL = metrics.bias(df_tmp_OL)  # Bias = bias_site[0]
        ubRMSD_site_OL = metrics.ubRMSD(df_tmp_OL)  # ubRMSD = ubRMSD_site[0]
        pearson_R_site_OL = metrics.Pearson_R(df_tmp_OL)  # Pearson_R = pearson_R_site[0]
        RMSD_site_OL = copy.deepcopy(ubRMSD_site_OL)
        RMSD_site_OL[0] = (ubRMSD_site_OL[0] ** 2 + bias_site_OL[0] ** 2) ** 0.5

        bias_site_DA = metrics.bias(df_tmp_DA)  # Bias = bias_site[0]
        ubRMSD_site_DA = metrics.ubRMSD(df_tmp_DA)  # ubRMSD = ubRMSD_site[0]
        pearson_R_site_DA = metrics.Pearson_R(df_tmp_DA)  # Pearson_R = pearson_R_site[0]
        RMSD_site_DA = copy.deepcopy(ubRMSD_site_DA)
        RMSD_site_DA[0] = (ubRMSD_site_DA[0] ** 2 + bias_site_DA[0] ** 2) ** 0.5

        #anomR calculation with a minimum of three years of data
        threeYears = df_tmp['In-situ'].count()
        if threeYears >= 1095:
            anomaly_obs = calc_anomaly(df_tmp_OL['In-situ'])
            anomaly_mod = calc_anomaly(df_tmp_OL['OL'])
            anomalies = pd.concat([anomaly_mod, anomaly_obs], axis=1)
            anomR_site_OL = metrics.Pearson_R(anomalies)
            anomaly_obs = calc_anomaly(df_tmp_DA['In-situ'])
            anomaly_mod = calc_anomaly(df_tmp_DA['DA'])
            anomalies = pd.concat([anomaly_mod, anomaly_obs], axis=1)
            anomR_site_DA = metrics.Pearson_R(anomalies)
        else:
            anomR_site = np.array(['nan'])
            anomR_site_OL = pd.Series(anomR_site, index=['R'])
            anomR_site_DA = pd.Series(anomR_site, index=['R'])

        # Save metrics in df_metrics.
        df_metrics_OL.loc[site]['bias (m)'] = bias_site_OL[0]
        df_metrics_OL.loc[site]['ubRMSD (m)'] = ubRMSD_site_OL[0]
        df_metrics_OL.loc[site]['R (-)'] = pearson_R_site_OL[0]
        df_metrics_OL.loc[site]['RMSD (m)'] = RMSD_site_OL[0]
        df_metrics_OL.loc[site]['anomR (-)'] = anomR_site_OL[0]

        df_metrics_DA.loc[site]['bias (m)'] = bias_site_DA[0]
        df_metrics_DA.loc[site]['ubRMSD (m)'] = ubRMSD_site_DA[0]
        df_metrics_DA.loc[site]['R (-)'] = pearson_R_site_DA[0]
        df_metrics_DA.loc[site]['RMSD (m)'] = RMSD_site_DA[0]
        df_metrics_DA.loc[site]['anomR (-)'] = anomR_site_DA[0]

        # Create x-axis matching in situ data.
        x_start = df_tmp.index[0]  # Start a-axis with the first day with an observed wtd value.
        x_end = df_tmp.index[-1]  # End a-axis with the last day with an observed wtd value.
        Xlim = [x_start, x_end]

        site_lon = master_table.loc[master_table['site_id'] == site,['lon']].values[0]
        site_lat = master_table.loc[master_table['site_id'] == site,['lat']].values[0]

        # Calculate z-score for the time series.
        df_zscore = df_tmp.dropna(axis=0).apply(zscore)

        plt.figure(figsize=(16, 6.5))
        fontsize = 12

        ax1 = plt.subplot(311)
        df_tmp.plot(ax=ax1, fontsize=fontsize, style=['.', '-', '-'], color = ['#1f77b4', 'g', 'darkorange'], linewidth=2, xlim=Xlim)
        plt.ylabel('zbar [m]')
        Title = site
        plt.title(Title)

        ax1 = plt.subplot(312)
        df_tmp_OL = pd.concat((wtd_obs[site], wtd_mod[0][site]), axis=1)
        df_tmp_OL.columns=['In-situ', 'OL']
        df_tmp_OL.plot(ax=ax1, fontsize=fontsize, style=['.', '-'], color = ['#1f77b4','g'] ,linewidth=2,
                    xlim=[df_zscore.index[0], df_zscore.index[-1]])
        plt.ylabel('zbar [m]')

        Title = 'OL' + '\n' + ' bias = ' + str(bias_site_OL[0]) + ', ubRMSD = ' + str(
            ubRMSD_site_OL[0]) + ', R = ' + str(pearson_R_site_OL[0]) + ', RMSD = ' + str(
            RMSD_site_OL[0]) + ', anomR = ' + str(anomR_site_OL[0])
        plt.title(Title)

        ax2 = plt.subplot(313)
        df_tmp_DA = pd.concat((wtd_obs[site], wtd_mod[1][site]), axis=1)
        df_tmp_DA.columns=['In-situ', 'DA']
        df_tmp_DA.plot(ax=ax2, fontsize=fontsize, style=[ '.', '-'], color = ['#1f77b4','darkorange'],linewidth=2,
                       xlim=[df_zscore.index[0], df_zscore.index[-1]])
        plt.ylabel('zbar [m]')
        Title = 'DA' + '\n' + ' bias = ' + str(bias_site_DA[0]) + ', ubRMSD = ' + str(
            ubRMSD_site_DA[0]) + ', R = ' + str(pearson_R_site_DA[0]) + ', RMSD = ' + str(
            RMSD_site_DA[0]) + ', anomR = ' + str(anomR_site_DA[0])
        plt.title(Title)

        plt.legend()
        plt.tight_layout()
        fname = site
        fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '_DA.png')
        plt.savefig(fname_long, dpi=150)
        plt.close()


    plt.figure(figsize=(16, 5))
    ax1 = plt.subplot(121)
    df_metrics_OL.boxplot(ax=ax1, medianprops=dict(color = 'g'))
    fname = 'metrics_OL'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    mean = df_metrics_OL.mean()
    mean = mean.round(decimals=2)
    Title = 'mean bias (m) = ' + str(mean['bias (m)']) + ', mean ubRMSD (m) = ' + str(
        mean['ubRMSD (m)']) + ', mean R (-) = ' + str(mean['R (-)']) + '\n' + ' mean RMSD (m) = ' + str(
        mean['RMSD (m)']) + ', mean anomR (m) = ' + str(mean['anomR (-)'])
    plt.title(Title, fontsize=9)

    ax1 = plt.subplot(122)
    df_metrics_DA.boxplot(ax=ax1, medianprops=dict(color = 'darkorange'))
    fname = 'metrics_DA'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    mean = df_metrics_DA.mean()
    mean = mean.round(decimals=2)
    Title = 'mean bias (m) = ' + str(mean['bias (m)']) + ', mean ubRMSD (m) = ' + str(
        mean['ubRMSD (m)']) + ', mean R (-) = ' + str(mean['R (-)']) + '\n' + ' mean RMSD (m) = ' + str(
        mean['RMSD (m)']) + ', mean anomR (m) = ' + str(mean['anomR (-)'])
    plt.title(Title, fontsize=9)

    plt.savefig(fname_long, dpi=150)
    plt.close()

    plt.figure()
    df_metrics_diff=df_metrics_DA.subtract(df_metrics_OL)
    df_metrics_diff.boxplot()
    fname = 'metrics_diff'
    fname_long = os.path.join(outpath + '/comparison_insitu_data/' + fname + '.png')
    Title = 'DA-OL'
    plt.title(Title, fontsize=9)
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_anomaly_JulyAugust_zbar(exp, domain, root, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    fname = '01_GWanomaly_2012'
    io = LDAS_io('daily', exp, domain, root)
    # reference month
    # gs = (pd.to_datetime(io.timeseries['time'].values).month > 6) & (pd.to_datetime(io.timeseries['time'].values).month < 8)
    # get poros grid and bf1 and bf2 parameters
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # tmp = io.timeseries['zbar'][gs,:,:].values
    # data = io.timeseries['zbar'][cday,:,:].values - tmp.mean(axis=0)
    ## load anomaly from nc file
    ds = xr.open_dataset("/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/anomaly_JulyAugust_zbar.nc")
    data = ds['anomaly_zbar_JulyAugust'].values
    np.place(data, poros < 0.8, np.nan)
    cmap = 'jet_r'
    cmin = -0.5
    cmax = 0.5
    # open figure
    figsize = (10, 10)
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=200, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    # m = Basemap(projection='lcc', llcrnrlat=50, urcrnrlat=70, llcrnrlon=60,urcrnrlon=80, resolution='l')
    m = Basemap(projection='laea', width=2000000, height=1500000, lat_ts=50, lat_0=61, lon_0=75., resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    # ign = m.readshapefile('/data/leuven/317/vsc31786/gis/PEATFIRE_Ignitions_Siberia_2012_export', 'PEATFIRE_Ignitions_Siberia_2012_export')
    m.readshapefile('/data/leuven/317/vsc31786/gis/PEATFIRE_Siberia_2012_export', 'PEATFIRE_Siberia_2012_export',
                    linewidth=1.0, color='k')
    # m.plot()
    parallels = np.arange(-80.0, 81, 5.)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 351., 10.)
    m.drawmeridians(meridians, labels=[True, False, False, True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    # lat=53.
    # lon=38.5
    # x,y = m(lon, lat)
    # m.plot(x, y, 'ko', markersize=6,mfc='none')
    # cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "bottom", size="6%", pad="15%", shrink=0.5)
    cb.set_label("groundwater anomaly (m), 21 July 2012", fontsize=14)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    # plt.title("groundwater table anomaly (July 2012) and  peatland wildfires", fontsize=fontsize)
    # plt.tight_layout()
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def plot_RTMparams_old(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
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
        fname = param
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=param)


def plot_RTMparams_filled(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')

    # params[['bh','bv','omega','rgh_hmin','rgh_hmax']].fillna(-9999)
    params.fillna(-9999, inplace=True)
    cond = params.groupby(['bh', 'bv', 'omega', 'rgh_hmin', 'rgh_hmax']).vegcls.transform(len) > 70
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[cond == False] = np.nan
    for param in params:
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname = param + '_filled'
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=param)


def plot_RTMparams_delta(exp1, exp2, domain, root, outpath):
    # plot RTM parameters for LDAS output
    outpath = os.path.join(outpath, exp1, 'maps', 'rtmparams', 'compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    # setup grid for plots
    io = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    # set up grid
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    # read parameters
    # params_exp1 = LDAS_io(exp=exp1, domain=domain, root='/staging/leuven/stg_00024/OUTPUT/michelb').read_params('RTMparam')
    params_exp1 = LDAS_io(exp=exp1, domain=domain, root=root).read_params('RTMparam')
    params_exp1.fillna(-9999, inplace=True)
    params_exp2 = LDAS_io(exp=exp2, domain=domain, root=root).read_params('RTMparam')
    params_exp2.fillna(-9999, inplace=True)

    for param in params_exp1:
        # if ['bh','bv','omega','rgh_hmin','rgh_hmax'].count(param)==0:
        #    continue
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params_exp2[param].values - params_exp1[param].values
        data = np.ma.masked_invalid(img)
        fname = param
        # param = param+' (CLSM, recal_NORTH + SMOS_v620Tb) - '+param+" (CLSM, L4SM_v001_Lit4_CalD0)"
        param = param + ' ' + exp2 + ' - ' + param + ' ' + exp1
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                              plot_title=param)


def plot_RTMparams(exp, domain, root, outpath):
    # plot RTM parameters for LDAS output
    # + make plots of calib and filled grid cells as well as give fraction of calibrated cells.
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    # setup grid for plots
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    # set up grid
    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    # read parameters
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params.fillna(-9999, inplace=True)

    # if more than four (random choice, could be one also) 36 km grid cells (4 x 16 (M09) grid cells --> 64) (have exactly the same parameters, this set must be 'filled' one.
    cond_f = params.groupby(['bh', 'bv', 'omega', 'rgh_hmin', 'rgh_hmax']).vegcls.transform(len) > 16
    cond_filled = cond_f & (params.bh != -9999) & (params.poros > 0.05)
    cond_calib = (cond_f == False) & (params.bh != -9999) & (params.poros > 0.05)
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[(cond_filled == False) & (cond_calib == False)] = np.nan
    for param in params:
        if ['bh', 'bv', 'omega', 'rgh_hmin', 'rgh_hmax'].count(param) == 0:
            continue
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname = param
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=param)
    # histograms
    f = plt.figure(num=None, figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    i = 0
    for param in params:
        if ['bh', 'bv', 'omega', 'rgh_hmin', 'rgh_hmax'].count(param) == 0:
            continue
        i = i + 1
        plt.subplot(3, 2, i)
        plt.hist(params[param][cond_calib].values)
        plt.title(param)
    fname = os.path.join(outpath, 'histogram_RTM_params.png')
    plt.savefig(fname, dpi=300)

    params.bh[cond_filled] = 0
    params.bh[cond_calib] = 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cond_filled
    cond_filled_2D = img == 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cond_calib
    cond_calib_2D = img == 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params['bh'].values
    data = np.ma.masked_invalid(img)
    fname = '00_calib_filled'
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='filled=0, calib=1', cmap='winter')

    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ############## n_valid_innov
    n_valid_innov = ds['n_valid_innov'][:, :, :].sum(dim='species', skipna=True) / 2.0 / N_days
    cond_valid = n_valid_innov != 0
    n_valid_innov = n_valid_innov.where(cond_valid)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    data_valid = np.copy(data)
    data_valid[cond_valid == False] = np.nan
    data_valid = obs_M09_to_M36(data_valid)
    fname = '00_calib_filled_valid'
    figure_single_default(data=data_valid, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='filled=0, calib=1', cmap='winter')

    print('N(calib) / N(all): %.4f' % (np.size(np.where(cond_calib == True)[0]) / (
            np.size(np.where(cond_calib == True)[0]) + np.size(np.where(cond_filled == True)[0]))))
    print('N(calib, valid) / N(all, valid): %.4f' % (np.size(np.where(cond_calib_2D & cond_valid.values)[0]) / (
            np.size(np.where(cond_calib_2D & cond_valid.values)[0]) + np.size(
        np.where(cond_filled_2D & cond_valid.values)[0]))))

    ### same for peat
    outpath = os.path.join(outpath, exp, 'maps', 'rtmparams')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # other parameter definitions
    cmin = None
    cmax = None

    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')

    # params[['bh','bv','omega','rgh_hmin','rgh_hmax']].fillna(-9999)
    params.fillna(-9999, inplace=True)
    cond_f = params.groupby(['bh', 'bv', 'omega', 'rgh_hmin', 'rgh_hmax']).vegcls.transform(len) > 16
    cond_filled_peat = cond_f & (params.bh != -9999) & (params.poros > 0.65)
    cond_calib_peat = (cond_f == False) & (params.bh != -9999) & (params.poros > 0.65)
    params = LDAS_io(exp=exp, domain=domain, root=root).read_params('RTMparam')
    params[(cond_filled_peat == False) & (cond_calib_peat == False)] = np.nan
    for param in params:
        img = np.full(lons.shape, np.nan)
        img[tc.j_indg.values, tc.i_indg.values] = params[param].values
        data = np.ma.masked_invalid(img)
        fname = param + '_peat'
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=param)

    params.bh[cond_filled_peat] = 0
    params.bh[cond_calib_peat] = 1
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = params['bh'].values
    data = np.ma.masked_invalid(img)
    fname = '00_peat_calib_filled'
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='filled=0, calib=1 (peat distribution)', cmap='winter')

    print('N(calib, peat) / N(peat): %.4f' % (np.size(np.where(cond_calib == True)[0]) / (
            np.size(np.where(cond_calib == True)[0]) + np.size(np.where(cond_filled == True)[0]))))
    print('N(calib, peat, valid) / N(peat, valid): %.4f' % (np.size(np.where(cond_calib_2D & cond_valid.values)[0]) / (
            np.size(np.where(cond_calib_2D & cond_valid.values)[0]) + np.size(
        np.where(cond_filled_2D & cond_valid.values)[0]))))

    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ############## n_valid_innov
    n_valid_innov = ds['n_valid_innov'][:, :, :].sum(dim='species', skipna=True) / 2.0 / N_days
    cond_valid = n_valid_innov != 0
    n_valid_innov = n_valid_innov.where(n_valid_innov != 0)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    data_valid = np.copy(data)
    data_valid[cond_valid == False] = np.nan
    data_valid = obs_M09_to_M36(data_valid)
    fname = '00_peat_calib_filled_valid'
    figure_single_default(data=data_valid, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='filled=0, calib=1 (peat distribution)', cmap='winter')


def plot_lag_autocorr_innov(exp, domain, root, outpath):
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)

    data0 = obs_M09_to_M36(data[0, :, :])
    data1 = obs_M09_to_M36(data[1, :, :])
    data2 = obs_M09_to_M36(data[2, :, :])
    data3 = obs_M09_to_M36(data[3, :, :])

    tau, acor_lag1 = lag1_autocorr_from_numpy_array(data)





def plot_lag1_autocor(exp, domain, root, outpath):
    # set up grid
    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max() + 1)]
    lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max() + 1)]
    tc.i_indg -= tg.loc['domain', 'i_offg']  # col / lon
    tc.j_indg -= tg.loc['domain', 'j_offg']  # row / lat
    lons, lats = np.meshgrid(lons, lats)
    llcrnrlat = np.min(lats)
    urcrnrlat = np.max(lats)
    llcrnrlon = np.min(lons)
    urcrnrlon = np.max(lons)

    # calculate variable to plot
    # tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    tau, acor_lag1 = lag1_autocorr_from_numpy_array(tmp_incr)

    # open figure
    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 5)
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(tau)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-80.0, 81, 5.)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 351., 10.)
    m.drawmeridians(meridians, labels=[True, False, False, True])
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
    # plt.tight_layout()
    plt.savefig(fname, dpi=f.dpi)
    plt.close()


def plot_obs_std_single(exp, domain, root, outpath):
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['obs_ana']
    # tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    # incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std0 = tmp_incr[:, 0, :, :].std(dim='time', skipna=True).values
    incr_std1 = tmp_incr[:, 1, :, :].std(dim='time', skipna=True).values
    incr_std2 = tmp_incr[:, 2, :, :].std(dim='time', skipna=True).values
    incr_std3 = tmp_incr[:, 3, :, :].std(dim='time', skipna=True).values
    incr_std = np.nanmean(np.stack((incr_std0, incr_std1, incr_std2, incr_std3), axis=2), 2)
    data = obs_M09_to_M36(incr_std)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname = 'obs_std'
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='obs std (mm)')


def plot_obs_ana_std_quatro(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_ana']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname = 'obs_ana_std_quatro'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['obs_ana std (mm) H-Asc', 'obs_ana std (mm) H-Des', 'obs_ana std (mm) V-Asc',
                                           'obs_ana std (mm) V-Des']),mstats=mstats)


def plot_filter_diagnostics(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    try:
        n_valid_incr = ds['n_valid_incr'][:, :].values
    except:
        n_valid_incr = ds['n_valid_innov'][:, :].values

    np.place(n_valid_incr, n_valid_incr == 0, np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ########## obs mean #############
    data = ds['obs_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds['obs_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds['obs_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds['obs_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    data0[poros < 0.7] = np.nan
    data1[poros < 0.7] = np.nan
    data2[poros < 0.7] = np.nan
    data3[poros < 0.7] = np.nan
    cmin = 250.
    cmax = 290.
    fname = 'obs_mean_peatonly'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(['obs_mean_H_Asc', 'obs_mean_H_Des', 'obs_mean_V_Asc', 'obs_mean_V_Des']),mstats=mstats)

    ############## n_valid_innov
    data = ds['n_valid_innov'][:, :, :].sum(dim='species', skipna=True) / 2.0 / N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0.
    cmax = 0.4
    fname = 'n_valid_innov'
    figure_single_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data)))

    ############## n_valid_incr
    data = ds['n_valid_incr'] / N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0.
    cmax = 0.8
    fname = 'n_valid_incr'
    figure_single_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data)))

    ########## norm innov std double#############
    #data = ds['norm_innov_var'][0, :, :].values ** 0.5
    #data0 = obs_M09_to_M36(data)
    #data = ds['norm_innov_var'][1, :, :].values ** 0.5
    #data1 = obs_M09_to_M36(data)
    #data = ds['norm_innov_var'][2, :, :].values ** 0.5
    #data2 = obs_M09_to_M36(data)
    #data = ds['norm_innov_var'][3, :, :].values ** 0.5
    #data3 = obs_M09_to_M36(data)
    #data0[poros < 0.7] = np.nan
    #data1[poros < 0.7] = np.nan
    #data2[poros < 0.7] = np.nan
    #data3[poros < 0.7] = np.nan
    #data0 = 0.5 * (data0 + data1)
    #data1 = 0.5 * (data2 + data3)
    #data2 = 0.5 * (data0 + data1)
    #cmin = 0.
    #cmax = 2.
    #fname = 'norm_innov_std_double'
    #my_title0 = r'$H-pol,\/std(normalized\/O-F)\/(-)$'
    #my_title1 = r'$V-pol,\/std(normalized\/O-F)\/(-)$'
    #my_title2 = r'$std(normalized\/O-F)\/CLSM\/(-)$'
    #mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
    #               'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
    #               'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
    #               'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
    #               ])
    #latmin = -15.
    #latmax = 15.
    #lonmin = -90.
    #lonmax = 160.
    #lons1 = lons
    #lats1 = lats
    #[data0, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data0, lons1, lats1, latmin, latmax, lonmin, lonmax)
    #[data1, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data1, lons1, lats1, latmin, latmax, lonmin, lonmax)
    #[data2, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data2, lons1, lats1, latmin, latmax, lonmin, lonmax)
    #data_all = list([data0, data1, data2])
    #figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
    #                      urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
    #                      plot_title=([my_title0, my_title1, my_title2]), mstats=mstats, cmap='jet')
    # plot_title=(['norm_innov_std_H_Asc [(K)]', 'norm_innov_std_H_Des [(K)]', 'norm_innov_std_V_Asc [(K)]', 'norm_innov_std_V_Des [(K)]']))

    ############## n_valid_innov_quatro
    data = ds['n_valid_innov'][0, :, :] / N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][1, :, :] / N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][2, :, :] / N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][3, :, :] / N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname = 'n_valid_innov_quatro'
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname, mstats=mstats,
                          plot_title=(['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc', 'n_valid_innov_V_Des']))

    ########## innov mean #############
    data = ds['innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds['innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds['innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds['innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    data0[poros < 0.7] = np.nan
    data1[poros < 0.7] = np.nan
    data2[poros < 0.7] = np.nan
    data3[poros < 0.7] = np.nan
    cmin = -1.
    cmax = 1.
    fname = 'innov_mean_peatonly'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname, mstats=mstats,
                          plot_title=(['innov_mean_H_Asc', 'innov_mean_H_Des', 'innov_mean_V_Asc', 'innov_mean_V_Des']))

    ########## innov std #############
    data = ds['innov_var'][0, :, :].values ** 0.5
    data0 = obs_M09_to_M36(data)
    data = ds['innov_var'][1, :, :].values ** 0.5
    data1 = obs_M09_to_M36(data)
    data = ds['innov_var'][2, :, :].values ** 0.5
    data2 = obs_M09_to_M36(data)
    data = ds['innov_var'][3, :, :].values ** 0.5
    data3 = obs_M09_to_M36(data)
    #data0[poros < 0.7] = np.nan
    #data1[poros < 0.7] = np.nan
    #data2[poros < 0.7] = np.nan
    #data3[poros < 0.7] = np.nan
    cmin = None
    cmax = None
    fname = 'innov_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname, mstats = mstats,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## norm innov mean #############
    data = ds['norm_innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    #data0[poros < 0.7] = np.nan
    #data1[poros < 0.7] = np.nan
    #data2[poros < 0.7] = np.nan
    #data3[poros < 0.7] = np.nan
    cmin = None
    cmax = None
    fname = 'norm_innov_mean'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname, mstats = mstats,
                          plot_title=(
                              ['norm_innov_mean_H_Asc [(K)]', 'norm_innov_mean_H_Des [(K)]', 'norm_innov_mean_V_Asc [(K)]', 'norm_innov_mean_V_Des [(K)]']))

    ########## innov std #############
    data = ds['norm_innov_var'][0, :, :].values ** 0.5
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][1, :, :].values ** 0.5
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][2, :, :].values ** 0.5
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][3, :, :].values ** 0.5
    data3 = obs_M09_to_M36(data)
    data0[poros < 0.7] = np.nan
    data1[poros < 0.7] = np.nan
    data2[poros < 0.7] = np.nan
    data3[poros < 0.7] = np.nan
    cmin = 0.5
    cmax = 1.5
    fname = 'norm_innov_std_peatonly'
    my_title0 = 'H-pol, asc., std of norm. innov.: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0), np.nanstd(data0))
    my_title1 = 'H-pol, des., std of norm. innov.: m = %.2f, s = %.2f [mm]' % (np.nanmean(data1), np.nanstd(data1))
    my_title2 = 'V-pol, asc., std of norm. innov.: m = %.2f, s = %.2f [mm]' % (np.nanmean(data2), np.nanstd(data2))
    my_title3 = 'V-pol, des., std of norm. innov.: m = %.2f, s = %.2f [mm]' % (np.nanmean(data3), np.nanstd(data3))
    latmin = -15.
    latmax = 15.
    lonmin = -90.
    lonmax = 160.
    lons1 = lons
    lats1 = lats
    [data0, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data0, lons1, lats1, latmin, latmax,
                                                                                  lonmin, lonmax)
    [data1, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data1, lons1, lats1, latmin, latmax,
                                                                                  lonmin, lonmax)
    [data2, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data2, lons1, lats1, latmin, latmax,
                                                                                  lonmin, lonmax)
    [data3, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data3, lons1, lats1, latmin, latmax,
                                                                                  lonmin, lonmax)
    data_all = list([data0, data1, data2, data3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data0), np.nanstd(data0)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data1), np.nanstd(data1)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data2), np.nanstd(data2)),
                   'm = %.2f, s = %.2f' % (np.nanmean(data3), np.nanstd(data3))
                   ])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname, mstats =mstats,
                          plot_title=([my_title0, my_title1, my_title2, my_title3]))
    # plot_title=(['norm_innov_std_H_Asc [(K)]', 'norm_innov_std_H_Des [(K)]', 'norm_innov_std_V_Asc [(K)]', 'norm_innov_std_V_Des [(K)]']))

    ########## incr std #############
    data = ds['incr_srfexc_var'].values ** 0.5
    cmin = 0
    cmax = None
    fname = 'incr_srfexc_std'
    my_title = 'std(\u0394srfexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)

    ########## incr std #############
    data = ds['incr_rzexc_var'].values ** 0.5
    cmin = 0
    cmax = None
    fname = 'incr_rzexc_std'
    my_title = 'std(\u0394rzexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)

    ########## incr std #############
    data = ds['incr_catdef_var'].values ** 0.5
    cmin = 0
    cmax = 20
    fname = 'incr_catdef_std'
    my_title = 'std(\u0394catdef): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)

    ########## incr std #############
    #io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    #data = ds['incr_tws_var'].values ** 0.5
    #cmin = 0
    #cmax = 20
    #fname = 'incr_tws_std'
    #my_title = 'std(\u0394tws): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    #figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
    #                      urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
    #                      plot_title=my_title)


def plot_ensstd_insitu_error(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    n_valid_incr = ds['n_valid_incr'][:, :].values
    np.place(n_valid_incr, n_valid_incr == 0, np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ds_ensstd = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    if exp.startswith('CO') or exp.startswith('IN'):
        ncpath_OL = io.paths.root + '/' + exp + '/output_postprocessed/'
        ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')
        ds_ensstd_OL_CLSM = xr.open_dataset(
            io.paths.root + '/' + 'CONGO_M09_CLSM_v01_SMOSfw_OL' + '/output_postprocessed/ensstd_mean.nc')
        ds_ensstd_DA_CLSM = xr.open_dataset(
            io.paths.root + '/' + 'CONGO_M09_CLSM_v01_SMOSfw_DA' + '/output_postprocessed/ensstd_mean.nc')
    else:
        ncpath_OL = io.paths.root + '/' + exp[:-3] + '/output_postprocessed/'
        ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')
        ds_ensstd_OL_CLSM = xr.open_dataset(
            '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw/output_postprocessed/ensstd_mean.nc')
        ds_ensstd_DA_CLSM = xr.open_dataset(
            '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA/output_postprocessed/ensstd_mean.nc')
    ### ensstd vs ubRMSD plot
    ensstd_vs_ubRMSD_plot = 0
    if ensstd_vs_ubRMSD_plot == 1:
        master_table = pd.read_csv('/vsc-hard-mounts/leuven-data/317/vsc31786/FIG_tmp/00DA/20190228_M09/wtd_stats.txt',
                                   sep=',')
        if exp.startswith('CO'):
            insitu_path = '/data/leuven/317/vsc31786/peatland_data/tropics/WTD'
            mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'
            filenames = find_files(insitu_path, mastertable_filename)
            if isinstance(find_files(insitu_path, mastertable_filename), str):
                master_table = pd.read_csv(filenames, sep=';')
            else:
                for filename in filenames:
                    if filename.endswith('csv'):
                        master_table = pd.read_csv(filename, sep=';')
                        continue
                    else:
                        logging.warning(
                            "some files, maybe swp files, exist that start with master table searchstring !")
        ensstd_OL = np.zeros([59])
        ensstd_OL_CLSM = np.zeros([59])
        ensstd_DA = np.zeros([59])
        ensstd_DA_CLSM = np.zeros([59])

        for i, site_ID in enumerate(master_table.iloc[:, 0]):

            # if not site_ID.startswith('IN') and not site_ID.startswith('BR'):
            #    # Only use sites in Indonesia and Brunei.
            #    continue

            # if blacklist.iloc[:, 0].str.contains(site_ID).any() or master_table.iloc[i,4] == 0:
            #    # If site is on the blacklist (contains bad data), or if "comparison" =  0, then don include that site in the dataframe.
            #    continue

            # Get lat lon from master table for site.
            lon = master_table.iloc[i, 3]
            lat = master_table.iloc[i, 2]
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
            crow = np.array(
                [1, -1, 1, -1, 0, 0, 2, -2, 2, -2, 2, -2, 0, 0, 3, -3, 3, -3, 3, -3, 3, -3, 0, 0, 4, -4, 4, -4, 4, -4,
                 4, -4, 4, -4, 0, 0])
            ccol = np.array(
                [0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 2, -2, -2, 2, 1, -1, 2, -2, 3, -3, 0, 0, -3, 3, 1, -1, 2, -2, 3, -3,
                 4, -4, 0, 0, -4, 4])
            row = srow
            col = scol
            print("i: " + str(i))
            continue_yes = 0
            dp = 0
            for p in range(0, 70):
                if p == 35:
                    crow = crow * (-1)
                    dp = -35
                print("p: " + str(p))
                # Get poros for col row.
                siteporos = poros[row, col]
                print(siteporos)
                if siteporos <= 0.7:
                    row = srow + crow[p + dp]
                    col = scol + ccol[p + dp]
                else:
                    ensstd_OL[i] = ds_ensstd_OL['zbar'].values[row, col]
                    ensstd_DA[i] = ds_ensstd['zbar'].values[row, col]
                    ensstd_OL_CLSM[i] = ds_ensstd_OL_CLSM['zbar'].values[row, col]
                    ensstd_DA_CLSM[i] = ds_ensstd_DA_CLSM['zbar'].values[row, col]
                    break

        ubRMSD_OL = master_table.iloc[:, 14]
        ubRMSD_DA = master_table.iloc[:, 15]
        ubRMSD_OL_CLSM = master_table.iloc[:, 12]
        ubRMSD_DA_CLSM = master_table.iloc[:, 13]
        # open figure
        figsize = (5, 5)
        fontsize = 10
        f = plt.figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
        plt.plot(ubRMSD_OL, ensstd_OL, 'o', markersize=7, markeredgecolor='royalblue', markerfacecolor='whitesmoke',
                 label='OL')
        plt.plot(ubRMSD_DA, ensstd_DA, 'o', markersize=7, markeredgecolor='lawngreen', markerfacecolor='whitesmoke',
                 label='DA')
        plt.plot(np.array([0, 0.25]), np.array([0, 0.25]), 'k-')
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
        NSE_OL = 1 - np.sum((ensstd_OL - ubRMSD_OL) ** 2.0) / np.sum((ubRMSD_OL - np.mean(ubRMSD_OL)) ** 2.0)
        NSE_DA = 1 - np.sum((ensstd_DA - ubRMSD_DA) ** 2.0) / np.sum((ubRMSD_DA - np.mean(ubRMSD_DA)) ** 2.0)
        NSE_OL_formatted = '%.4f' % (NSE_OL)
        NSE_DA_formatted = '%.4f' % (NSE_DA)
        RMSD_OL = np.nanmean((ensstd_OL - ubRMSD_OL) ** 2.0) ** 0.5
        RMSD_DA = np.nanmean((ensstd_DA - ubRMSD_DA) ** 2.0) ** 0.5
        RMSD_OL_formatted = '%.4f' % (RMSD_OL)
        RMSD_DA_formatted = '%.4f' % (RMSD_DA)
        RMSD_OL_CLSM = np.nanmean((ensstd_OL_CLSM - ubRMSD_OL_CLSM) ** 2.0) ** 0.5
        RMSD_DA_CLSM = np.nanmean((ensstd_DA_CLSM - ubRMSD_DA_CLSM) ** 2.0) ** 0.5
        RMSD_OL_CLSM_formatted = '%.4f' % (RMSD_OL_CLSM)
        RMSD_DA_CLSM_formatted = '%.4f' % (RMSD_DA_CLSM)
        # NSE_OL = 1 - np.sum((ensstd_OL-ubRMSD_OL)**2.0)/np.sum((ubRMSD_OL-np.mean(ubRMSD_OL))**2.0)
        # NSE_DA = 1 - np.sum((ensstd_DA-ubRMSD_DA)**2.0)/np.sum((ubRMSD_DA-np.mean(ubRMSD_DA))**2.0)
        plt.plot(ubRMSD_OL, ensstd_OL, 'o', markersize=7, markeredgecolor='royalblue', markerfacecolor='whitesmoke',
                 label='OL (PEATCLSM); ' + 'RMSD = ' + RMSD_OL_formatted)
        plt.plot(ubRMSD_DA, ensstd_DA, 'o', markersize=7, markeredgecolor='lawngreen', markerfacecolor='whitesmoke',
                 label='DA (PEATCLSM); ' + 'RMSD = ' + RMSD_DA_formatted)
        plt.plot(ubRMSD_OL_CLSM, ensstd_OL_CLSM, 'D', markersize=7, markeredgecolor='royalblue',
                 markerfacecolor='whitesmoke', label='OL (CLSM); ' + 'RMSD = ' + RMSD_OL_CLSM_formatted)
        plt.plot(ubRMSD_DA_CLSM, ensstd_DA_CLSM, 'D', markersize=7, markeredgecolor='lawngreen',
                 markerfacecolor='whitesmoke', label='DA (CLSM); ' + 'RMSD = ' + RMSD_DA_CLSM_formatted)
        plt.plot(np.array([0, 1.1]), np.array([0, 1.1]), 'k-')
        plt.xlabel('ubRMSD (m)')
        plt.ylabel('<ensstd> (m)')
        plt.legend()
        fname_long = os.path.join(outpath, 'ubRMSD_ensstd_both.png')
        plt.tight_layout()
        plt.savefig(fname_long, dpi=f.dpi)
        plt.close()

    ########## ensstd OL DA mean #############
    data = ds_ensstd_OL['zbar'][:, :].values
    np.place(data, n_valid_incr < 100, np.nan)
    data_p1 = 1000 * data
    data = ds_ensstd['zbar'][:, :].values
    np.place(data, n_valid_incr < 1, np.nan)
    data_p2 = 1000 * data
    data = data_p2 - data_p1
    np.place(data, n_valid_incr < 1, np.nan)
    data_p3 = data
    np.place(data_p3, (n_valid_incr < 1) | (np.isnan(n_valid_incr) | (poros < 0.7)), np.nan)
    cmin = ([None, None, -40])
    cmax = ([None, None, 40])
    fname = '04b_delta_OL_DA_ensstd_zbar_mean_triple'
    # my_title='srfexc: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3), np.nanstd(data_p3))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=([r'$<ensstd(\overline{z}_{WT})>\/(OL\/PEATCLSM)\/[m]$',
                                       r'$<ensstd(zbar)>\/(DA\/PEATCLSM)\/[m]$', \
                                       r'$<\!ensstd(\overline{z}_{WT,DA})\!> - <\!ensstd(\overline{z}_{WT,OL})\!>\enspace(mm)$']),
                          mstats=mstats)

    # hist figure
    figsize = (7.0, 6)
    plt.rcParams.update({'font.size': 33})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "OL"}).set(xlim=(22, 90.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "DA"}).set(xlim=(22, 90.))
    plt.ylim(0., 0.10)
    plt.yticks((0.0, 0.1))
    plt.xlabel('$ensstd(\overline{z}_{WT})\enspace(mm)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_ensstd_zWT'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    # plt.savefig(fname_long)


def plot_filter_diagnostics_gs(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    outpath = os.path.join(outpath, exp, 'maps', 'diagnostics_gs')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    io_daily = LDAS_io('daily', exp=exp, domain=domain, root=root)
    # N_days = (io.images.time.values[-1]-io.images.time.values[0]).astype('timedelta64[D]').item().days
    gs_daily = (pd.to_datetime(io_daily.timeseries['time'].values).month > 7) & (
            pd.to_datetime(io_daily.timeseries['time'].values).month < 10)
    N_days = io_daily.timeseries['time'][gs_daily].size
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'filter_diagnostics_gs.nc')
    ds_ensstd = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    ncpath_OL = io.paths.root + '/' + exp[:-3] + '/output_postprocessed/'
    ds_ensstd_OL = xr.open_dataset(ncpath_OL + 'ensstd_mean.nc')

    n_valid_incr = ds['n_valid_incr'][:, :].values
    np.place(n_valid_incr, n_valid_incr == 0, np.nan)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    ############## n_valid_innov
    data = ds['n_valid_innov'][:, :, :].sum(dim='species', skipna=True) / 2.0 / N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname = 'n_valid_innov'
    figure_single_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data)))

    ############## n_valid_innov_quatro
    data = ds['n_valid_innov'][0, :, :] / N_days
    data = data.where(data != 0)
    data0 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][1, :, :] / N_days
    data = data.where(data != 0)
    data1 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][2, :, :] / N_days
    data = data.where(data != 0)
    data2 = obs_M09_to_M36(data)
    data = ds['n_valid_innov'][3, :, :] / N_days
    data = data.where(data != 0)
    data3 = obs_M09_to_M36(data)
    cmin = 0
    cmax = 0.35
    fname = 'n_valid_innov_quatro'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(
                              ['n_valid_innov_H_Asc', 'n_valid_innov_H_Des', 'n_valid_innov_V_Asc',
                               'n_valid_innov_V_Des']))

    ########## innov mean #############
    data = ds['innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds['innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds['innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds['innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    cmin = -3.
    cmax = 3.
    fname = 'innov_mean'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(['innov_mean_H_Asc', 'innov_mean_H_Des', 'innov_mean_V_Asc', 'innov_mean_V_Des']))

    ########## innov std #############
    data = ds['innov_var'][0, :, :].values ** 0.5
    data0 = obs_M09_to_M36(data)
    data = ds['innov_var'][1, :, :].values ** 0.5
    data1 = obs_M09_to_M36(data)
    data = ds['innov_var'][2, :, :].values ** 0.5
    data2 = obs_M09_to_M36(data)
    data = ds['innov_var'][3, :, :].values ** 0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'innov_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## innov mean #############
    data = ds['norm_innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'norm_innov_mean'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(
                              ['norm_innov_mean_H_Asc [(K)]', 'norm_innov_mean_H_Des [(K)]',
                               'norm_innov_mean_V_Asc [(K)]',
                               'norm_innov_mean_V_Des [(K)]']))

    ########## innov std #############
    data = ds['norm_innov_var'][0, :, :].values ** 0.5
    data0 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][1, :, :].values ** 0.5
    data1 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][2, :, :].values ** 0.5
    data2 = obs_M09_to_M36(data)
    data = ds['norm_innov_var'][3, :, :].values ** 0.5
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'norm_innov_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=(
                              ['norm_innov_std_H_Asc [(K)]', 'norm_innov_std_H_Des [(K)]', 'norm_innov_std_V_Asc [(K)]',
                               'norm_innov_std_V_Des [(K)]']))

    ############## n_valid_incr
    data = ds['n_valid_incr'] / N_days
    data = data.where(data != 0)
    data = obs_M09_to_M36(data)
    cmin = 0
    cmax = 1.00
    fname = 'n_valid_incr'
    figure_single_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='N per day: m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data)))

    ########## incr std #############
    data = ds['incr_srfexc_var'].values ** 0.5
    cmin = 0
    cmax = None
    fname = 'incr_srfexc_std'
    my_title = 'std(\u0394srfexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)

    ########## incr std #############
    data = ds['incr_rzexc_var'].values ** 0.5
    cmin = 0
    cmax = None
    fname = 'incr_rzexc_std'
    my_title = 'std(\u0394rzexc): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)

    ########## incr std #############
    data = ds['incr_catdef_var'].values ** 0.5
    cmin = 0
    cmax = 20
    fname = 'incr_catdef_std'
    my_title = 'std(\u0394catdef): m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)


def plot_scaling_delta(exp1, exp2, domain, root, outpath):
    exp = exp1
    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('ObsFcstAna', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # get scaling nc
    ncpath = io.paths.root + '/' + exp1 + '/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'scaling.nc')
    ncpath = io.paths.root + '/' + exp2 + '/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'scaling.nc')

    AscDes = ['A', 'D']
    data0 = ds1['m_mod_H_%2i' % angle][:, :, :, 0].mean(axis=2, skipna=True) - ds2['m_mod_H_%2i' % angle][:, :, :,
                                                                               0].mean(axis=2, skipna=True)
    data1 = ds1['m_mod_H_%2i' % angle][:, :, :, 1].mean(axis=2, skipna=True) - ds2['m_mod_H_%2i' % angle][:, :, :,
                                                                               1].mean(axis=2, skipna=True)
    data2 = ds1['m_mod_V_%2i' % angle][:, :, :, 0].mean(axis=2, skipna=True) - ds2['m_mod_V_%2i' % angle][:, :, :,
                                                                               0].mean(axis=2, skipna=True)
    data3 = ds1['m_mod_V_%2i' % angle][:, :, :, 1].mean(axis=2, skipna=True) - ds2['m_mod_V_%2i' % angle][:, :, :,
                                                                               1].mean(axis=2, skipna=True)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname = 'delta_mean_scaling'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))


def plot_filter_diagnostics_delta(exp1, exp2, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    latmin = 45.
    latmax = 70.
    lonmin = -170.
    lonmax = 95.
    outpath = os.path.join(outpath, exp1, 'maps', 'compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # set up grid
    io = LDAS_io('daily', exp=exp2, domain=domain, root=root)
    io2 = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    lons1 = lons
    lats1 = lats
    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp1 + '/output_postprocessed/'
    # ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA_old_scaling/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    # ds1e = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    ncpath = io.paths.root + '/' + exp2 + '/output_postprocessed/'
    # ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    # ds2e = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    # OL
    ncpath = io.paths.root + '/' + exp1[0:-2] + 'OL' + '/output_postprocessed/'
    # ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA_old_scaling/output_postprocessed/'
    ds1_OL = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ds1_OL_R = xr.open_dataset(ncpath + 'filter_diagnostics_R_nonrescaled_obs.nc')
    # ds1_OL_resc = xr.open_dataset(ncpath + 'filter_diagnostics_OL_with_rescaled_obs.nc')
    # ds1e = xr.open_dataset(ncpath + 'ensstd_mean.nc')
    ncpath = io.paths.root + '/' + exp2[0:-2] + 'OL' + '/output_postprocessed/'
    # ncpath = '/staging/leuven/stg_00024/OUTPUT/michelb/SMAP_EASEv2_M09_CLSM_SMOSfw_DA/output_postprocessed/'
    ds2_OL = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ds2_OL_R = xr.open_dataset(ncpath + 'filter_diagnostics_R_nonrescaled_obs.nc')
    # ds2_OL_resc = xr.open_dataset(ncpath + 'filter_diagnostics_OL_with_rescaled_obs.nc')
    # ds2e = xr.open_dataset(ncpath + 'ensstd_mean.nc')

    n_valid_innov = np.nansum(ds1['n_valid_innov'][:, :, :].values, axis=0) / 2.0
    np.place(n_valid_innov, n_valid_innov == 0, np.nan)
    n_valid_innov = obs_M09_to_M36(n_valid_innov)
    n_valid_incr = ds1['n_valid_incr'][:, :].values
    np.place(n_valid_incr, n_valid_incr == 0, np.nan)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    catparam_CLSM = io2.read_params('catparam')
    poros_CLSM = np.full(lons.shape, np.nan)
    poros_CLSM[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam_CLSM['poros'].values

    # np.place(poros, poros_CLSM<0.7, 0.4)
    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values, ds1['innov_var'][1, :, :].values,
                                 ds1['innov_var'][2, :, :].values, ds1['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values, ds2['innov_var'][1, :, :].values,
                                 ds2['innov_var'][2, :, :].values, ds2['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -15])
    if np.mean(lats) > 40:
        cmax = ([270, 270, 15])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_innov_var_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$', \
                                       r'$var(O-F)_{PEATCLSM} - var(O-F)_{CLSM}\enspace((K)^2)$']), mstats=mstats)
    # cross plot
    figsize = (7.0, 6)
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    plt.plot(data_p1[poros > 0.7], data_p2[poros > 0.7], '.')
    plt.xlabel('CLSM')
    plt.ylabel('PEATCLSM')
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    fname = 'cross_var_O-F'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 50.0)}).set(xlim=(0, 22.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 50.0)}).set(xlim=(0, 22.))
    plt.ylim(0., 0.28)
    plt.xlabel('$var(O-F)\enspace((K)^2)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    # plt.show()
    # plt.legend((h1,h2),('CLSM', 'PEATCLSM'))

    # plt.vlines(x = np.nanmean(data_p1[poros>0.7]), ymin=0.0, ymax=0.1,
    #           color = 'red', linestyle = 'dotted', linewidth = 2)

    # plt.vlines(x = np.nanmean(data_p2[poros>0.7]), ymin=0.0, ymax=0.1,
    #            color = 'blue', linestyle = 'dotted', linewidth = 2)
    fname = 'hist_var_O-F'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # plt.savefig(fname_long)

    ########## innov norescnoDA var triple #############
    data = np.nanmean(np.dstack((ds1_OL['innov_var'][0, :, :].values, ds1_OL['innov_var'][1, :, :].values,
                                 ds1_OL['innov_var'][2, :, :].values, ds1_OL['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2_OL['innov_var'][0, :, :].values, ds2_OL['innov_var'][1, :, :].values,
                                 ds2_OL['innov_var'][2, :, :].values, ds2_OL['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][0, :, :].values - ds1_OL['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][1, :, :].values - ds1_OL['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][2, :, :].values - ds1_OL['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][3, :, :].values - ds1_OL['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -15])
    if np.mean(lats) > 40:
        cmax = ([270, 270, 15])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_innov_norescnoDA_var_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$', \
                                       r'$var(O-F)_{PEATCLSM} - var(O-F)_{CLSM}\enspace((K)^2)$']), mstats=mstats)

    ########## innov norescnoDA RMSD triple #############
    data = np.nanmean(np.dstack((ds1_OL['innov_var'][0, :, :].values ** 0.5, ds1_OL['innov_var'][1, :, :].values ** 0.5,
                                 ds1_OL['innov_var'][2, :, :].values ** 0.5,
                                 ds1_OL['innov_var'][3, :, :].values ** 0.5)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2_OL['innov_var'][0, :, :].values ** 0.5, ds2_OL['innov_var'][1, :, :].values ** 0.5,
                                 ds2_OL['innov_var'][2, :, :].values ** 0.5,
                                 ds2_OL['innov_var'][3, :, :].values ** 0.5)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][0, :, :].values ** 0.5 - ds1_OL['innov_var'][0, :, :].values ** 0.5)
    data0 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][1, :, :].values ** 0.5 - ds1_OL['innov_var'][1, :, :].values ** 0.5)
    data1 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][2, :, :].values ** 0.5 - ds1_OL['innov_var'][2, :, :].values ** 0.5)
    data2 = obs_M09_to_M36(data)
    data = (ds2_OL['innov_var'][3, :, :].values ** 0.5 - ds1_OL['innov_var'][3, :, :].values ** 0.5)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -2.5])
    if np.mean(lats) > 40:
        cmax = ([15, 15, 2.5])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_innov_norescnoDA_RMSD_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$RMSD_{CLSM}\/(K)$', r'$RMSD_{PEATCLSM}\/(K)$', \
                                       r'$RMSD(Tb_{PEATCLSM},Tb_{SMOS}) - RMSD(Tb_{CLSM},Tb_{SMOS})\enspace(K)$']),
                          mstats=mstats)
    # figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$RMSD_{CLSM}\/(K)$', r'$RMSD_{PEATCLSM}\/(K)$', \
    #                                   r'$RMSD_{PEATCLSM} - RMSD_{CLSM}\enspace(K)$']),mstats=mstats)

    np.nanmean(data_p2[poros > 0.7]) / np.nanmean(data_p1[poros > 0.7])
    worse = np.where(data_p3[poros > 0.7] > 0.15)[0]
    better = np.where(data_p3[poros > 0.7] < 0.15)[0]
    np.size(better) / (np.size(better) + np.size(worse))

    plt.rcParams.update({'font.size': 10})
    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 12.0)}).set(xlim=(0, 10.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 12.0)}).set(xlim=(0, 10.))
    plt.ylim(0., 0.6)
    plt.xlabel('$RMSD (K)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_RMSD_O-F'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})

    ########## R pearson norescnoDA var triple #############
    data = np.nanmean(np.dstack((ds1_OL_R['R_mean'][0, :, :].values, ds1_OL_R['R_mean'][1, :, :].values,
                                 ds1_OL_R['R_mean'][2, :, :].values, ds1_OL_R['R_mean'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2_OL_R['R_mean'][0, :, :].values, ds2_OL_R['R_mean'][1, :, :].values,
                                 ds2_OL_R['R_mean'][2, :, :].values, ds2_OL_R['R_mean'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2_OL_R['R_mean'][0, :, :].values - ds1_OL_R['R_mean'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2_OL_R['R_mean'][1, :, :].values - ds1_OL_R['R_mean'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2_OL_R['R_mean'][2, :, :].values - ds1_OL_R['R_mean'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2_OL_R['R_mean'][3, :, :].values - ds1_OL_R['R_mean'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p1, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    np.place(data_p2, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -0.2])
    if np.mean(lats) > 40:
        cmax = ([1, 1, 0.2])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_R_norescnoDA_triple'
    # my_title='R_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$R_{CLSM}\/[-]$', r'$R_{PEATCLSM}\/[-]$',
                                       r'$R(Tb_{PEATCLSM},Tb_{SMOS}) - R(Tb_{CLSM},Tb_{SMOS})\enspace(-)$']),
                          mstats=mstats, cmap='jet_r')
    # figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$R_{CLSM}\/[-]$', r'$R_{PEATCLSM}\/[-]$', \
    #                                   r'$R_{PEATCLSM} - R_{CLSM} (-)$']),mstats=mstats,cmap='jet_r')
    worse = np.where(data_p3[poros > 0.7] > 0.0)[0]
    better = np.where(data_p3[poros > 0.7] < 0.0)[0]
    np.size(better) / (np.size(better) + np.size(worse))

    plt.rcParams.update({'font.size': 10})
    # hist figure
    figsize = (7.0, 5.9)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    ax = plt.subplot(111)
    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 1.0)}).set(xlim=(0, 1.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 1.0)}).set(xlim=(0, 1.))
    plt.ylim(0., 11.0)
    plt.xlabel('$R\enspace(-)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0.9)
    fname = 'hist_R_O-F'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})

    ########## innov std triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values ** 0.5, ds1['innov_var'][1, :, :].values ** 0.5,
                                 ds1['innov_var'][2, :, :].values ** 0.5, ds1['innov_var'][3, :, :].values ** 0.5)),
                      axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values ** 0.5, ds2['innov_var'][1, :, :].values ** 0.5,
                                 ds2['innov_var'][2, :, :].values ** 0.5, ds2['innov_var'][3, :, :].values ** 0.5)),
                      axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][0, :, :].values ** 0.5 - ds1['innov_var'][0, :, :].values ** 0.5
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values ** 0.5 - ds1['innov_var'][1, :, :].values ** 0.5
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values ** 0.5 - ds1['innov_var'][2, :, :].values ** 0.5
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values ** 0.5 - ds1['innov_var'][3, :, :].values ** 0.5
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    mean_peat = np.nanmean(data_p3[poros > 0.7])
    mean_peat_SI = np.nanmean(data_p3[(poros > 0.7) & (lons1 > 63) & (lons1 < 87) & (lats1 > 56) & (lats1 < 67)])
    mean_peat_HU = np.nanmean(data_p3[(poros > 0.7) & (lons1 > -95) & (lons1 < -80) & (lats1 > 50) & (lats1 < 60)])
    print('mean SI and HU')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)
    cmin = ([0, 0, -2.5])
    cmax = ([10, 10, 2.5])
    fname = '02b_delta_innov_std_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[(K)]$', r'$std(O-F)\/(PEATCLSM)\/[(K)]$',
                                       r'$std(O-F)\/(PEATCLSM) - std(O-F)\/(CLSM)\/[(K)]$']), mstats=mstats)

    worse = np.where(data_p3[poros > 0.7] > 0.015)[0]
    better = np.where(data_p3[poros > 0.7] < 0.015)[0]
    np.size(better) / (np.size(better) + np.size(worse))

    plt.rcParams.update({'font.size': 10})
    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 10.0)}).set(xlim=(0, 7.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 10.0)}).set(xlim=(0, 7.))
    plt.ylim(0., 1.5)
    plt.xlabel('$std(O-F)\enspace(K)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    # plt.show()
    # plt.legend((h1,h2),('CLSM', 'PEATCLSM'))

    # plt.vlines(x = np.nanmean(data_p1[poros>0.7]), ymin=0.0, ymax=0.1,
    #           color = 'red', linestyle = 'dotted', linewidth = 2)

    # plt.vlines(x = np.nanmean(data_p2[poros>0.7]), ymin=0.0, ymax=0.1,
    #            color = 'blue', linestyle = 'dotted', linewidth = 2)
    fname = 'hist_std_O-F'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # plt.savefig(fname_long)

    ########## catdef var pct #############
    data = (ds1['incr_catdef_var'][:, :].values + ds1['incr_rzexc_var'][:, :].values + ds1['incr_srfexc_var'][:,
                                                                                       :].values) ** 0.5
    np.place(data, n_valid_incr < 1, np.nan)
    data_p1 = data
    data = (ds2['incr_catdef_var'][:, :].values + ds2['incr_rzexc_var'][:, :].values + ds2['incr_srfexc_var'][:,
                                                                                       :].values) ** 0.5
    np.place(data, n_valid_incr < 1, np.nan)
    data_p2 = data
    data = (data_p2 - data_p1) \
           / (data_p1 + data_p2)
    data_p3 = 100 * data
    data = (data_p2 - data_p1)
    data_p3 = data
    np.place(data_p1, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    np.place(data_p2, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    np.place(data_p3, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    cmin = ([None, None, -10])
    cmax = ([None, None, 10])
    fname = '03b_delta_std_incr_catdef_triple'
    # my_title='total_water: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    cond = data_p3 < 15.0
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[cond]), np.nanstd(data_p3[cond]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<std(catdef)\/(CLSM)\/[mm]$', r'$<std(catdef)\/(PEATCLSM)\/[mm]$', \
                                       r'$std(\Delta wtot_{PEATCLSM}) - std(\Delta wtot_{CLSM})\enspace(mm)$']),
                          mstats=mstats)

    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 30.0)}).set(xlim=(0, 16.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 30.0)}).set(xlim=(0, 16.))
    plt.ylim(0., 0.33)
    plt.xlabel('$std(\Delta wtot)\enspace(mm)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_std_wtot'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # plt.savefig(fname_long)

    ########## catdef mean #############
    data1 = (ds1['incr_catdef_mean'][:, :].values + ds1['incr_rzexc_mean'][:, :].values + ds1['incr_srfexc_mean'][:,
                                                                                          :].values) ** 0.5
    np.place(data1, n_valid_incr < 1, np.nan)
    data_p1 = data1
    data2 = (ds2['incr_catdef_mean'][:, :].values + ds2['incr_rzexc_mean'][:, :].values + ds2['incr_srfexc_mean'][:,
                                                                                          :].values) ** 0.5
    np.place(data2, n_valid_incr < 1, np.nan)
    data_p2 = data2
    data_p3 = (data_p2 - data_p1)
    np.place(data_p1, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    np.place(data_p2, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    np.place(data_p3, (n_valid_incr < 1) | np.isnan(n_valid_incr) | (poros < 0.7), np.nan)
    cmin = ([-0.1, -0.1, -0.1])
    cmax = ([0.1, 0.1, 0.1])
    fname = '03b_delta_mean_incr_catdef_triple'
    # my_title='total_water: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3), np.nanstd(data_p3))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<mean(catdef)\/(CLSM)\/(mm)$', r'$<mean(catdef)\/(PEATCLSM)\/(mm)$', \
                                       r'$mean(\Delta wtot_{PEATCLSM}) - mean(\Delta wtot_{CLSM})\enspace(mm)$']),
                          mstats=mstats)

    # hist figure
    figsize = (7.0, 6)
    if np.mean(lats) > 40:
        plt.rcParams.update({'font.size': 33})
    else:
        plt.rcParams.update({'font.size': 20})
    f = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    h1 = seaborn.distplot(data_p1[(poros > 0.7) & (np.isnan(data_p1) == False)], hist=False, \
                          kde_kws={"color": (0.42745098, 0.71372549, 1.), "lw": 6, "label": "CLSM",
                                   'clip': (0.0, 2.0)}).set(xlim=(0, 4.))
    h2 = seaborn.distplot(data_p2[(poros > 0.7) & (np.isnan(data_p2) == False)], hist=False, \
                          kde_kws={"color": (0.14117647, 1., 0.14117647), "lw": 6, "label": "PEATCLSM",
                                   'clip': (0.0, 2.0)}).set(xlim=(0, 4.))
    # plt.ylim(0., 0.33)
    plt.xlabel('$mean(\Delta wtot)\enspace(mm)$')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = 'hist_mean_wtot'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.rcParams.update({'font.size': 10})
    # plt.savefig(fname_long)

    ########## n valid innov #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values, ds1['innov_var'][1, :, :].values,
                                 ds1['innov_var'][2, :, :].values, ds1['innov_var'][3, :, :].values)), axis=2)
    data = data ** 0.5
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values, ds2['innov_var'][1, :, :].values,
                                 ds2['innov_var'][2, :, :].values, ds2['innov_var'][3, :, :].values)), axis=2)
    data = data ** 0.5
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    data_p3 = data_p3 ** 0.5
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)

    data = ds2['n_valid_innov'][:, :, :].sum(dim='species', skipna=True) / 2.0
    data = data.where(data != 0)
    data_p3 = obs_M09_to_M36(data)

    cmin = ([0, 0, 0])
    cmax = ([15, 15, 1100])
    fname = '01_n_valid_innov'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3), np.nanstd(data_p3))])
    latmin = 45.
    latmax = 70.
    lonmin = -170.
    lonmax = 95.
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[(K)]$', r'$std(O-F)\/(PEATCLSM)\/[(K)]$', \
                                       r'Number of assimilated observations (-)']), mstats=mstats, cmap='jet')

    ########## innov std pct #############
    data = (ds2['innov_var'][0, :, :].values ** 0.5 - ds1['innov_var'][0, :, :].values ** 0.5) \
           / (ds1['innov_var'][0, :, :].values ** 0.5)
    np.place(data, ds1['n_valid_innov'][0, :, :] < 1, np.nan)
    data0 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][1, :, :].values ** 0.5 - ds1['innov_var'][1, :, :].values ** 0.5) \
           / (ds1['innov_var'][1, :, :].values ** 0.5)
    np.place(data, ds1['n_valid_innov'][1, :, :] < 1, np.nan)
    data1 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][2, :, :].values ** 0.5 - ds1['innov_var'][2, :, :].values ** 0.5) \
           / (ds1['innov_var'][2, :, :].values ** 0.5)
    np.place(data, ds1['n_valid_innov'][2, :, :] < 1, np.nan)
    data2 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][3, :, :].values ** 0.5 - ds1['innov_var'][3, :, :].values ** 0.5) \
           / (ds1['innov_var'][3, :, :].values ** 0.5)
    np.place(data, ds1['n_valid_innov'][3, :, :] < 1, np.nan)
    data3 = obs_M09_to_M36(data)

    data_ = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    [data, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_, lons1, lats1, latmin, latmax,
                                                                                 lonmin, lonmax)
    # mean over peatlands
    mean_peat = np.nanmean(data[poros > 0.7])
    mean_peat_SI = np.nanmean(data[(poros > 0.7) & (lons1 > 63) & (lons1 < 87) & (lats1 > 56) & (lats1 < 67)])
    mean_peat_HU = np.nanmean(data[(poros > 0.7) & (lons1 > -95) & (lons1 < -80) & (lats1 > 50) & (lats1 < 60)])
    print('mean peat, SI and HU std innov (pct)')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)

    worse = np.where(data[poros > 0.7] > 0.0)[0]
    better = np.where(data[poros > 0.7] < 0.0)[0]
    np.size(better) / (np.size(better) + np.size(worse))
    cmin = -30
    cmax = 30
    fname = 'delta_innov_std_pct'
    data = data * 100.
    my_title = 'delta std(O-F): m (peat) = %.1f, s (peat) = %.1f (pct)' % (
        np.nanmean(data[poros > 0.7]), np.nanstd(data[poros > 0.7]))
    figure_single_default(data=100 * data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=my_title)

    ########## innov var #############
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values) \
           / (ds1['innov_var'][0, :, :].values)
    np.place(data, ds1['n_valid_innov'][0, :, :] < 1, np.nan)
    data0 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values) \
           / (ds1['innov_var'][1, :, :].values)
    np.place(data, ds1['n_valid_innov'][1, :, :] < 1, np.nan)
    data1 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values) \
           / (ds1['innov_var'][2, :, :].values)
    np.place(data, ds1['n_valid_innov'][2, :, :] < 1, np.nan)
    data2 = obs_M09_to_M36(data)

    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values) \
           / (ds1['innov_var'][3, :, :].values)
    np.place(data, ds1['n_valid_innov'][3, :, :] < 1, np.nan)
    data3 = obs_M09_to_M36(data)

    data = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    # mean over peatlands
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    mean_peat = np.nanmean(data[poros > 0.7])
    mean_peat_SI = np.nanmean(data[(poros > 0.7) & (lons1 > 63) & (lons1 < 87) & (lats1 > 56) & (lats1 < 67)])
    mean_peat_HU = np.nanmean(data[(poros > 0.7) & (lons1 > -95) & (lons1 < -80) & (lats1 > 50) & (lats1 < 60)])
    print('mean SI and HU innov var (pct)')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)

    worse = np.where(data[poros > 0.7] > 0.0)[0]
    better = np.where(data[poros > 0.7] < 0.0)[0]
    np.size(better) / (np.size(better) + np.size(worse))
    cmin = -50
    cmax = 50
    fname = 'delta_innov_var_pct'
    data = data * 100.
    my_title = 'delta var(O-F): m (peat) = %.1f, s (peat) = %.1f (pct)' % (
        np.nanmean(data[poros > 0.7]), np.nanstd(data[poros > 0.7]))
    figure_single_default(data=100 * data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=my_title)

    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values, ds1['innov_var'][1, :, :].values,
                                 ds1['innov_var'][2, :, :].values, ds1['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values, ds2['innov_var'][1, :, :].values,
                                 ds2['innov_var'][2, :, :].values, ds2['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (np.isnan(n_valid_innov)), np.nan)
    mean_peat = np.nanmean(data_p3[poros > 0.7])
    mean_peat_SI = np.nanmean(data_p3[(poros > 0.7) & (lons1 > 63) & (lons1 < 87) & (lats1 > 56) & (lats1 < 67)])
    mean_peat_HU = np.nanmean(data_p3[(poros > 0.7) & (lons1 > -95) & (lons1 < -80) & (lats1 > 50) & (lats1 < 60)])
    print('mean SI and HU')
    print(mean_peat)
    print(mean_peat_SI)
    print(mean_peat_HU)
    cmin = ([0, 0, None])
    cmax = ([270, 270, None])
    fname = '02_delta_innov_var_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3), np.nanstd(data_p3))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$',
                                       r'$var(O-F)\/(PEATCLSM) - var(O-F)\/(CLSM)\/[(K)^2]$']), mstats=mstats)

    ########## norm innov std triple #############
    data = np.nanmean(np.dstack((ds1['norm_innov_var'][0, :, :].values, ds1['norm_innov_var'][1, :, :].values,
                                 ds1['norm_innov_var'][2, :, :].values, ds1['norm_innov_var'][3, :, :].values)), axis=2)
    np.place(data, (n_valid_innov < 1) | (np.isnan(n_valid_innov)), np.nan)
    data_p1 = obs_M09_to_M36(data) ** 0.5
    data = np.nanmean(np.dstack((ds2['norm_innov_var'][0, :, :].values, ds2['norm_innov_var'][1, :, :].values,
                                 ds2['norm_innov_var'][2, :, :].values, ds2['norm_innov_var'][3, :, :].values)), axis=2)
    np.place(data, (n_valid_innov < 1) | (np.isnan(n_valid_innov)), np.nan)
    data_p2 = obs_M09_to_M36(data) ** 0.5
    data_p3 = (data_p2 - 1.) - (data_p1 - 1.)
    np.place(data_p3, (n_valid_innov < 1) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, None])
    cmax = ([3.0, 3.0, None])
    fname = 'delta_norm_innov_std_avg_triple'
    # my_title='norm_innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.2f, s = %.2f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.2f, s = %.2f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$stdnorm(O-F)\/(CLSM)\/[(K)/(K)]$', r'$stdnorm(O-F)\/(PEATCLSM)\/[(K)/(K)]$',
                                       r'$(stdnorm(O-F)\/(PEATCLSM) - 1) - (stdnorm(O-F)\/(CLSM) - 1)\/[(K)/(K)]$']),
                          mstats=mstats)

    ########## ensstd sfmc mean triple #############
    # data = ds1e['sfmc'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p1 = data
    # data = ds2e['sfmc'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p2 = data
    # data = ds2e['sfmc'][:, :].values - ds1e['sfmc'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p3 = data
    # np.place(data_p3,(n_valid_incr<100) | (np.isnan(n_valid_incr)),np.nan)
    # cmin = ([None,None,None])
    # cmax = ([None,None,None])
    # fname='delta_sfmc_ensstd_mean_triple'
    ##my_title='srfexc: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    # data_all = list([data_p1,data_p2,data_p3])
    # mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    # figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$<ensstd(sfmc)>)\/(CLSM)\/[mm]$', r'$<ensstd(sfmc)>)\/(PEATCLSM)\/[mm]$', r'$<ensstd(sfmc)>)\/(PEATCLSM) - <ensstd(sfmc)>)\/(CLSM)\/[mm]$']), mstats=mstats)

    ########## ensstd total water mean triple #############
    # data = ds1e['total_water'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p1 = data
    # data = ds2e['total_water'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p2 = data
    # data = ds2e['total_water'][:, :].values - ds1e['total_water'][:, :].values
    # np.place(data,n_valid_incr<100,np.nan)
    # data_p3 = data
    # np.place(data_p3,(n_valid_incr<100) | (np.isnan(n_valid_incr)),np.nan)
    # cmin = ([None,None,None])
    # cmax = ([None,None,None])
    # fname='delta_total_water_ensstd_mean_triple'
    ##my_title='total_water: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    # data_all = list([data_p1,data_p2,data_p3])
    # mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1),np.nanstd(data_p1)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p2),np.nanstd(data_p2)),
    #               'm = %.3f, s = %.3f' % (np.nanmean(data_p3),np.nanstd(data_p3))])
    # figure_triple_default(data=data_all,lons=lons,lats=lats,cmin=cmin,cmax=cmax,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    #                      llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,outpath=outpath,exp=exp1,fname=fname,
    #                      plot_title=([r'$<ensstd(total\/water)>)\/(CLSM)\/[mm]$', r'$<ensstd(total\/water)>)\/(PEATCLSM)\/[mm]$', r'$<ensstd(total\/water)>)\/(PEATCLSM) - <ensstd(total\/water)>)\/(CLSM)\/[mm]$']),mstats=mstats)

    ########## norm innov mean triple #############
    data = np.nanmean(np.dstack((ds1['norm_innov_mean'][0, :, :].values, ds1['norm_innov_mean'][1, :, :].values,
                                 ds1['norm_innov_mean'][2, :, :].values, ds1['norm_innov_mean'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['norm_innov_mean'][0, :, :].values, ds2['norm_innov_mean'][1, :, :].values,
                                 ds2['norm_innov_mean'][2, :, :].values, ds2['norm_innov_mean'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][0, :, :].values - ds1['norm_innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][1, :, :].values - ds1['norm_innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][2, :, :].values - ds1['norm_innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][3, :, :].values - ds1['norm_innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 100) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([-1.5, -1.5, None])
    cmax = ([1.5, 1.5, None])
    fname = 'delta_norm_innov_mean_avg_triple'
    # my_title='norm_innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<norm(O-F)>\/(CLSM)\/[(K)]$', r'$<norm(O-F)>\/(PEATCLSM)\/[(K)]$',
                                       r'$<norm(O-F)>\/(PEATCLSM) - <norm(O-F)>\/(CLSM)\/[(K)]$']), mstats=mstats)

    ########## norm innov mean triple H=pol #############
    data = np.nanmean(np.dstack((ds1['norm_innov_mean'][0, :, :].values, ds1['norm_innov_mean'][1, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['norm_innov_mean'][0, :, :].values, ds2['norm_innov_mean'][1, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][0, :, :].values - ds1['norm_innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][1, :, :].values - ds1['norm_innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1)), axis=2)
    np.place(data_p3, (n_valid_innov < 100) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([-1.5, -1.5, None])
    cmax = ([1.5, 1.5, None])
    fname = 'delta_norm_innov_mean_Hpol_avg_triple'
    # my_title='norm_innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<norm(O-F)>\/(CLSM)\/[(K)]$', r'$<norm(O-F)>\/(PEATCLSM)\/[(K)]$',
                                       r'$<norm(O-F)>\/(PEATCLSM) - <norm(O-F)>\/(CLSM)\/[(K)]$']), mstats=mstats)

    ########## norm innov mean triple V=pol #############
    data = np.nanmean(np.dstack((ds1['norm_innov_mean'][2, :, :].values, ds1['norm_innov_mean'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['norm_innov_mean'][2, :, :].values, ds2['norm_innov_mean'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][2, :, :].values - ds1['norm_innov_mean'][2, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['norm_innov_mean'][3, :, :].values - ds1['norm_innov_mean'][3, :, :].values
    data1 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1)), axis=2)
    np.place(data_p3, (n_valid_innov < 100) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([-1.5, -1.5, None])
    cmax = ([1.5, 1.5, None])
    fname = 'delta_norm_innov_mean_Vpol_avg_triple'
    # my_title='norm_innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<norm(O-F)>\/(CLSM)\/[(K)]$', r'$<norm(O-F)>\/(PEATCLSM)\/[(K)]$',
                                       r'$<norm(O-F)>\/(PEATCLSM) - <norm(O-F)>\/(CLSM)\/[(K)]$']), mstats=mstats)

    ########## innov var triple #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values, ds1['innov_var'][1, :, :].values,
                                 ds1['innov_var'][2, :, :].values, ds1['innov_var'][3, :, :].values)), axis=2)
    data = data ** 0.5
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values, ds2['innov_var'][1, :, :].values,
                                 ds2['innov_var'][2, :, :].values, ds2['innov_var'][3, :, :].values)), axis=2)
    data = data ** 0.5
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    data_p3 = data_p3 ** 0.5
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -4])
    cmax = ([15, 15, 4])
    fname = '02_delta_innov_std_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$std(O-F)\/(CLSM)\/[(K)]$', r'$std(O-F)\/(PEATCLSM)\/[(K)]$', \
                                       r'$std(O-F)_{PEATCLSM} - std(O-F)_{CLSM}\enspace(K)$']), mstats=mstats)

    ########## innov var triple pct #############
    data = np.nanmean(np.dstack((ds1['innov_var'][0, :, :].values, ds1['innov_var'][1, :, :].values,
                                 ds1['innov_var'][2, :, :].values, ds1['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_var'][0, :, :].values, ds2['innov_var'][1, :, :].values,
                                 ds2['innov_var'][2, :, :].values, ds2['innov_var'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][0, :, :].values - ds1['innov_var'][0, :, :].values) \
           / (ds1['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][1, :, :].values - ds1['innov_var'][1, :, :].values) \
           / (ds1['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][2, :, :].values - ds1['innov_var'][2, :, :].values) \
           / (ds1['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2['innov_var'][3, :, :].values - ds1['innov_var'][3, :, :].values) \
           / (ds1['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = 100. * np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -60])
    cmax = ([270, 270, 60])
    fname = '02_delta_pct_innov_var_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$', \
                                       r'$(var(O-F)_{PEATCLSM} - var(O-F)_{CLSM})/var(O-F)_{CLSM}\enspace\times 100\enspace(\%)$']),
                          mstats=mstats)

    ########## innov mean triple #############
    data = np.nanmean(np.dstack((ds1['innov_mean'][0, :, :].values, ds1['innov_mean'][1, :, :].values,
                                 ds1['innov_mean'][2, :, :].values, ds1['innov_mean'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2['innov_mean'][0, :, :].values, ds2['innov_mean'][1, :, :].values,
                                 ds2['innov_mean'][2, :, :].values, ds2['innov_mean'][3, :, :].values)), axis=2)
    np.place(data, n_valid_innov < 100, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][0, :, :].values - ds1['innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][1, :, :].values - ds1['innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][2, :, :].values - ds1['innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][3, :, :].values - ds1['innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 100) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([-5, -5, None])
    cmax = ([5, 5, None])
    fname = 'delta_innov_mean_avg_triple'
    mstats = list(['m = %.3f, s = %.3f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.3f, s = %.3f' % (np.nanmean(data_p3), np.nanstd(data_p3))])
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data_p1, data_p2, data_p3])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$<(O-F)>\/(CLSM)\/[(K)]$', r'$<(O-F)>\/(PEATCLSM)\/[(K)]$',
                                       r'$<(O-F)>\/(PEATCLSM) - <(O-F)>\/(CLSM)\/[(K)]$']), mstats=mstats)

    ########## n_innov #############
    data = ds2['n_valid_innov'][0, :, :].values - ds1['n_valid_innov'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][1, :, :].values - ds1['n_valid_innov'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][2, :, :].values - ds1['n_valid_innov'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['n_valid_innov'][3, :, :].values - ds1['n_valid_innov'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_n_valid_innov'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(['H_Asc', 'H_Des', 'V_Asc', 'V_Des']))

    ########## innov mean #############
    data = ds2['innov_mean'][0, :, :].values - ds1['innov_mean'][0, :, :].values
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][1, :, :].values - ds1['innov_mean'][1, :, :].values
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][2, :, :].values - ds1['innov_mean'][2, :, :].values
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_mean'][3, :, :].values - ds1['innov_mean'][3, :, :].values
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_innov_mean'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(['H_Asc', 'H_Des', 'V_Asc', 'V_Des']))

    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values ** 0.5 - ds1['innov_var'][0, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][0, :, :] < 100, np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values ** 0.5 - ds1['innov_var'][1, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][1, :, :] < 100, np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values ** 0.5 - ds1['innov_var'][2, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][2, :, :] < 100, np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values ** 0.5 - ds1['innov_var'][3, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][3, :, :] < 100, np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_innov_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## catdef std #############
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 0, np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 200, np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 500, np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 1000, np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_catdef_incr_std_quatro'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(['n_valid_incr>0', 'n_valid_incr>200', 'n_valid_incr>500', 'n_valid_incr>1000']))

    ########## catdef std #############
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    cmin = None
    cmax = None
    fname = 'delta_catdef_incr_std'
    my_title = 'mean delta std: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0), np.nanstd(data0))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=my_title)

    ########## catdef std pct #############
    data = 100 * ((ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5) \
                  / (ds1['incr_catdef_var'][:, :].values ** 0.5))
    # / (0.5*(ds2['incr_catdef_var'][:, :].values**0.5 + ds1['incr_catdef_var'][:, :].values**0.5))
    cmin = -200
    cmax = 200
    fname = 'delta_catdef_incr_std_pct'
    my_title = 'mean delta std pct: m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=my_title)

    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values ** 0.5 - ds1['innov_var'][0, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][0, :, :] < 100, np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values ** 0.5 - ds1['innov_var'][1, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][1, :, :] < 100, np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values ** 0.5 - ds1['innov_var'][2, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][2, :, :] < 100, np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values ** 0.5 - ds1['innov_var'][3, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][3, :, :] < 100, np.nan)
    data3 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    cmin = None
    cmax = None
    fname = 'delta_innov_std'
    my_title = 'delta innov std: m = %.2f, s = %.2f [mm]' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=my_title)

    ########## innov rescnoDA var triple #############
    data = np.nanmean(np.dstack((ds1_OL_resc['innov_var'][0, :, :].values, ds1_OL_resc['innov_var'][1, :, :].values,
                                 ds1_OL_resc['innov_var'][2, :, :].values, ds1_OL_resc['innov_var'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2_OL_resc['innov_var'][0, :, :].values, ds2_OL_resc['innov_var'][1, :, :].values,
                                 ds2_OL_resc['innov_var'][2, :, :].values, ds2_OL_resc['innov_var'][3, :, :].values)),
                      axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][0, :, :].values - ds1_OL_resc['innov_var'][0, :, :].values)
    data0 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][1, :, :].values - ds1_OL_resc['innov_var'][1, :, :].values)
    data1 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][2, :, :].values - ds1_OL_resc['innov_var'][2, :, :].values)
    data2 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][3, :, :].values - ds1_OL_resc['innov_var'][3, :, :].values)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -15])
    if np.mean(lats) > 40:
        cmax = ([270, 270, 15])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_innov_rescnoDA_var_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1), np.nanstd(data_p1)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2), np.nanstd(data_p2)),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$', \
                                       r'$var(O-F)_{PEATCLSM} - var(O-F)_{CLSM}\enspace((K)^2)$']), mstats=mstats)

    ########## innov rescnoDA RMSD triple #############
    data = np.nanmean(np.dstack((ds1_OL_resc['innov_var'][0, :, :].values ** 0.5,
                                 ds1_OL_resc['innov_var'][1, :, :].values ** 0.5,
                                 ds1_OL_resc['innov_var'][2, :, :].values ** 0.5,
                                 ds1_OL_resc['innov_var'][3, :, :].values ** 0.5)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p1 = obs_M09_to_M36(data)
    data = np.nanmean(np.dstack((ds2_OL_resc['innov_var'][0, :, :].values ** 0.5,
                                 ds2_OL_resc['innov_var'][1, :, :].values ** 0.5,
                                 ds2_OL_resc['innov_var'][2, :, :].values ** 0.5,
                                 ds2_OL_resc['innov_var'][3, :, :].values ** 0.5)), axis=2)
    np.place(data, n_valid_innov < 1, np.nan)
    data_p2 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][0, :, :].values ** 0.5 - ds1_OL_resc['innov_var'][0, :, :].values ** 0.5)
    data0 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][1, :, :].values ** 0.5 - ds1_OL_resc['innov_var'][1, :, :].values ** 0.5)
    data1 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][2, :, :].values ** 0.5 - ds1_OL_resc['innov_var'][2, :, :].values ** 0.5)
    data2 = obs_M09_to_M36(data)
    data = (ds2_OL_resc['innov_var'][3, :, :].values ** 0.5 - ds1_OL_resc['innov_var'][3, :, :].values ** 0.5)
    data3 = obs_M09_to_M36(data)
    data_p3 = np.nanmean(np.dstack((data0, data1, data2, data3)), axis=2)
    np.place(data_p3, (n_valid_innov < 1) | (poros < 0.7) | (np.isnan(n_valid_innov)), np.nan)
    cmin = ([0, 0, -15])
    if np.mean(lats) > 40:
        cmax = ([270, 270, 15])
    else:
        cmax = ([30, 30, 15])
    fname = '02b_delta_innov_rescnoDA_RMSD_avg_triple'
    # my_title='innov_var: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    mstats = list(['m = %.1f, s = %.1f' % (np.nanmean(data_p1[poros > 0.7]), np.nanstd(data_p1[poros > 0.7])),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p2[poros > 0.7]), np.nanstd(data_p2[poros > 0.7])),
                   'm = %.1f, s = %.1f' % (np.nanmean(data_p3[poros > 0.7]), np.nanstd(data_p3[poros > 0.7]))])
    [data_p1_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p1, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p2_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p2, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    [data_p3_, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = figure_zoom(data_p3, lons1, lats1, latmin,
                                                                                     latmax, lonmin, lonmax)
    data_all = list([data_p1_, data_p2_, data_p3_])
    figure_triple_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=([r'$var(O-F)\/(CLSM)\/[(K)^2]$', r'$var(O-F)\/(PEATCLSM)\/[(K)^2]$', \
                                       r'$var(O-F)_{PEATCLSM} - var(O-F)_{CLSM}\enspace((K)^2)$']), mstats=mstats)


def plot_daily_delta(exp1, exp2, domain, root, outpath):
    outpath = os.path.join(outpath, exp1, 'maps', 'compare')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype('timedelta64[D]').item().days
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # get filter diagnostics
    ncpath = io.paths.root + '/' + exp1 + '/output_postprocessed/'
    ds1 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')
    ncpath = io.paths.root + '/' + exp2 + '/output_postprocessed/'
    ds2 = xr.open_dataset(ncpath + 'filter_diagnostics.nc')

    ########## innov std #############
    data = ds2['innov_var'][0, :, :].values ** 0.5 - ds1['innov_var'][0, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][0, :, :] < 100, np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['innov_var'][1, :, :].values ** 0.5 - ds1['innov_var'][1, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][1, :, :] < 100, np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['innov_var'][2, :, :].values ** 0.5 - ds1['innov_var'][2, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][2, :, :] < 100, np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['innov_var'][3, :, :].values ** 0.5 - ds1['innov_var'][3, :, :].values ** 0.5
    np.place(data, ds1['n_valid_innov'][3, :, :] < 100, np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_innov_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(['innov_std_H_Asc', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))

    ########## innov std #############
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 0, np.nan)
    data0 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 200, np.nan)
    data1 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 500, np.nan)
    data2 = obs_M09_to_M36(data)
    data = ds2['incr_catdef_var'][:, :].values ** 0.5 - ds1['incr_catdef_var'][:, :].values ** 0.5
    np.place(data, ds1['n_valid_incr'][:, :] < 1000, np.nan)
    data3 = obs_M09_to_M36(data)
    cmin = None
    cmax = None
    fname = 'delta_catdef_incr_std'
    # my_title='innov_mean: m = %.2f, s = %.2f [mm]' % (np.nanmean(data0),np.nanstd(data0))
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data=data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=(
                              ['delta_catdef_incr_std', 'innov_std_H_Des', 'innov_std_V_Asc', 'innov_std_V_Des']))


def plot_increments(exp, domain, root, outpath, dti):
    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1 == timestamp)[0]
        data = io.images['catdef'][timestep[0], :, :] + io.images['srfexc'][timestep[0], :, :] + io.images['rzexc'][
                                                                                                 timestep[0], :, :]
        if np.nanmean(data) == 0:
            continue
        data = data.where(data != 0)
        # other parameter definitions
        cmin = None
        cmax = None
        fname = 'incr_' + str(timestamp)[0:13]
        my_title = 'increments (total water) (mm)'
        figure_single_default(data=-data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=my_title)


def plot_increments_tp1_std(exp, domain, root, outpath):
    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['tp1']
    # tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    # incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    data = tmp_incr.std(dim='time', skipna=True).values

    # other parameter definitions
    cmin = 0
    cmax = None

    fname = 'incr_tp1_std'
    my_title = 'std(increments tp1) (K): m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)


def plot_increments_number(exp, domain, root, outpath):
    io = LDAS_io('incr', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    tmp_incr = io.timeseries['catdef']  # + io.timeseries['srfexc'] + io.timeseries['rzexc']
    # tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    # incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    N_3hourly = (io.driver['DRIVER_INPUTS']['end_time']['year'] - io.driver['DRIVER_INPUTS']['start_time'][
        'year']) * 365 * 8
    data = tmp_incr.count(dim='time').values / N_3hourly

    # other parameter definitions
    cmin = 0
    cmax = 0.35

    fname = 'incr_number'
    my_title = 'N per day (incr.): m = %.2f, s = %.2f' % (np.nanmean(data), np.nanstd(data))
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)


def plot_Fcst_std_delta(exp1, exp2, domain, root, outpath):
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data_1_0 = obs_M09_to_M36(data_std0)
    data_1_1 = obs_M09_to_M36(data_std1)
    data_1_2 = obs_M09_to_M36(data_std2)
    data_1_3 = obs_M09_to_M36(data_std3)

    io = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
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

    fname = 'delta_Fcst'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=list(['delta std (K) H-Asc', 'delta std (K) H-Des', 'delta std (K) V-Asc',
                                           'delta std (K) V-Des']))


def plot_increments_std_delta(exp1, exp2, domain, root, outpath):
    io = LDAS_io('incr', exp=exp1, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    # tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    # incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std1 = tmp_incr.std(dim='time', skipna=True).values

    io = LDAS_io('incr', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    tmp_incr = io.timeseries['catdef'] + io.timeseries['srfexc'] + io.timeseries['rzexc']
    # tmp_incr = io.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    # incr_std = (io.timeseries['srfexc'] + io.timeseries['rzexc'] - io.timeseries['catdef']).std(dim='time',skipna=True).values
    incr_std2 = tmp_incr.std(dim='time', skipna=True).values

    data = incr_std2 - incr_std1
    # other parameter definitions
    cmin = -5
    cmax = 5

    fname = 'delta_incr'
    plot_title = 'incr_CLSM - incr_PEATCLSM'
    figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp1, fname=fname,
                          plot_title=plot_title)


def plot_obs_obs_std_quatro(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs']
    # data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname = 'obs_obs_std_quatro'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_default(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['obs_obs std (mm) H-Asc', 'obs_obs std (mm) H-Des', 'obs_obs std (mm) V-Asc',
                                           'obs_obs std (mm) V-Des']))

    N_days = (io.images.time.values[-1] - io.images.time.values[0]).astype(
        'timedelta64[D]').item().days  # for 3hourly ObsFcstAna
    data_num0 = data[:, 0, :, :].count(dim='time').values / N_days
    data_num1 = data[:, 1, :, :].count(dim='time').values / N_days
    data_num2 = data[:, 2, :, :].count(dim='time').values / N_days
    data_num3 = data[:, 3, :, :].count(dim='time').values / N_days
    data_num2 = data_num0 + data_num1
    data_num0[data_num0 == 0] = np.nan
    data_num1[data_num1 == 0] = np.nan
    data_num2[data_num2 == 0] = np.nan
    data_num3[data_num3 == 0] = np.nan
    data_num0 = obs_M09_to_M36(data_num0)
    data_num1 = obs_M09_to_M36(data_num1)
    data_num2 = obs_M09_to_M36(data_num2)
    data_num3 = obs_M09_to_M36(data_num3)
    data_num3[:, :] = 0.0

    # other parameter definitions
    cmin = 0
    cmax = 0.35
    fname = 'obs_number_quatro'
    my_title_0 = 'N_obs per day (Asc.): m = %.2f, s = %.2f' % (np.nanmean(data_num0), np.nanstd(data_num0))
    my_title_1 = 'N_obs per day (Des.): m = %.2f, s = %.2f' % (np.nanmean(data_num1), np.nanstd(data_num1))
    my_title_2 = 'N_obs per day (Asc.+Des.): m = %.2f, s = %.2f' % (np.nanmean(data_num2), np.nanstd(data_num2))
    my_title_3 = '...'
    my_title = list([my_title_0, my_title_1, my_title_2, my_title_3])
    data_all = list([data_num0, data_num1, data_num2, data_num3])
    figure_quatro_default(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=my_title)


def plot_obs_fcst_std_quatro(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 20
    fname = 'obs_fcst_std_quatro'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(
                              ['obs_fcst std (mm) H-Asc', 'obs_fcst std (mm) H-Des', 'obs_fcst std (mm) V-Asc',
                               'obs_fcst std (mm) V-Des']))


def plot_innov_std_quatro(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0
    cmax = 15
    fname = 'innov_std_quatro'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['innov std (Tb) H-Asc', 'innov std (Tb) H-Des', 'innov std (Tb) V-Asc',
                                           'innov std (Tb) V-Des']))


def plot_innov_quatro(exp, domain, root, outpath, dti):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1 == timestamp)[0]
        data = io.images['obs_obs'][timestep[0], :, :, :] - io.images['obs_fcst'][timestep[0], :, :, :]
        if np.nanmean(data) == 0:
            continue
        # calculate variable to plot
        data0 = obs_M09_to_M36(data[0, :, :])
        data1 = obs_M09_to_M36(data[1, :, :])
        data2 = obs_M09_to_M36(data[2, :, :])
        data3 = obs_M09_to_M36(data[3, :, :])

        # other parameter definitions
        cmin = None
        cmax = None
        fname = 'innov_quatro_' + str(timestamp)[0:13]
        data = list([data0, data1, data2, data3])
        figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=list(
                                  ['O-F (Tb in (K)) H-Asc', 'O-F (Tb in (K)) H-Des', 'O-F (Tb in (K)) V-Asc',
                                   'O-F (Tb in (K)) V-Des']))


def plot_kalman_gain(exp, domain, root, outpath, dti):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    s1 = io.images.time.values
    s2 = dti.values
    timestamps = pd.Series(list(set(s1).intersection(set(s2)))).values
    for timestamp in timestamps:
        timestep = np.where(s1 == timestamp)[0]
        # OF
        data = io.images['obs_obs'][timestep[0], :, :, :] - io.images['obs_fcst'][timestep[0], :, :, :]
        if np.nanmean(data) == 0:
            continue
        # calculate variable to plot
        # tmp = data.sum(dim='species',skipna=True)
        tmp = data[0, :, :]
        tmp = tmp.where(tmp != 0)
        OF = obs_M09_to_M36(tmp)
        # incr
        data = io.images['obs_ana'][timestep[0], :, :, :] - io.images['obs_fcst'][timestep[0], :, :, :]
        if np.nanmean(data) == 0:
            continue
        # calculate variable to plot
        # tmp = data.sum(dim='species',skipna=True)
        tmp = data[0, :, :]
        tmp = tmp.where(tmp != 0)
        incr = obs_M09_to_M36(tmp)
        # other parameter definitions
        cmin = 0
        cmax = 1
        fname = 'kalman_gain_' + str(timestamp)[0:13]
        data = incr / OF
        my_title = 'incr / (O - F) '
        figure_single_default(data=data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                              urcrnrlat=urcrnrlat,
                              llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                              plot_title=my_title)


def plot_innov_norm_std_quatro(exp, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des

    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data / np.sqrt(io.timeseries['obs_obsvar'] + io.timeseries['obs_fcstvar'])
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)

    # other parameter definitions
    cmin = 0.5
    cmax = 1.5
    fname = 'innov_norm_std_quatro'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(
                              ['innov norm std (-) H-Asc', 'innov norm std (-) H-Des', 'innov norm std (-) V-Asc',
                               'innov norm std (-) V-Des']))


def plot_innov_delta_std_single(exp1, exp2, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    exp = exp1
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data_exp1_0 = obs_M09_to_M36(data_std0)
    data_exp1_1 = obs_M09_to_M36(data_std1)
    data_exp1_2 = obs_M09_to_M36(data_std2)
    data_exp1_3 = obs_M09_to_M36(data_std3)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data0 = obs_M09_to_M36(data_std0)
    data1 = obs_M09_to_M36(data_std1)
    data2 = obs_M09_to_M36(data_std2)
    data3 = obs_M09_to_M36(data_std3)
    data0 = data0 - data_exp1_0
    data1 = data1 - data_exp1_1
    data2 = data2 - data_exp1_2
    data3 = data3 - data_exp1_3
    data = - 0.25 * (data0 + data1 + data2 + data3)
    # other parameter definitions
    cmin = -2
    cmax = 2
    fname = 'innov_delta_std_single'
    figure_single_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title='change in std(O-F) (K)')


def plot_innov_delta_std_quatro(exp1, exp2, domain, root, outpath):
    # H-Asc
    # H-Des
    # V-Asc
    # V-Des
    exp = exp1
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
    data_exp1_0 = obs_M09_to_M36(data_std0)
    data_exp1_1 = obs_M09_to_M36(data_std1)
    data_exp1_2 = obs_M09_to_M36(data_std2)
    data_exp1_3 = obs_M09_to_M36(data_std3)
    # set up grid
    io = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    # calculate variable to plot
    data = io.timeseries['obs_obs'] - io.timeseries['obs_fcst']
    data = data.where(data != 0)
    data_std0 = data[:, 0, :, :].std(dim='time', skipna=True).values
    data_std1 = data[:, 1, :, :].std(dim='time', skipna=True).values
    data_std2 = data[:, 2, :, :].std(dim='time', skipna=True).values
    data_std3 = data[:, 3, :, :].std(dim='time', skipna=True).values
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
    fname = 'innov_delta_std_quatro'
    data = list([data0, data1, data2, data3])
    figure_quatro_default(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['d_innov std (Tb) H-Asc', 'd_innov std (Tb) H-Des', 'd_innov std (Tb) V-Asc',
                                           'd_innov std (Tb) V-Des']))


def figure_quatro_default(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                          llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title, mstats):
    # open figure
    figsize = (10, 10)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0, 4):
        cmap = 'coolwarm_r'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0 and cmax > 0.0:
            cmax = np.max([-cmin, cmax])
            cmin = -cmax
            cmap = 'coolwarm_r'
        cbrange = (cmin, cmax)
        ax = plt.subplot(4, 1, i + 1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        parallels = np.arange(-80.0, 81, 5.)
        m.drawparallels(parallels, linewidth=0.5, labels=[True, False, False, False])
        meridians = np.arange(0., 351., 20.)
        m.drawmeridians(meridians, linewidth=0.5, labels=[False, False, False, True])
        #         http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
        # lat=48.
        # lon=51.0
        # x,y = m(lon, lat)
        # m.plot(x, y, 'ko', markersize=6,mfc='none')
        if fname == '':
            bounds = np.array([-2., -1., -0.1, 0.1, 1., 2.])
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = m.pcolormesh(lons, lats, plt_img, norm=norm, cmap=cmap, latlon=True)
            cb = m.colorbar(im, "bottom", size="7%", pad="15%", extend='both')
        else:
            # color bar
            im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
            im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
            # cb = m.colorbar(im, "bottom", size="7%", pad="22%", shrink=0.5)
            # cb = matplotlib.pyplot.colorbar(im)
            if np.mean(lats) > 40:
                im_ratio = np.shape(data)[1] / np.shape(data)[2]
                cb = matplotlib.pyplot.colorbar(im, fraction=0.13 * im_ratio, pad=0.02)
            else:
                cb = matplotlib.pyplot.colorbar(im)
            # ticklabs = cb.ax.get_yticklabels()
            # cb.ax.set_yticklabels(ticklabs,ha='right')
            # cb.ax.yaxis.set_tick_params(pad=45)  # your number may vary
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
            t.set_horizontalalignment('right')
            if np.mean(lats) > 40:
                t.set_x(9.0)
            else:
                t.set_x(4.0)
        plt.title(plot_title[i], fontsize=fontsize)
        if np.mean(lats) > 40:
            matplotlib.pyplot.text(1.0, 1.0, mstats[i], horizontalalignment='right', verticalalignment='bottom',
                                   transform=ax.transAxes, fontsize=fontsize-2)
        else:
            matplotlib.pyplot.text(1.0, 0.07, mstats[i], bbox=dict(facecolor='white', alpha=1.0),
                                   horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                                   fontsize=fontsize-2)
    fname_long = os.path.join(outpath, fname + '.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def figure_triple_default(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                          llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title, mstats, cmap='seismic'):
    # open figure
    if np.mean(lats) > 40:
        figsize = (0.85 * 13, 0.85 * 10)
    else:
        figsize = (0.85 * 7, 0.85 * 16)
    fontsize = 13
    f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
    for i in np.arange(0, 3):
        if np.size(cmin) > 1:
            cmin_ = cmin[i]
        else:
            cmin_ = cmin
        if np.size(cmax) > 1:
            cmax_ = cmax[i]
        else:
            cmax_ = cmax
        if (cmin_ == None) | (cmin_ == -9999):
            cmin_ = np.nanmin(data[i])
        if (cmax_ == None) | (cmax_ == -9999):
            cmax_ = np.nanmax(data[i])
        if cmin_ < 0.0 and cmax_ > 0.0:
            cmax_ = np.max([-cmin_, cmax_])
            cmin_ = -cmax_
            cmap = 'seismic'
            if plot_title[0].find('R_') != -1:
                cmap = 'seismic_r'
        cbrange = (cmin_, cmax_)
        ax = plt.subplot(3, 1, i + 1)
        plt_img = np.ma.masked_invalid(data[i])
        # m=Basemap(projection='merc',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=-170.,urcrnrlon=urcrnrlon,resolution='l')
        m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        parallels = np.arange(-80.0, 81, 5.)
        m.drawparallels(parallels, linewidth=0.5, labels=[True, False, False, False])
        meridians = np.arange(0., 351., 20.)
        m.drawmeridians(meridians, linewidth=0.5, labels=[False, False, False, True])
        #         http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
        # lat=48.
        # lon=51.0
        # x,y = m(lon, lat)
        # m.plot(x, y, 'ko', markersize=6,mfc='none')
        if fname == '':
            bounds = np.array([-2., -1., -0.1, 0.1, 1., 2.])
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = m.pcolormesh(lons, lats, plt_img, norm=norm, cmap=cmap, latlon=True)
            cb = m.colorbar(im, "bottom", size="7%", pad="15%", extend='both')
        else:
            # color bar
            im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
            im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
            # cb = m.colorbar(im, "bottom", size="7%", pad="22%", shrink=0.5)
            # cb = matplotlib.pyplot.colorbar(im)
            if np.mean(lats) > 40:
                im_ratio = np.shape(data)[1] / np.shape(data)[2]
                cb = matplotlib.pyplot.colorbar(im, fraction=0.13 * im_ratio, pad=0.02)
            else:
                cb = matplotlib.pyplot.colorbar(im)
            # ticklabs = cb.ax.get_yticklabels()
            # cb.ax.set_yticklabels(ticklabs,ha='right')
            # cb.ax.yaxis.set_tick_params(pad=45)  # your number may vary
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
            t.set_horizontalalignment('right')
            if np.mean(lats) > 40:
                t.set_x(9.0)
            else:
                t.set_x(4.0)
        tit = plt.title(plot_title[i], fontsize=fontsize)
        if np.mean(lats) > 40:
            matplotlib.pyplot.text(1.0, 1.0, mstats[i], horizontalalignment='right', verticalalignment='bottom',
                                   transform=ax.transAxes, fontsize=fontsize)
        else:
            matplotlib.pyplot.text(1.0, 0.2, mstats[i], bbox=dict(facecolor='white', alpha=1.0),
                                   horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                                   fontsize=fontsize)
    fname_long = os.path.join(outpath, fname + '.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def figure_quatro_scaling(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                          llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title, mstats):
    # open figure
    figsize = (10, 10)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0, 4):
        cmap = 'jet'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0:
            cmax = np.max([-cmin, cmax])
            cmin = -cmax
            cmap = 'seismic'
        if i == 2:
            cmax = cmax * 0.5

        cbrange = (cmin, cmax)
        plt.subplot(4, 1, i + 1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(-80.0, 81, 5.)
        m.drawparallels(parallels, labels=[False, True, True, False])
        meridians = np.arange(0., 351., 10.)
        m.drawmeridians(meridians, labels=[True, False, False, True])
        lat = 53.
        lon = 38.5
        lon = -103
        lat = 47
        x, y = m(lon, lat)
        m.plot(x, y, 'ko', markersize=6, mfc='none')
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="4%", pad="16%")
        # scatter
        # x, y = m(lons, lats)
        # ax.scatter(x, y, s=10, c=res['m_mod_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)
        # label size
        if np.mean(lats) > 40:
            matplotlib.pyplot.text(1.0, 1.0, mstats[i], horizontalalignment='right', verticalalignment='bottom',
                                   transform=ax.transAxes, fontsize=fontsize)
        else:
            matplotlib.pyplot.text(1.0, 0.2, mstats[i], bbox=dict(facecolor='white', alpha=1.0),
                                   horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                                   fontsize=fontsize)

        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
        plt.title(plot_title[i], fontsize=fontsize)

    fname_long = os.path.join(outpath, fname + '.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def figure_double_scaling(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                          llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title):
    # open figure
    figsize = (17, 8)
    fontsize = 14
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    for i in np.arange(0, 2):
        cmap = 'jet'
        if cmin == None:
            cmin = np.nanmin(data[i])
        if cmax == None:
            cmax = np.nanmax(data[i])
        if cmin < 0.0:
            cmax = np.max([-cmin, cmax])
            cmin = -cmax
            cmap = 'seismic'

        cbrange = (cmin, cmax)
        plt.subplot(2, 1, i + 1)
        plt_img = np.ma.masked_invalid(data[i])
        m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon, resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(-80.0, 81, 5.)
        m.drawparallels(parallels, labels=[False, True, True, False])
        meridians = np.arange(0., 351., 10.)
        m.drawmeridians(meridians, labels=[True, False, False, True])
        # color bar
        im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
        cb = m.colorbar(im, "bottom", size="4%", pad="16%")
        # scatter
        # x, y = m(lons, lats)
        # ax.scatter(x, y, s=10, c=res['m_mod_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)
        # label size
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)
        plt.title(plot_title[i], fontsize=fontsize)

    fname_long = os.path.join(outpath, fname + '.png')
    plt.tight_layout()
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def figure_zoom(data, lons, lats, latmin, latmax, lonmin, lonmax):
    # data: xarray or ndarray with 2D data array
    # lons and lats: 2D arrays with lons and lats of each grid cell
    # latmin, latmax, lonmin, lonmax: zoom boundaries
    # Usage:
    # [data_zoom,lons_zoom,lats_zoom,llcrnrlat_zoom, urcrnrlat_zoom, llcrnrlon_zoom,urcrnrlon_zoom] = figure_zoom(tmp_data.values,lons,lats,latmin,latmax,lonmin,lonmax)
    cond = np.where((lons > lonmin) & (lons < lonmax) & (lats > latmin) & (lats < latmax), 1, np.nan)
    colind = np.all(np.isnan(cond), axis=0) == False
    rowind = np.all(np.isnan(cond), axis=1) == False
    try:
        data = data.values[:, colind]
    except:
        data = data[:, colind]
    data = data[rowind, :]
    lons = lons[:, colind]
    lons = lons[rowind, :]
    lats = lats[:, colind]
    lats = lats[rowind, :]
    llcrnrlat = np.min(lats)
    urcrnrlat = np.max(lats)
    llcrnrlon = np.min(lons)
    urcrnrlon = np.max(lons)
    return data, lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon


def figure_single_default(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                          llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title, cmap='coolwarm_r'):
    # if plot_title.startswith('zbar'):
    #    cmap = 'jet_r'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < -3.0 and cmap == 'seismic':
        cmax = np.max([-cmin, cmax])
        cmin = -cmax
    # open figure
    # Norther peatland:
    if np.mean(lats) > 30:
        fig_aspect_ratio = (0.1 * (np.max(lons) - np.min(lons))) / (0.18 * (np.max(lats) - np.min(lats)))
        figsize = (fig_aspect_ratio + 10, 10)
        parallels = np.arange(-80.0, 81, 5.)
        meridians = np.arange(0., 351., 20.)
    else:
        figsize = (10, 10)
        parallels = np.arange(10 * np.floor(np.min(lats) / 10.), 10 * np.ceil(np.max(lats) / 10.),
                              np.ceil((np.max(lats) - np.min(lats)) / 4.))
        meridians = np.arange(10 * np.floor(np.min(lons) / 10.), 10 * np.ceil(np.max(lons) / 10.),
                              np.ceil((np.max(lons) - np.min(lons)) / 4.))
    fontsize = 14
    cbrange = (cmin, cmax)

    f = plt.figure(num=None, figsize=figsize, dpi=450, facecolor='w', edgecolor='k')
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(parallels, labels=[False, False, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, False])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    # lat=53.
    # lon=38.5
    # x,y = m(lon, lat)
    # m.plot(x, y, 'ko', markersize=6,mfc='none')
    # cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "right", size="7%", pad="4%", shrink=0.5)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize + 10)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize + 10)
    plt.title(plot_title, fontsize=fontsize + 6)
    # plt.tight_layout()
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=f.dpi)
    plt.close()


def figure_single_default_mb(data, lons, lats, cmin, cmax, llcrnrlat, urcrnrlat,
                             llcrnrlon, urcrnrlon, outpath, exp, fname, plot_title, cmap='jet'):
    # if plot_title.startswith('zbar'):
    #    cmap = 'jet_r'
    if cmin == None:
        cmin = np.nanmin(data)
    if cmax == None:
        cmax = np.nanmax(data)
    if cmin < 0.0 and cmax > 0.0:
        cmax = np.max([-cmin, cmax])
        cmin = -cmax
        cmap = 'seismic'
    cmap = 'jet'
    # open figure
    # Norther peatland:
    if np.mean(lats) > 30:
        fig_aspect_ratio = (0.1 * (np.max(lons) - np.min(lons))) / (0.18 * (np.max(lats) - np.min(lats)))
        figsize = (fig_aspect_ratio + 10, 10)
        parallels = np.arange(-80.0, 81, 5.)
        meridians = np.arange(0., 351., 20.)
    else:
        figsize = (10, 10)
        parallels = np.arange(10 * np.floor(np.min(lats) / 10.), 10 * np.ceil(np.max(lats) / 10.),
                              np.ceil((np.max(lats) - np.min(lats)) / 4.))
        meridians = np.arange(10 * np.floor(np.min(lons) / 10.), 10 * np.ceil(np.max(lons) / 10.),
                              np.ceil((np.max(lons) - np.min(lons)) / 4.))
    fontsize = 14
    cbrange = (cmin, cmax)

    mstats = 'm = %.4f, s = %.4f' % (np.nanmean(data), np.nanstd(data))
    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')
    ax = plt.subplot(1, 1, 1)
    plt_img = np.ma.masked_invalid(data)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(parallels, labels=[False, True, True, False])
    m.drawmeridians(meridians, labels=[True, False, False, True])
    # color bar
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    # lat=53.
    # lon=38.5
    # x,y = m(lon, lat)
    # m.plot(x, y, 'ko', markersize=6,mfc='none')
    # cmap = plt.get_cmap(cmap)
    cb = m.colorbar(im, "bottom", size="6%", pad="22%", shrink=0.5)
    # label size
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title(plot_title, fontsize=fontsize)
    if np.mean(lats) > 40:
        matplotlib.pyplot.text(1.0, 1.0, mstats, horizontalalignment='right', verticalalignment='bottom',
                               transform=ax.transAxes, fontsize=fontsize)
    else:
        matplotlib.pyplot.text(1.0, 0.2, mstats, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
                               verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
    # plt.tight_layout()
    fname_long = os.path.join(outpath, fname + '.png')
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
    outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Drained'

    fname = r"D:\work\LDAS\2018-02_scaling\ismn_eval\no_da_cal_uncal_ma_harm\validation_masked.csv"

    res = pd.read_csv(fname, index_col=0)

    variables = ['sm_surface', 'sm_rootzone', 'sm_profile']
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
    root2 = '/staging/leuven/stg_00024/OUTPUT/michelb'
    root2 = root
    io_daily_PCLSM_OL = LDAS_io('daily', exp=exp[0:-3], domain=domain, root=root2)
    io_daily_CLSM_OL = LDAS_io('daily', exp=exp2[0:-3], domain=domain, root=root2)

    # io = LDAS_io('ObsFcstAna', exp='SMAP_EASEv2_M09_SMOSfw', domain=domain, root=root)
    # io_CLSM = LDAS_io('ObsFcstAna', exp='SMAP_EASEv2_M09_CLSM_SMOSfw', domain=domain, root=root)
    io = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)
    io_CLSM = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)
    io_incr = LDAS_io('incr', exp=exp, domain=domain, root=root)
    io_incr_CLSM = LDAS_io('incr', exp=exp2, domain=domain, root=root)
    # io_ens = LDAS_io('ensstd', exp=exp, domain=domain, root=root)
    # io_ens_CLSM = LDAS_io('ensstd', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    [col, row] = get_M09_ObsFcstAna(io, lon, lat)

    dcols = [-2, -1, 0, 1, 2]
    drows = [-2, -1, 0, 1, 2]
    col_ori = col
    row_ori = row
    for icol, dcol in enumerate(dcols):
        for icol, drow in enumerate(drows):
            col = col_ori + dcol * 4
            row = row_ori + drow * 4
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
            ts_innov_1[ts_assim_1 != -1] = np.nan
            ts_innov_1.name = 'innov (species=1,PCLSM)'
            ts_innov_2 = ts_obs_2 - ts_fcst_2
            ts_innov_2[ts_assim_2 != -1] = np.nan
            ts_innov_2.name = 'innov (species=2,PCLSM)'
            ts_innov_3 = ts_obs_3 - ts_fcst_3
            ts_innov_3[ts_assim_3 != -1] = np.nan
            ts_innov_3.name = 'innov (species=3,PCLSM)'
            ts_innov_4 = ts_obs_4 - ts_fcst_4
            ts_innov_4[ts_assim_4 != -1] = np.nan
            ts_innov_4.name = 'innov (species=4,PCLSM)'

            # incr PCLSM
            ts_incr_catdef = io_incr.read_ts('catdef', col, row, lonlat=False).replace(0, np.nan)
            ts_incr_catdef.name = 'catdef incr'
            ts_incr_srfexc = io_incr.read_ts('srfexc', col, row, lonlat=False).replace(0, np.nan)
            ts_incr_srfexc.name = 'srfexc incr'

            # ensstd PCLSM
            # ts_ens_catdef = io_ens.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            # ts_ens_catdef.name = 'catdef'
            # ts_ens_srfexc = io_ens.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            # ts_ens_srfexc.name = 'srfexc'

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
            ts_CLSM_innov_1[ts_CLSM_assim_1 != -1] = np.nan
            ts_CLSM_innov_1.name = 'innov (species=1,CLSM)'
            ts_CLSM_innov_2 = ts_CLSM_obs_2 - ts_CLSM_fcst_2
            ts_CLSM_innov_2[ts_CLSM_assim_2 != -1] = np.nan
            ts_CLSM_innov_2.name = 'innov (species=2,CLSM)'
            ts_CLSM_innov_3 = ts_CLSM_obs_3 - ts_CLSM_fcst_3
            ts_CLSM_innov_3[ts_CLSM_assim_3 != -1] = np.nan
            ts_CLSM_innov_3.name = 'innov (species=3,CLSM)'
            ts_CLSM_innov_4 = ts_CLSM_obs_4 - ts_CLSM_fcst_4
            ts_CLSM_innov_4[ts_CLSM_assim_4 != -1] = np.nan
            ts_CLSM_innov_4.name = 'innov (species=4,CLSM)'

            # incr CLSM
            ts_incr_CLSM_catdef = io_incr_CLSM.read_ts('catdef', col, row, lonlat=False).replace(0, np.nan)
            ts_incr_CLSM_catdef.name = 'catdef incr (CLSM)'
            ts_incr_CLSM_srfexc = io_incr_CLSM.read_ts('srfexc', col, row, lonlat=False).replace(0, np.nan)
            ts_incr_CLSM_srfexc.name = 'srfexc incr (CLSM)'

            # ensstd PCLSM
            # ts_ens_CLSM_catdef = io_ens_CLSM.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
            # ts_ens_CLSM_catdef.name = 'catdef (CLSM)'
            # ts_ens_CLSM_srfexc = io_ens_CLSM.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
            # ts_ens_CLSM_srfexc.name = 'srfexc (CLSM)'

            # daily CLSM
            ts_catdef_CLSM = io_daily_CLSM.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_CLSM.name = 'analysis (CLSM)'
            ts_catdef_CLSM_OL = io_daily_CLSM_OL.read_ts('catdef', col, row, lonlat=False)
            ts_catdef_CLSM_OL.name = 'open loop (CLSM)'

            df = pd.concat((ts_obs_1, ts_CLSM_obs_1, ts_innov_1, ts_innov_2, ts_innov_3, ts_innov_4, ts_incr_catdef,
                            ts_ana_1, ts_ana_2, ts_ana_3, ts_ana_4, ts_assim_1, ts_assim_2, ts_assim_3, ts_assim_4,
                            ts_catdef_PCLSM, ts_catdef_PCLSM_OL,
                            ts_CLSM_innov_1, ts_CLSM_innov_2, ts_CLSM_innov_3, ts_CLSM_innov_4, ts_incr_CLSM_catdef,
                            ts_CLSM_ana_1, ts_CLSM_ana_2, ts_CLSM_ana_3, ts_CLSM_ana_4, ts_CLSM_assim_1,
                            ts_CLSM_assim_2, ts_CLSM_assim_3, ts_CLSM_assim_4,
                            ts_catdef_CLSM, ts_catdef_CLSM_OL), axis=1)
            # df = pd.concat((ts_obs_1, ts_obs_2, ts_obs_3, ts_obs_4, ts_fcst_1, ts_fcst_2, ts_fcst_3, ts_fcst_4, ts_incr_catdef),axis=1).dropna()
            # mask = (df.index > '2016-05-01 00:00:00') & (df.index <= '2016-06-01 00:00:00')
            # df = df.loc[mask]
            plt.figure(figsize=(19, 7))
            fontsize = 12

            ax1 = plt.subplot(311)
            # df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            # df[['innov (species=1,PCLSM)','innov (species=2,PCLSM)','innov (species=3,PCLSM)','innov (species=4,PCLSM)',
            #    'innov (species=1,CLSM)','innov (species=2,CLSM)','innov (species=3,CLSM)','innov (species=4,CLSM)',]].plot(ax=ax1,
            #    fontsize=fontsize, style=['o','o','o','o','x','x','x','x'], linewidth=2)
            df[['Tb obs (species=1)', 'Tb obs (species=1,CLSM)']].plot(ax=ax1,
                                                                       fontsize=fontsize, style=['o', 'x'], linewidth=2)
            plt.ylabel('innov (O-F) [(K)]')

            ax2 = plt.subplot(312)
            df[['catdef incr']].plot(ax=ax2, fontsize=fontsize, style=['o:'], linewidth=2)
            plt.ylabel('incr [mm]')
            my_title = 'lon=%s, lat=%s' % (lon, lat)
            plt.title(my_title, fontsize=fontsize + 2)

            # ax3 = plt.subplot(413)
            # df[['a1','a2','a3','a4']].plot(ax=ax3, fontsize=fontsize, style=['.-','.-','.-','.-'], linewidth=2)
            # plt.ylabel('Tb [(K)]')

            ax4 = plt.subplot(313)
            # df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            df[['analysis (PCLSM)', 'analysis (CLSM)', 'open loop (PCLSM)', 'open loop (CLSM)']].plot(ax=ax4,
                                                                                                      fontsize=fontsize,
                                                                                                      style=['.-', '.-',
                                                                                                             '.-',
                                                                                                             '.-'],
                                                                                                      linewidth=2)
            plt.ylabel('catdef [mm]')

            # ax5 = plt.subplot(313)
            ##df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
            # df[['srfexc','srfexc (CLSM)']].plot(ax=ax5, fontsize=fontsize, style=['.-','x-'], linewidth=2)
            # plt.ylim(bottom=0)
            # plt.ylabel('ensstd [mm]')

            plt.tight_layout()
            fname = 'innov_incr_col_%s_row_%s' % (col, row)
            fname_long = os.path.join(outpath, exp, fname + '.png')
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


def plot_scaling_parameters_timeseries(exp, domain, root, outpath, scalepath='', scalename='', lon='', lat=''):
    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('ObsFcstAna', exp, domain, root)
    [col, row] = get_M09_ObsFcstAna(io, lon, lat)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # fpath = os.path.join(root,exp,'output',domain,'stats/z_score_clim/pentad/')
    if scalepath == '':
        scalepath = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalepath'].split()[0]
        # fname = 'ScMO_Thvf_TbSM_001_SMOS_zscore_stats_2010_p1_2018_p73_hscale_0.00_W_9p_Nmin_20'
        scalename = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalename'][5:].split()[0]
    mod_ha = []
    mod_va = []
    mod_hd = []
    mod_vd = []
    obs_ha = []
    obs_va = []
    obs_hd = []
    obs_vd = []
    for i_AscDes, AscDes in enumerate(list(["A", "D"])):
        for i_pentad in np.arange(1, 74):
            fname = scalepath + scalename + '_' + AscDes + '_p' + "%02d" % (i_pentad,) + '.bin'
            res = io.read_scaling_parameters(fname=fname)

            res = res[['lon', 'lat', 'm_mod_H_%2i' % angle, 'm_mod_V_%2i' % angle, 'm_obs_H_%2i' % angle,
                       'm_obs_V_%2i' % angle]]
            res.replace(-9999., np.nan, inplace=True)

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data0 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data1 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data2 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data3 = np.ma.masked_invalid(img)

            if AscDes == 'A':
                mod_ha.append(data0.data[row, col])
                mod_va.append(data1.data[row, col])
                obs_ha.append(data2.data[row, col])
                obs_va.append(data3.data[row, col])
            else:
                mod_hd.append(data0.data[row, col])
                mod_vd.append(data1.data[row, col])
                obs_hd.append(data2.data[row, col])
                obs_vd.append(data3.data[row, col])

    plt.figure(figsize=(6.5, 13))
    fontsize = 12
    ax1 = plt.subplot(411)
    plt.plot(np.arange(1, 74), mod_ha, color='red')
    plt.plot(np.arange(1, 74), obs_ha, color='black')
    plt.legend(['mod', 'obs'])
    plt.xlabel('pentad')
    plt.ylabel('H-Asc')
    plt.xlim(1, 73)
    plt.ylim(210, 300)
    ax2 = plt.subplot(412)
    plt.plot(np.arange(1, 74), mod_va, color='red')
    plt.plot(np.arange(1, 74), obs_va, color='black')
    plt.legend(['mod', 'obs'])
    plt.xlabel('pentad')
    plt.ylabel('V-Asc')
    plt.xlim(1, 73)
    plt.ylim(210, 300)
    ax3 = plt.subplot(413)
    plt.plot(np.arange(1, 74), mod_hd, color='red')
    plt.plot(np.arange(1, 74), obs_hd, color='black')
    plt.legend(['mod', 'obs'])
    plt.xlabel('pentad')
    plt.ylabel('H-Des')
    plt.xlim(1, 73)
    plt.ylim(210, 300)
    ax4 = plt.subplot(414)
    plt.plot(np.arange(1, 74), mod_vd, color='red')
    plt.plot(np.arange(1, 74), obs_vd, color='black')
    plt.legend(['mod', 'obs'])
    plt.xlabel('pentad')
    plt.ylabel('V-Des')
    plt.xlim(1, 73)
    plt.ylim(210, 300)

    fname_long = os.path.join(outpath, 'lon_%02d_' % (lon,) + 'lat_%02d_' % (lat,) + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_scaling_parameters(exp, domain, root, outpath, scalepath='', scalename=''):
    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('ObsFcstAna', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # fpath = os.path.join(root,exp,'output',domain,'stats/z_score_clim/pentad/')
    if scalepath == '':
        scalepath = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalepath'].split()[0]
        # fname = 'ScMO_Thvf_TbSM_001_SMOS_zscore_stats_2010_p1_2018_p73_hscale_0.00_W_9p_Nmin_20'
        scalename = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalename'][5:].split()[0]
    for i_AscDes, AscDes in enumerate(list(["A", "D"])):
        for i_pentad in np.arange(37, 39):
            fname = scalepath + scalename + '_' + AscDes + '_p' + "%02d" % (i_pentad,) + '.bin'
            res = io.read_scaling_parameters(fname=fname)

            res = res[['lon', 'lat', 'm_mod_H_%2i' % angle, 'm_mod_V_%2i' % angle, 'm_obs_H_%2i' % angle,
                       'm_obs_V_%2i' % angle]]
            res.replace(-9999., np.nan, inplace=True)

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data0 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data1 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data2 = np.ma.masked_invalid(img)
            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data3 = np.ma.masked_invalid(img)
            # other parameter definitions
            cmin = 210
            cmax = 300
            fname = 'scaling' + '_' + AscDes + '_p' + "%02d" % (i_pentad,)
            data = list([data0, data1, data2, data3])
            figure_quatro_scaling(data, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                                  urcrnrlat=urcrnrlat,
                                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                                  plot_title=list(['m_mod_H_%2i' % angle, 'm_mod_V_%2i' % angle, 'm_obs_H_%2i' % angle,
                                                   'm_obs_V_%2i' % angle]))


def plot_scaling_parameters_average(exp, domain, root, outpath):
    angle = 40
    outpath = os.path.join(outpath, exp, 'scaling')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('ObsFcstAna', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids

    # get scaling nc
    ncpath = io.paths.root + '/' + exp + '/output_postprocessed/'
    ds = xr.open_dataset(ncpath + 'scaling.nc')
    AscDes = ['A', 'D']

    #### std_mod
    data0 = ds['m_obs_H_%2i' % angle][:, :, :, 0].std(axis=2)
    data1 = ds['m_obs_H_%2i' % angle][:, :, :, 1].std(axis=2)
    data2 = ds['m_obs_V_%2i' % angle][:, :, :, 0].std(axis=2)
    data3 = ds['m_obs_V_%2i' % angle][:, :, :, 1].std(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 20
    fname = 'std_obs'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    #### std_mod
    data0 = ds['m_mod_H_%2i' % angle][:, :, :, 0].std(axis=2)
    data1 = ds['m_mod_H_%2i' % angle][:, :, :, 1].std(axis=2)
    data2 = ds['m_mod_V_%2i' % angle][:, :, :, 0].std(axis=2)
    data3 = ds['m_mod_V_%2i' % angle][:, :, :, 1].std(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 20
    fname = 'std_mod'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    #### mean_obs
    data0 = ds['m_obs_H_%2i' % angle][:, :, :, 0].mean(axis=2)
    data1 = ds['m_obs_H_%2i' % angle][:, :, :, 1].mean(axis=2)
    data2 = ds['m_obs_V_%2i' % angle][:, :, :, 0].mean(axis=2)
    data3 = ds['m_obs_V_%2i' % angle][:, :, :, 1].mean(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 240
    cmax = 285
    fname = 'mean_obs'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    #### mean_mod
    data0 = ds['m_mod_H_%2i' % angle][:, :, :, 0].mean(axis=2)
    data1 = ds['m_mod_H_%2i' % angle][:, :, :, 1].mean(axis=2)
    data2 = ds['m_mod_V_%2i' % angle][:, :, :, 0].mean(axis=2)
    data3 = ds['m_mod_V_%2i' % angle][:, :, :, 1].mean(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 240
    cmax = 285
    fname = 'mean_mod'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    #### mean_mod - mean_obs
    data0 = ds['m_mod_H_%2i' % angle][:, :, :, 0].mean(axis=2) - ds['m_obs_H_%2i' % angle][:, :, :, 0].mean(axis=2)
    data1 = ds['m_mod_H_%2i' % angle][:, :, :, 1].mean(axis=2) - ds['m_obs_H_%2i' % angle][:, :, :, 1].mean(axis=2)
    data2 = ds['m_mod_V_%2i' % angle][:, :, :, 0].mean(axis=2) - ds['m_obs_V_%2i' % angle][:, :, :, 0].mean(axis=2)
    data3 = ds['m_mod_V_%2i' % angle][:, :, :, 1].mean(axis=2) - ds['m_obs_V_%2i' % angle][:, :, :, 1].mean(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname = 'mean_mod-mean_obs'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    #### mean_mod - mean_obs
    data0 = (ds['m_mod_H_%2i' % angle][:, :, :, 0] - ds['m_obs_H_%2i' % angle][:, :, :, 0]).mean(axis=2)
    data1 = (ds['m_mod_H_%2i' % angle][:, :, :, 1] - ds['m_obs_H_%2i' % angle][:, :, :, 1]).mean(axis=2)
    data2 = (ds['m_mod_V_%2i' % angle][:, :, :, 0] - ds['m_obs_V_%2i' % angle][:, :, :, 0]).mean(axis=2)
    data3 = (ds['m_mod_V_%2i' % angle][:, :, :, 1] - ds['m_obs_V_%2i' % angle][:, :, :, 1]).mean(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = None
    cmax = None
    fname = 'mean_mod-obs'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(['H-Asc', 'H-Des', 'V-Asc', 'V-Des']))

    AscDes = ['A', 'D']
    data0 = ds['m_obs_H_%2i' % angle][:, :, :, 0].count(axis=2)
    data1 = ds['m_obs_H_%2i' % angle][:, :, :, 1].count(axis=2)
    data2 = ds['m_obs_V_%2i' % angle][:, :, :, 0].count(axis=2)
    data3 = ds['m_obs_V_%2i' % angle][:, :, :, 1].count(axis=2)
    data0 = data0.where(data0 != 0)
    data1 = data1.where(data1 != 0)
    data2 = data2.where(data2 != 0)
    data3 = data3.where(data3 != 0)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 73
    fname = 'n_valid_scaling'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(
                              ['N pentads (H-Asc)', 'N pentads (H-Des)', 'N pentads (V-Asc)', 'N pentads (V-Des)']))

    AscDes = ['A', 'D']
    tmp = ds['m_mod_H_%2i' % angle][:, :, :, 0].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data0 = tmp.sum(axis=2)
    tmp = ds['m_mod_H_%2i' % angle][:, :, :, 1].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data1 = tmp.sum(axis=2)
    tmp = ds['m_mod_V_%2i' % angle][:, :, :, 0].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data2 = tmp.sum(axis=2)
    tmp = ds['m_mod_V_%2i' % angle][:, :, :, 1].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    data3 = tmp.sum(axis=2)
    np.place(data0, data0 == 0, np.nan)
    np.place(data1, data1 == 0, np.nan)
    np.place(data2, data2 == 0, np.nan)
    np.place(data3, data3 == 0, np.nan)
    data0 = obs_M09_to_M36(data0)
    data1 = obs_M09_to_M36(data1)
    data2 = obs_M09_to_M36(data2)
    data3 = obs_M09_to_M36(data3)
    cmin = 0
    cmax = 73
    fname = 'n_valid_scaling_mod'
    data_all = list([data0, data1, data2, data3])
    figure_quatro_scaling(data_all, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=list(
                              ['N pentads (H-Asc)', 'N pentads (H-Des)', 'N pentads (V-Asc)', 'N pentads (V-Des)']))


def plot_timeseries_RSEpaper(exp1, exp2, domain, root, outpath, lat=53, lon=25):
    import matplotlib.gridspec as gridspec
    ## in situ data
    insitu_path = '/data/leuven/317/vsc31786/peatland_data'
    mastertable_filename = 'WTD_peatlands_global_WGS84_DA.csv'
    wtd_obs, wtd_mod_exp1, precip_obs, precip_mod = read_wtd_data(insitu_path, mastertable_filename, exp1, domain, root)
    # mastertable for locations
    filenames = find_files(insitu_path, mastertable_filename)
    if isinstance(find_files(insitu_path, mastertable_filename), str):
        master_table = pd.read_csv(filenames, sep=',')
    else:
        for filename in filenames:
            if filename.endswith('csv'):
                master_table = pd.read_csv(filename, sep=',')
                continue
            else:
                logging.warning("some files, maybe swp files, exist that start with master table searchstring !")

    # simulations
    daily_CLSM_OL = LDAS_io('daily', exp=exp1[0:-3], domain=domain, root=root)
    daily_PCLSM_OL = LDAS_io('daily', exp=exp2[0:-3], domain=domain, root=root)
    daily_CLSM_DA = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    daily_PCLSM_DA = LDAS_io('daily', exp=exp2, domain=domain, root=root)

    ObsFcstAna_CLSM_OL = LDAS_io('ObsFcstAna', exp=exp1[0:-3], domain=domain, root=root)
    ObsFcstAna_PCLSM_OL = LDAS_io('ObsFcstAna', exp=exp2[0:-3], domain=domain, root=root)
    ObsFcstAna_CLSM_DA = LDAS_io('ObsFcstAna', exp=exp1, domain=domain, root=root)
    ObsFcstAna_PCLSM_DA = LDAS_io('ObsFcstAna', exp=exp2, domain=domain, root=root)

    catparam = LDAS_io(exp=exp1, domain=domain, root=root).read_params('catparam')
    catparam_PCLSM = LDAS_io(exp=exp2, domain=domain, root=root).read_params('catparam')

    incr_CLSM = LDAS_io('incr', exp=exp1, domain=domain, root=root)
    incr_PCLSM = LDAS_io('incr', exp=exp2, domain=domain, root=root)
    # io_ens = LDAS_io('ensstd', exp=exp, domain=domain, root=root)
    # io_ens_CLSM = LDAS_io('ensstd', exp=exp2, domain=domain, root=root)
    # ncpath = ObsFcstAna_CLSM_OL.paths.root +'/' + exp1 + '/output_postprocessed/'
    # scaling_CLSM = xr.open_dataset(ncpath + 'scaling.nc')
    # ncpath = ObsFcstAna_PCLSM_OL.paths.root +'/' + exp2 + '/output_postprocessed/'
    # scaling_PCLSM = xr.open_dataset(ncpath + 'scaling.nc')

    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(daily_PCLSM_OL)
    # lon = -110.
    # lat=52.
    # EE
    lon = 26.2163
    lat = 58.8789
    # SI
    lon = 71.5
    lat = 58.1
    # CA_JB_Fen2
    lon = -83.9534418
    lat = 52.7241324
    site_name = 'CA_JB_Bog2'
    site_ind = np.where(master_table['Pegel_ID'] == site_name)[0][0]
    lon = master_table['lon'][site_ind]
    lat = master_table['lat'][site_ind]
    [col, row] = get_M09_ObsFcstAna(ObsFcstAna_PCLSM_OL, lon, lat)
    ObsFcstAna_PCLSM_OL.timeseries['obs_obs'][:, 0, row, col]
    poros = np.full(lons.shape, np.nan)
    poros[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam[
        'poros'].values
    if poros[row, col] > 0.65:
        print("It is peat")
    bf1 = np.full(lons.shape, np.nan)
    bf1[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam_PCLSM[
        'bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam_PCLSM[
        'bf2'].values
    ars1 = np.full(lons.shape, np.nan)
    ars1[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam_PCLSM[
        'ars1'].values
    ars2 = np.full(lons.shape, np.nan)
    ars2[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam_PCLSM[
        'ars2'].values
    ars3 = np.full(lons.shape, np.nan)
    ars3[daily_PCLSM_OL.grid.tilecoord.j_indg.values, daily_PCLSM_OL.grid.tilecoord.i_indg.values] = catparam_PCLSM[
        'ars3'].values

    ###################### PLSM #######################
    # daily PCLSM
    ts_catdef_PCLSM_DA = daily_PCLSM_DA.read_ts('catdef', col, row, lonlat=False)
    ts_catdef_PCLSM_DA.name = 'catdef analysis (PCLSM)'
    ts_catdef_PCLSM_OL = daily_PCLSM_OL.read_ts('catdef', col, row, lonlat=False)
    ts_catdef_PCLSM_OL.name = 'catdef open loop (PCLSM)'
    ts_zbar_PCLSM_DA = daily_PCLSM_DA.read_ts('zbar', col, row, lonlat=False)
    ts_zbar_PCLSM_DA.name = 'zbar analysis (PCLSM)'
    ts_zbar_PCLSM_OL = daily_PCLSM_OL.read_ts('zbar', col, row, lonlat=False)
    ts_zbar_PCLSM_OL.name = 'zbar open loop (PCLSM)'
    ts_tp1_PCLSM_DA = daily_PCLSM_DA.read_ts('tp1', col, row, lonlat=False)
    ts_tp1_PCLSM_DA.name = 'tp1 analysis (PCLSM)'
    ts_tp1_PCLSM_OL = daily_PCLSM_OL.read_ts('tp1', col, row, lonlat=False)
    ts_tp1_PCLSM_OL.name = 'tp1 open loop (PCLSM)'
    ts_sfmc_PCLSM_DA = daily_PCLSM_DA.read_ts('sfmc', col, row, lonlat=False)
    ts_sfmc_PCLSM_DA.name = 'sfmc analysis (PCLSM)'
    ts_sfmc_PCLSM_OL = daily_PCLSM_OL.read_ts('sfmc', col, row, lonlat=False)
    ts_sfmc_PCLSM_OL.name = 'sfmc open loop (PCLSM)'

    # ObsFcstAna PCLSM
    ts_obs_1 = ObsFcstAna_PCLSM_OL.read_ts('obs_obs', col, row, species=1, lonlat=False)
    ts_obs_1.name = 'Tb obs (species=1)'
    ts_obs_2 = ObsFcstAna_PCLSM_OL.read_ts('obs_obs', col, row, species=2, lonlat=False)
    ts_obs_2.name = 'Tb obs (species=2)'
    ts_obs_3 = ObsFcstAna_PCLSM_OL.read_ts('obs_obs', col, row, species=3, lonlat=False)
    ts_obs_3.name = 'Tb obs (species=3)'
    ts_obs_4 = ObsFcstAna_PCLSM_OL.read_ts('obs_obs', col, row, species=4, lonlat=False)
    ts_obs_4.name = 'Tb obs (species=4)'

    ts_obs_resc_PCLSM_1 = ObsFcstAna_PCLSM_DA.read_ts('obs_obs', col, row, species=1, lonlat=False)
    ts_obs_resc_PCLSM_1.name = 'Tb obs resc. PCLSM (species=1)'
    ts_obs_resc_PCLSM_2 = ObsFcstAna_PCLSM_DA.read_ts('obs_obs', col, row, species=2, lonlat=False)
    ts_obs_resc_PCLSM_2.name = 'Tb obs resc. PCLSM (species=2)'
    ts_obs_resc_PCLSM_3 = ObsFcstAna_PCLSM_DA.read_ts('obs_obs', col, row, species=3, lonlat=False)
    ts_obs_resc_PCLSM_3.name = 'Tb obs resc. PCLSM (species=3)'
    ts_obs_resc_PCLSM_4 = ObsFcstAna_PCLSM_DA.read_ts('obs_obs', col, row, species=4, lonlat=False)
    ts_obs_resc_PCLSM_4.name = 'Tb obs resc. PCLSM (species=4)'

    ts_fcst_PCLSM_OL_1 = ObsFcstAna_PCLSM_OL.read_ts('obs_fcst', col, row, species=1, lonlat=False)
    ts_fcst_PCLSM_OL_1.name = 'Tb fcst PCLSM OL (species=1)'
    ts_fcst_PCLSM_OL_2 = ObsFcstAna_PCLSM_OL.read_ts('obs_fcst', col, row, species=2, lonlat=False)
    ts_fcst_PCLSM_OL_2.name = 'Tb fcst PCLSM OL (species=2)'
    ts_fcst_PCLSM_OL_3 = ObsFcstAna_PCLSM_OL.read_ts('obs_fcst', col, row, species=3, lonlat=False)
    ts_fcst_PCLSM_OL_3.name = 'Tb fcst PCLSM OL (species=3)'
    ts_fcst_PCLSM_OL_4 = ObsFcstAna_PCLSM_OL.read_ts('obs_fcst', col, row, species=4, lonlat=False)
    ts_fcst_PCLSM_OL_4.name = 'Tb fcst PCLSM OL (species=4)'

    ts_ana_PCLSM_1 = ObsFcstAna_PCLSM_DA.read_ts('obs_ana', col, row, species=1, lonlat=False)
    ts_ana_PCLSM_1.name = 'Tb ana PCLSM (species=1)'
    ts_ana_PCLSM_2 = ObsFcstAna_PCLSM_DA.read_ts('obs_ana', col, row, species=2, lonlat=False)
    ts_ana_PCLSM_2.name = 'Tb ana PCLSM (species=2)'
    ts_ana_PCLSM_3 = ObsFcstAna_PCLSM_DA.read_ts('obs_ana', col, row, species=3, lonlat=False)
    ts_ana_PCLSM_3.name = 'Tb ana PCLSM (species=3)'
    ts_ana_PCLSM_4 = ObsFcstAna_PCLSM_DA.read_ts('obs_ana', col, row, species=4, lonlat=False)
    ts_ana_PCLSM_4.name = 'Tb ana PCLSM (species=4)'

    ts_fcst_PCLSM_DA_1 = ObsFcstAna_PCLSM_DA.read_ts('obs_fcst', col, row, species=1, lonlat=False)
    ts_fcst_PCLSM_DA_1.name = 'Tb fcst PCLSM DA (species=1)'
    ts_fcst_PCLSM_DA_2 = ObsFcstAna_PCLSM_DA.read_ts('obs_fcst', col, row, species=2, lonlat=False)
    ts_fcst_PCLSM_DA_2.name = 'Tb fcst PCLSM DA (species=2)'
    ts_fcst_PCLSM_DA_3 = ObsFcstAna_PCLSM_DA.read_ts('obs_fcst', col, row, species=3, lonlat=False)
    ts_fcst_PCLSM_DA_3.name = 'Tb fcst PCLSM DA (species=3)'
    ts_fcst_PCLSM_DA_4 = ObsFcstAna_PCLSM_DA.read_ts('obs_fcst', col, row, species=4, lonlat=False)
    ts_fcst_PCLSM_DA_4.name = 'Tb fcst PCLSM DA (species=4)'

    # ts_innov_1 = ts_obs_1 - ts_fcst_1
    # ts_innov_1[ts_assim_1!=-1] = np.nan
    # ts_innov_1.name = 'innov (species=1,PCLSM)'
    # ts_innov_2 = ts_obs_2 - ts_fcst_2
    # ts_innov_2[ts_assim_2!=-1] = np.nan
    # ts_innov_2.name = 'innov (species=2,PCLSM)'
    # ts_innov_3 = ts_obs_3 - ts_fcst_3
    # ts_innov_3[ts_assim_3!=-1] = np.nan
    # ts_innov_3.name = 'innov (species=3,PCLSM)'
    # ts_innov_4 = ts_obs_4 - ts_fcst_4
    # ts_innov_4[ts_assim_4!=-1] = np.nan
    # ts_innov_4.name = 'innov (species=4,PCLSM)'

    # incr PCLSM
    ts_incr_catdef_PCLSM = incr_PCLSM.read_ts('catdef', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_catdef_PCLSM.name = 'catdef incr (PCLSM)'
    ts_incr_srfexc_PCLSM = incr_PCLSM.read_ts('srfexc', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_srfexc_PCLSM.name = 'srfexc incr (PCLSM)'
    ts_incr_rzexc_PCLSM = incr_PCLSM.read_ts('rzexc', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_rzexc_PCLSM.name = 'rzexc incr (PCLSM)'
    ts_ar1_PCLSM_DA = daily_PCLSM_DA.read_ts('ar1', col, row, lonlat=False).replace(0, np.nan)
    ts_ar1_PCLSM_DA.name = 'ar1 analysis (PCLSM)'

    df_incr = pd.concat(
        (ts_incr_catdef_PCLSM, ts_incr_srfexc_PCLSM, ts_incr_rzexc_PCLSM, ts_ar1_PCLSM_DA, ts_catdef_PCLSM_DA), axis=1)
    df_incr['catdef analysis (PCLSM)'].interpolate(method='pad', limit=7)
    df_incr['ar1 analysis (PCLSM)'].interpolate(method='pad', limit=7)
    df_incr['catdef0 analysis (PCLSM)'] = df_incr['catdef analysis (PCLSM)'] - df_incr['catdef incr (PCLSM)']
    catdef0 = df_incr['catdef0 analysis (PCLSM)']
    df_incr['ar0 analysis (PCLSM)'] = (1.0 + ars1[row, col] * catdef0) / (
            1.0 + ars2[row, col] * catdef0 + ars3[row, col] * catdef0 ** 2.0)
    df_incr['zbar0'] = -1.0 * (np.sqrt(0.000001 + df_incr['catdef0 analysis (PCLSM)'] / bf1[row, col]) - bf2[row, col])
    df_incr['zbar'] = -1.0 * (np.sqrt(0.000001 + df_incr['catdef analysis (PCLSM)'] / bf1[row, col]) - bf2[row, col])
    df_incr['ar1 incr (PCLSM)'] = -(df_incr['zbar'] - df_incr['zbar0']) * 1000 * (0.5 * (
            df_incr['ar0 analysis (PCLSM)'] + df_incr[
        'ar1 analysis (PCLSM)']))  # incr surface water storage but expressed as deficit change, so '-'
    df_incr['total incr (PCLSM)'] = df_incr['ar1 incr (PCLSM)'] + df_incr['catdef incr (PCLSM)'] + df_incr[
        'rzexc incr (PCLSM)'] + df_incr['srfexc incr (PCLSM)']
    ts_incr_total_PCLSM = df_incr['total incr (PCLSM)']
    incr_catdef_mean = df_incr['catdef incr (PCLSM)'].mean(skipna=True)
    incr_total_mean = df_incr['total incr (PCLSM)'].mean(skipna=True)
    # ensstd PCLSM
    # ts_ens_catdef = io_ens.read_ts('catdef', col, row, lonlat=False).replace(0,np.nan)
    # ts_ens_catdef.name = 'catdef'
    # ts_ens_srfexc = io_ens.read_ts('srfexc', col, row, lonlat=False).replace(0,np.nan)
    # ts_ens_srfexc.name = 'srfexc'

    ###################### CLSM #######################
    # daily CLSM
    ts_catdef_CLSM_DA = daily_CLSM_DA.read_ts('catdef', col, row, lonlat=False)
    ts_catdef_CLSM_DA.name = 'catdef analysis (CLSM)'
    ts_catdef_CLSM_OL = daily_CLSM_OL.read_ts('catdef', col, row, lonlat=False)
    ts_catdef_CLSM_OL.name = 'catdef open loop (CLSM)'
    ts_zbar_CLSM_DA = daily_CLSM_DA.read_ts('zbar', col, row, lonlat=False)
    ts_zbar_CLSM_DA.name = 'zbar analysis (CLSM)'
    ts_zbar_CLSM_OL = daily_CLSM_OL.read_ts('zbar', col, row, lonlat=False)
    ts_zbar_CLSM_OL.name = 'zbar open loop (CLSM)'
    ts_tp1_CLSM_DA = daily_CLSM_DA.read_ts('tp1', col, row, lonlat=False)
    ts_tp1_CLSM_DA.name = 'tp1 analysis (CLSM)'
    ts_tp1_CLSM_OL = daily_CLSM_OL.read_ts('tp1', col, row, lonlat=False)
    ts_tp1_CLSM_OL.name = 'tp1 open loop (CLSM)'
    ts_sfmc_CLSM_DA = daily_CLSM_DA.read_ts('sfmc', col, row, lonlat=False)
    ts_sfmc_CLSM_DA.name = 'sfmc analysis (CLSM)'
    ts_sfmc_CLSM_OL = daily_CLSM_OL.read_ts('sfmc', col, row, lonlat=False)
    ts_sfmc_CLSM_OL.name = 'sfmc open loop (CLSM)'

    # ObsFcstAna CLSM
    ts_obs_resc_CLSM_1 = ObsFcstAna_CLSM_DA.read_ts('obs_obs', col, row, species=1, lonlat=False)
    ts_obs_resc_CLSM_1.name = 'Tb obs resc. CLSM (species=1)'
    ts_obs_resc_CLSM_2 = ObsFcstAna_CLSM_DA.read_ts('obs_obs', col, row, species=2, lonlat=False)
    ts_obs_resc_CLSM_2.name = 'Tb obs resc. CLSM (species=2)'
    ts_obs_resc_CLSM_3 = ObsFcstAna_CLSM_DA.read_ts('obs_obs', col, row, species=3, lonlat=False)
    ts_obs_resc_CLSM_3.name = 'Tb obs resc. CLSM (species=3)'
    ts_obs_resc_CLSM_4 = ObsFcstAna_CLSM_DA.read_ts('obs_obs', col, row, species=4, lonlat=False)
    ts_obs_resc_CLSM_4.name = 'Tb obs resc. CLSM (species=4)'

    ts_fcst_CLSM_OL_1 = ObsFcstAna_CLSM_OL.read_ts('obs_fcst', col, row, species=1, lonlat=False)
    ts_fcst_CLSM_OL_1.name = 'Tb fcst CLSM OL (species=1)'
    ts_fcst_CLSM_OL_2 = ObsFcstAna_CLSM_OL.read_ts('obs_fcst', col, row, species=2, lonlat=False)
    ts_fcst_CLSM_OL_2.name = 'Tb fcst CLSM OL (species=2)'
    ts_fcst_CLSM_OL_3 = ObsFcstAna_CLSM_OL.read_ts('obs_fcst', col, row, species=3, lonlat=False)
    ts_fcst_CLSM_OL_3.name = 'Tb fcst CLSM OL (species=3)'
    ts_fcst_CLSM_OL_4 = ObsFcstAna_CLSM_OL.read_ts('obs_fcst', col, row, species=4, lonlat=False)
    ts_fcst_CLSM_OL_4.name = 'Tb fcst CLSM OL (species=4)'

    ts_ana_CLSM_1 = ObsFcstAna_CLSM_DA.read_ts('obs_ana', col, row, species=1, lonlat=False)
    ts_ana_CLSM_1.name = 'Tb ana CLSM (species=1)'
    ts_ana_CLSM_2 = ObsFcstAna_CLSM_DA.read_ts('obs_ana', col, row, species=2, lonlat=False)
    ts_ana_CLSM_2.name = 'Tb ana CLSM (species=2)'
    ts_ana_CLSM_3 = ObsFcstAna_CLSM_DA.read_ts('obs_ana', col, row, species=3, lonlat=False)
    ts_ana_CLSM_3.name = 'Tb ana CLSM (species=3)'
    ts_ana_CLSM_4 = ObsFcstAna_CLSM_DA.read_ts('obs_ana', col, row, species=4, lonlat=False)
    ts_ana_CLSM_4.name = 'Tb ana CLSM (species=4)'

    ts_fcst_CLSM_DA_1 = ObsFcstAna_CLSM_DA.read_ts('obs_fcst', col, row, species=1, lonlat=False)
    ts_fcst_CLSM_DA_1.name = 'Tb fcst CLSM DA (species=1)'
    ts_fcst_CLSM_DA_2 = ObsFcstAna_CLSM_DA.read_ts('obs_fcst', col, row, species=2, lonlat=False)
    ts_fcst_CLSM_DA_2.name = 'Tb fcst CLSM DA (species=2)'
    ts_fcst_CLSM_DA_3 = ObsFcstAna_CLSM_DA.read_ts('obs_fcst', col, row, species=3, lonlat=False)
    ts_fcst_CLSM_DA_3.name = 'Tb fcst CLSM DA (species=3)'
    ts_fcst_CLSM_DA_4 = ObsFcstAna_CLSM_DA.read_ts('obs_fcst', col, row, species=4, lonlat=False)
    ts_fcst_CLSM_DA_4.name = 'Tb fcst CLSM DA (species=4)'

    # incr CLSM
    ts_incr_catdef_CLSM = incr_CLSM.read_ts('catdef', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_catdef_CLSM.name = 'catdef incr (CLSM)'
    ts_incr_srfexc_CLSM = incr_CLSM.read_ts('srfexc', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_srfexc_CLSM.name = 'srfexc incr (CLSM)'
    ts_incr_rzexc_CLSM = incr_CLSM.read_ts('rzexc', col, row, lonlat=False).replace(0, np.nan)
    ts_incr_rzexc_CLSM.name = 'rzexc incr (CLSM)'
    ts_ar1_CLSM_DA = daily_CLSM_DA.read_ts('ar1', col, row, lonlat=False).replace(0, np.nan)
    ts_ar1_CLSM_DA.name = 'ar1 analysis (CLSM)'

    df_incr = pd.concat(
        (ts_incr_catdef_CLSM, ts_incr_srfexc_CLSM, ts_incr_rzexc_CLSM, ts_ar1_CLSM_DA, ts_catdef_CLSM_DA), axis=1)
    df_incr['catdef analysis (CLSM)'].interpolate(method='pad', limit=7)
    df_incr['total incr (CLSM)'] = df_incr['catdef incr (CLSM)'] + df_incr['rzexc incr (CLSM)'] + df_incr[
        'srfexc incr (CLSM)']
    ts_incr_total_CLSM = df_incr['total incr (CLSM)']
    incr_catdef_mean = df_incr['catdef incr (CLSM)'].mean(skipna=True)
    incr_total_mean = df_incr['total incr (CLSM)'].mean(skipna=True)

    incr_total_std_CLSM = df_incr['total incr (CLSM)'].std(skipna=True)
    incr_total_std_PCLSM = ts_incr_total_PCLSM.std(skipna=True)

    df = pd.concat((ts_obs_1, ts_obs_2, ts_obs_3, ts_obs_4,
                    ts_fcst_CLSM_OL_1, ts_fcst_CLSM_OL_2, ts_fcst_CLSM_OL_3, ts_fcst_CLSM_OL_4,
                    ts_fcst_PCLSM_OL_1, ts_fcst_PCLSM_OL_2, ts_fcst_PCLSM_OL_3, ts_fcst_PCLSM_OL_4,
                    ts_fcst_CLSM_DA_1, ts_fcst_CLSM_DA_2, ts_fcst_CLSM_DA_3, ts_fcst_CLSM_DA_4,
                    ts_fcst_PCLSM_DA_1, ts_fcst_PCLSM_DA_2, ts_fcst_PCLSM_DA_3, ts_fcst_PCLSM_DA_4,
                    ts_obs_resc_CLSM_1, ts_obs_resc_CLSM_2, ts_obs_resc_CLSM_3, ts_obs_resc_CLSM_4,
                    ts_obs_resc_PCLSM_1, ts_obs_resc_PCLSM_2, ts_obs_resc_PCLSM_3, ts_obs_resc_PCLSM_4,
                    ts_incr_total_CLSM,
                    ts_incr_total_PCLSM
                    ),
                   axis=1)

    ts_tp1_CLSM_OL[(ts_tp1_CLSM_OL < 273.15) & (ts_tp1_PCLSM_OL < 273.15)] = np.nan
    ts_tp1_PCLSM_OL[(ts_tp1_CLSM_OL < 273.15) & (ts_tp1_PCLSM_OL < 273.15)] = np.nan
    ts_sfmc_CLSM_OL[ts_tp1_PCLSM_OL < 273.15] = np.nan
    ts_sfmc_CLSM_DA[ts_tp1_PCLSM_DA < 273.15] = np.nan
    ts_sfmc_PCLSM_OL[ts_tp1_PCLSM_OL < 273.15] = np.nan
    ts_sfmc_PCLSM_DA[ts_tp1_PCLSM_DA < 273.15] = np.nan
    ts_zbar_CLSM_OL[ts_tp1_PCLSM_OL < 273.15] = np.nan
    ts_zbar_CLSM_DA[ts_tp1_PCLSM_DA < 273.15] = np.nan
    ts_zbar_PCLSM_OL[ts_tp1_PCLSM_OL < 273.15] = np.nan
    ts_zbar_PCLSM_DA[ts_tp1_PCLSM_DA < 273.15] = np.nan
    in_situ_data = wtd_obs[site_name]
    datelim1 = ts_sfmc_CLSM_OL.index.min(axis=0)
    datelim2 = ts_sfmc_CLSM_OL.index.max(axis=0)
    in_situ_data = in_situ_data[(in_situ_data.index > datelim1) & (in_situ_data.index < datelim2)]
    df1 = pd.concat((in_situ_data, ts_sfmc_CLSM_OL, ts_sfmc_CLSM_DA, ts_sfmc_PCLSM_OL, ts_sfmc_PCLSM_DA,
                     ts_tp1_CLSM_OL, ts_tp1_CLSM_DA, ts_tp1_PCLSM_OL, ts_tp1_PCLSM_DA,
                     ts_zbar_CLSM_OL, ts_zbar_CLSM_DA, ts_zbar_PCLSM_OL, ts_zbar_PCLSM_DA),
                    axis=1)
    df1.loc[df1['tp1 open loop (PCLSM)'] < 273.15, site_name] = np.nan

    df2 = pd.concat((ts_tp1_CLSM_OL, ts_tp1_PCLSM_OL, ts_fcst_PCLSM_OL_2), axis=1)
    df2['tp1 open loop (CLSM)'] = df2['tp1 open loop (CLSM)'].interpolate(method='pad', limit=6)
    df2['tp1 open loop (PCLSM)'] = df2['tp1 open loop (PCLSM)'].interpolate(method='pad', limit=6)
    # df = pd.concat((ts_obs_1, ts_obs_2, ts_obs_3, ts_obs_4, ts_fcst_1, ts_fcst_2, ts_fcst_3, ts_fcst_4, ts_incr_catdef),axis=1).dropna()
    col_obs = (0.0, 0.0, 0.0)
    col_CLSM = (0.42745098, 0.71372549, 1.)
    col_PCLSM = (0.14117647, 1., 0.14117647)
    col_CLSM_DA = (0, 50. / 256., 95. / 256.)
    col_PCLSM_DA = (11. / 256., 102. / 256., 35. / 256.)
    col_OL = (0, 85. / 255., 158. / 255.)
    col_red = (243. / 255., 94. / 255., 0. / 255.)
    col_obs = (243. / 255., 94. / 255., 0. / 255.)

    datelim1 = '2013-01-01'
    datelim2 = '2016-01-01'
    mask = (df.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask1 = (df1.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df1.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask2 = (df2.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df2.index <= pd.Timestamp(datelim2 + ' 00:00:00'))

    def get_axis_limits(ax, scale=1.05):
        return ax.get_xlim()[1] * scale, ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * (scale - 1.)

    x_anno = pd.Timestamp('2012-10-01 00:00:00')
    x_anno2 = pd.Timestamp('2014-09-15 00:00:00')
    x_anno3 = pd.Timestamp('2015-03-10 00:00:00')

    plt.figure(figsize=(11.3, 4.7))
    fontsize = 12.0

    ax1 = plt.subplot(321)
    ax1.plot(df1.loc[mask1, 'sfmc open loop (CLSM)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    ax1.plot(df1.loc[mask1, 'sfmc analysis (CLSM)'], linestyle='dotted', color=col_CLSM_DA, linewidth=1.5)
    ax1.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylabel(r'$SM\enspace(m^{3}m^{-3})$')
    plt.ylim([0.4, 0.85])
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax1.legend(['OL', 'DA'], ncol=2)
    ax1.set_title('CLSM')
    ax1.annotate('(a)', xy=(x_anno, get_axis_limits(ax1)[1]), annotation_clip=False)
    # ax1.legend()

    ax2 = plt.subplot(322)
    ax2.plot(df1.loc[mask1, 'sfmc open loop (PCLSM)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    ax2.plot(df1.loc[mask1, 'sfmc analysis (PCLSM)'], linestyle='dotted', color=col_PCLSM_DA, linewidth=1.5)
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([0.4, 0.85])
    ax2.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax2.axes.xaxis.set_ticklabels([])
    ax2.legend(['OL', 'DA'], ncol=2)
    ax2.set_title('PEATCLSM')
    ax2.annotate('(b)', xy=(x_anno, get_axis_limits(ax2)[1]), annotation_clip=False)

    ax3 = plt.subplot(323)
    ax3.plot(df1.loc[mask1, site_name], linestyle='-', color=col_obs)
    ax3.plot(df1.loc[mask1, 'zbar open loop (CLSM)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    ax3.plot(df1.loc[mask1, 'zbar analysis (CLSM)'], linestyle='dotted', color=col_CLSM_DA, linewidth=1.5)
    ax3.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylabel(r'$\overline{z}_{WT}\!\enspace(m)$')
    plt.ylim([-3, 1.3])
    ax3.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax3.axes.xaxis.set_ticklabels([])
    ax3.legend(['in situ', 'OL', 'DA'], ncol=3)
    a = np.ma.masked_invalid(df1.loc[mask1, site_name])
    b = np.ma.masked_invalid(df1.loc[mask1, 'zbar open loop (CLSM)'])
    c = np.ma.masked_invalid(df1.loc[mask1, 'zbar analysis (CLSM)'])
    msk = (~a.mask & ~b.mask)
    dubRMSD = np.sqrt(np.mean((a[msk] - c[msk]) ** 2.)) - np.sqrt(np.mean((a[msk] - b[msk]) ** 2.))
    dR = np.ma.corrcoef(a[msk], c[msk])[0, 1] - np.ma.corrcoef(a[msk], b[msk])[0, 1]
    mstats1 = '$\Delta$ubRMSD = %.2f, $\Delta$R = %.2f' % (dubRMSD, dR)
    ax3.annotate(mstats1, xy=(x_anno2, get_axis_limits(ax3)[1]), annotation_clip=False)
    # ax3.set_title('%f')
    ax3.annotate('(c)', xy=(x_anno, get_axis_limits(ax3)[1]), annotation_clip=False)

    ax4 = plt.subplot(324)
    ax4.plot(df1.loc[mask1, site_name], linestyle='-', color=col_obs)
    ax4.plot(df1.loc[mask1, 'zbar open loop (PCLSM)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    ax4.plot(df1.loc[mask1, 'zbar analysis (PCLSM)'], linestyle='dotted', color=col_PCLSM_DA, linewidth=1.5)
    ax4.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([-0.45, 0.02])
    ax4.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax4.axes.xaxis.set_ticklabels([])
    ax4.legend(['in situ', 'OL', 'DA'], ncol=3)
    a = np.ma.masked_invalid(df1.loc[mask1, site_name])
    b = np.ma.masked_invalid(df1.loc[mask1, 'zbar open loop (PCLSM)'])
    c = np.ma.masked_invalid(df1.loc[mask1, 'zbar analysis (PCLSM)'])
    msk = (~a.mask & ~b.mask)
    dubRMSD = np.sqrt(np.mean((a[msk] - c[msk]) ** 2.)) - np.sqrt(np.mean((a[msk] - b[msk]) ** 2.))
    dR = np.ma.corrcoef(a[msk], c[msk])[0, 1] - np.ma.corrcoef(a[msk], b[msk])[0, 1]
    mstats1 = '$\Delta$ubRMSD = %.2f, $\Delta$R = %.2f' % (dubRMSD, dR)
    ax4.annotate(mstats1, xy=(x_anno2, get_axis_limits(ax4)[1]), annotation_clip=False)
    ax4.annotate('(d)', xy=(x_anno, get_axis_limits(ax4)[1]), annotation_clip=False)

    condna = np.isnan(df['total incr (CLSM)'].values)
    condna[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    ts1 = -df.loc[(condna == False) & mask, 'total incr (CLSM)']
    ts2 = -df.loc[(condna == False) & mask, 'total incr (PCLSM)']
    mstats1 = 'std($\Delta$twot) = %.2f' % (np.nanstd(ts1))
    mstats2 = 'std($\Delta$twot) = %.2f' % (np.nanstd(ts2))
    # mstats2 = 'm = %.2f, s = %.2f' % (np.nanmean(ts2),np.nanstd(ts2))

    ax5 = plt.subplot(325)
    df['freeze'] = 0.
    ax5.plot(df['freeze'], color='grey')
    ax5.plot(-df.loc[(condna == False) & (mask), 'total incr (CLSM)'], linestyle='dotted', color=col_CLSM_DA,
             linewidth=1.5)
    ax5.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylabel(r'$\Delta wtot\enspace(mm)$')
    ax5.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    # ax1.set_xticks(['01-01-2013', 2014, 2015, 2016])
    # ax1.axes.xaxis.set_ticklabels([2013, 2014, 2015, 2016])
    plt.ylim([-30, 55])
    # plt.grid(which='major',axis='y')
    # ax1.legend(['in situ','OL','DA'],ncol=3)
    ax5.annotate('(e)', xy=(x_anno, get_axis_limits(ax5)[1]), annotation_clip=False)
    ax5.annotate(mstats1, xy=(x_anno3, get_axis_limits(ax5)[1]), annotation_clip=False)

    ax6 = plt.subplot(326)
    condna = np.isnan(df['total incr (PCLSM)'].values)
    condna[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    df['freeze'] = 0.
    ax6.plot(df['freeze'], color='grey')
    ax6.plot(-df.loc[(condna == False) & (mask), 'total incr (PCLSM)'], linestyle='dotted', color=col_PCLSM_DA,
             linewidth=1.5)
    ax6.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    ax6.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    plt.ylim([-30, 55])
    # plt.grid(which='major',axis='y')
    # ax1.legend(['in situ','OL','DA'],ncol=3)
    ax6.annotate('(f)', xy=(x_anno, get_axis_limits(ax6)[1]), annotation_clip=False)
    ax6.annotate(mstats2, xy=(x_anno3, get_axis_limits(ax5)[1]), annotation_clip=False)

    plt.tight_layout()
    fname = 'sfmc_zbar_resc_col_%s_row_%s_zoom' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    fontsize = 12

    ax1 = plt.subplot(211)
    condna1 = np.isnan(df['Tb obs (species=1)'].values)
    condna2 = np.isnan(df['Tb obs (species=2)'].values)
    condna3 = np.isnan(df['Tb obs (species=3)'].values)
    condna4 = np.isnan(df['Tb obs (species=4)'].values)
    condna1[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb obs (species=1)'], linestyle='-', color=col_obs, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst CLSM OL (species=1)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst PCLSM OL (species=1)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    mark1 = ax1.plot(df.loc[mask, 'Tb obs (species=1)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax1.plot(df.loc[mask, 'Tb obs (species=2)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax1.plot(df.loc[mask, 'Tb fcst CLSM OL (species=1)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax1.plot(df.loc[mask, 'Tb fcst CLSM OL (species=2)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax1.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=1)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax1.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=2)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax1.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([198.,247.])
    plt.ylabel('Tb H-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax1.legend(ncol=3)
    ax1.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.annotate('(a)', xy=(x_anno, get_axis_limits(ax1)[1]), annotation_clip=False)

    ax2 = plt.subplot(212)
    mark1 = ax2.plot(df.loc[mask, 'Tb obs (species=3)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax2.plot(df.loc[mask, 'Tb obs (species=4)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=3)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=4)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=3)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=4)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([216.,263.])
    plt.ylabel('Tb V-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax2.annotate('(b)', xy=(x_anno, get_axis_limits(ax2)[1]), annotation_clip=False)
    ax2.legend(ncol=3)
    ax2.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)

    plt.tight_layout()
    fname = 'Tb_col_%s_row_%s_zoom' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    fig = plt.figure(figsize=(7, 5.8))
    fontsize = 12

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], hspace=0)
    ax0 = fig.add_subplot(gs00[0])
    condna1 = np.isnan(df['Tb obs (species=1)'].values)
    condna2 = np.isnan(df['Tb obs (species=2)'].values)
    condna3 = np.isnan(df['Tb obs (species=3)'].values)
    condna4 = np.isnan(df['Tb obs (species=4)'].values)
    condna1[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb obs (species=1)'], linestyle='-', color=col_obs, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst CLSM OL (species=1)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst PCLSM OL (species=1)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    mark1 = ax0.plot(df.loc[mask, 'Tb obs (species=1)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax0.plot(df.loc[mask, 'Tb obs (species=2)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=1)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=2)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=1)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=2)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax0.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([198.,247.])
    plt.ylabel('Tb H-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax0.legend(ncol=3)
    ax0.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.annotate('(a)', xy=(x_anno, get_axis_limits(ax1)[1]), annotation_clip=False)

    ax1b = fig.add_subplot(gs00[1], sharex=ax0)
    df['Tb misfit delta (species=1)'] = np.abs(df['Tb fcst PCLSM OL (species=1)'] - df['Tb obs (species=1)']) - np.abs(
        df['Tb fcst CLSM OL (species=1)'] - df['Tb obs (species=1)'])
    df['Tb misfit delta (species=2)'] = np.abs(df['Tb fcst PCLSM OL (species=2)'] - df['Tb obs (species=2)']) - np.abs(
        df['Tb fcst CLSM OL (species=2)'] - df['Tb obs (species=2)'])
    mark1 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=1)'], marker='o', markersize=3., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=2)'], marker='o', markersize=3., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-5., 5.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    ax1b.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax1b.axes.xaxis.set_ticklabels([])

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1], hspace=0)
    ax2 = fig.add_subplot(gs01[0])
    mark1 = ax2.plot(df.loc[mask, 'Tb obs (species=3)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax2.plot(df.loc[mask, 'Tb obs (species=4)'], marker='o', markersize=4.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=3)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=4)'], marker='o', markersize=3.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=3)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=4)'], marker='o', markersize=2.2, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([216.,263.])
    plt.ylabel('Tb V-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax2.annotate('(b)', xy=(x_anno, get_axis_limits(ax2)[1]), annotation_clip=False)
    ax2.legend(ncol=3, loc='lower left')
    ax2.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax2.axes.xaxis.set_ticklabels([])

    ax2b = fig.add_subplot(gs01[1])
    df['Tb misfit delta (species=3)'] = np.abs(df['Tb fcst PCLSM OL (species=3)'] - df['Tb obs (species=3)']) - np.abs(
        df['Tb fcst CLSM OL (species=3)'] - df['Tb obs (species=3)'])
    df['Tb misfit delta (species=4)'] = np.abs(df['Tb fcst PCLSM OL (species=4)'] - df['Tb obs (species=4)']) - np.abs(
        df['Tb fcst CLSM OL (species=4)'] - df['Tb obs (species=4)'])
    mark1 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=3)'], marker='o', markersize=3., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=4)'], marker='o', markersize=3., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-5., 5.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax2b.legend([mark1],ncol=1, loc='lower left')
    ax2b.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)

    plt.tight_layout()
    fname = 'Tb_col_%s_row_%s_zoom_delta' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    datelim1 = '2010-01-01'
    datelim2 = '2020-01-01'
    mask = (df.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask1 = (df1.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df1.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask2 = (df2.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df2.index <= pd.Timestamp(datelim2 + ' 00:00:00'))

    x_anno = pd.Timestamp('2009-10-01 00:00:00')
    fig = plt.figure(figsize=(10, 5.8))
    fontsize = 12

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], hspace=0)
    ax0 = fig.add_subplot(gs00[0])
    condna1 = np.isnan(df['Tb obs (species=1)'].values)
    condna2 = np.isnan(df['Tb obs (species=2)'].values)
    condna3 = np.isnan(df['Tb obs (species=3)'].values)
    condna4 = np.isnan(df['Tb obs (species=4)'].values)
    condna1[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb obs (species=1)'], linestyle='-', color=col_obs, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst CLSM OL (species=1)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst PCLSM OL (species=1)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    mark1 = ax0.plot(df.loc[mask, 'Tb obs (species=1)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax0.plot(df.loc[mask, 'Tb obs (species=2)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=1)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=2)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=1)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=2)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax0.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([198.,247.])
    plt.ylabel('Tb H-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax0.legend(ncol=3)
    # ax0.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.annotate('(a)', xy=(x_anno, get_axis_limits(ax0)[1]), annotation_clip=False)

    ax1b = fig.add_subplot(gs00[1], sharex=ax0)
    df['Tb misfit delta (species=1)'] = np.abs(df['Tb fcst PCLSM OL (species=1)'] - df['Tb obs (species=1)']) - np.abs(
        df['Tb fcst CLSM OL (species=1)'] - df['Tb obs (species=1)'])
    df['Tb misfit delta (species=2)'] = np.abs(df['Tb fcst PCLSM OL (species=2)'] - df['Tb obs (species=2)']) - np.abs(
        df['Tb fcst CLSM OL (species=2)'] - df['Tb obs (species=2)'])
    mark1 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=1)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=2)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    df['avg_12'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=1)']) + np.nanmean(df['Tb misfit delta (species=2)']))
    mark3 = ax1b.plot(df.loc[mask, 'avg_12'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-6., 6.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    # ax1b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax1b.axes.xaxis.set_ticklabels([])

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1], hspace=0)
    ax2 = fig.add_subplot(gs01[0])
    mark1 = ax2.plot(df.loc[mask, 'Tb obs (species=3)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax2.plot(df.loc[mask, 'Tb obs (species=4)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=3)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=4)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=3)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=4)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([216.,263.])
    plt.ylabel('Tb V-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax2.annotate('(b)', xy=(x_anno, get_axis_limits(ax2)[1]), annotation_clip=False)
    ax2.legend(ncol=3, loc='lower left')
    ax2.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax2.axes.xaxis.set_ticklabels([])

    ax2b = fig.add_subplot(gs01[1])
    df['Tb misfit delta (species=3)'] = np.abs(df['Tb fcst PCLSM OL (species=3)'] - df['Tb obs (species=3)']) - np.abs(
        df['Tb fcst CLSM OL (species=3)'] - df['Tb obs (species=3)'])
    df['Tb misfit delta (species=4)'] = np.abs(df['Tb fcst PCLSM OL (species=4)'] - df['Tb obs (species=4)']) - np.abs(
        df['Tb fcst CLSM OL (species=4)'] - df['Tb obs (species=4)'])
    mark1 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=3)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=4)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    df['avg_34'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=3)']) + np.nanmean(df['Tb misfit delta (species=4)']))
    mark3 = ax2b.plot(df.loc[mask, 'avg_34'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    ax2b.set_xlim(pd.Timestamp(datelim1),
                  pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([-6., 6.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax2b.legend([mark1],ncol=1, loc='lower left')
    # ax2b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)

    plt.tight_layout()
    fname = 'Tb_col_%s_row_%s_full_delta' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    # O-F for within DA cycle

    datelim1 = '2010-01-01'
    datelim2 = '2020-01-01'
    mask = (df.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask1 = (df1.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df1.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask2 = (df2.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df2.index <= pd.Timestamp(datelim2 + ' 00:00:00'))

    x_anno = pd.Timestamp('2009-10-01 00:00:00')
    fig = plt.figure(figsize=(10, 6.8))
    fontsize = 12

    condna1C = np.isnan(df['Tb obs resc. CLSM (species=1)'].values)
    condna2C = np.isnan(df['Tb obs resc. CLSM (species=2)'].values)
    condna3C = np.isnan(df['Tb obs resc. CLSM (species=3)'].values)
    condna4C = np.isnan(df['Tb obs resc. CLSM (species=4)'].values)
    condna1P = np.isnan(df['Tb obs resc. PCLSM (species=1)'].values)
    condna2P = np.isnan(df['Tb obs resc. PCLSM (species=2)'].values)
    condna3P = np.isnan(df['Tb obs resc. PCLSM (species=3)'].values)
    condna4P = np.isnan(df['Tb obs resc. PCLSM (species=4)'].values)
    condna1C[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2C[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3C[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4C[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna1P[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2P[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3P[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4P[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0], hspace=0)
    ax0 = fig.add_subplot(gs00[0])
    mark1 = ax0.plot(df.loc[mask, 'Tb obs resc. CLSM (species=1)'], marker='o', markersize=2.9, linewidth=0,
                     color=col_obs, label="SMOS")
    mark2 = ax0.plot(df.loc[mask, 'Tb obs resc. CLSM (species=2)'], marker='o', markersize=2.9, linewidth=0,
                     color=col_obs, label="_")
    mark3 = ax0.plot(df.loc[mask, 'Tb fcst CLSM DA (species=1)'], marker='o', markersize=1.8, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax0.plot(df.loc[mask, 'Tb fcst CLSM DA (species=2)'], marker='o', markersize=1.8, linewidth=0,
                     color=col_CLSM, label="_")
    ax0.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([175., 255.])
    plt.grid(True)
    plt.ylabel('Tb H-pol. (K)')
    ax0.legend(ncol=2, loc='lower left')
    ax0.axes.xaxis.set_ticklabels([])
    ax0.set_yticks([180, 200, 220, 240], minor=False)
    ax0.annotate('(a)', xy=(x_anno, get_axis_limits(ax0)[1]), annotation_clip=False)

    ax0b = fig.add_subplot(gs00[1], sharex=ax0)
    mark1 = ax0b.plot(df.loc[mask, 'Tb obs resc. PCLSM (species=1)'], marker='o', markersize=2.9, linewidth=0,
                      color=col_obs, label="SMOS")
    mark2 = ax0b.plot(df.loc[mask, 'Tb obs resc. PCLSM (species=2)'], marker='o', markersize=2.9, linewidth=0,
                      color=col_obs, label="_")
    mark3 = ax0b.plot(df.loc[mask, 'Tb fcst PCLSM DA (species=1)'], marker='o', markersize=1.8, linewidth=0,
                      color=col_PCLSM, label="PEATCLSM")
    mark4 = ax0b.plot(df.loc[mask, 'Tb fcst PCLSM DA (species=2)'], marker='o', markersize=1.8, linewidth=0,
                      color=col_PCLSM, label="_")
    ax0b.set_xlim(pd.Timestamp(datelim1),
                  pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([175., 255.])
    plt.grid(True)
    plt.ylabel('Tb H-pol. (K)')
    ax0b.legend(ncol=2, loc='lower left')
    ax0b.axes.xaxis.set_ticklabels([])
    ax0b.set_yticks([180, 200, 220, 240], minor=False)

    ax1b = fig.add_subplot(gs00[2], sharex=ax0)
    df['Tb misfit delta (species=1)'] = np.abs(
        df['Tb fcst PCLSM DA (species=1)'] - df['Tb obs resc. PCLSM (species=1)']) - np.abs(
        df['Tb fcst CLSM DA (species=1)'] - df['Tb obs resc. CLSM (species=1)'])
    df['Tb misfit delta (species=2)'] = np.abs(
        df['Tb fcst PCLSM DA (species=2)'] - df['Tb obs resc. PCLSM (species=2)']) - np.abs(
        df['Tb fcst CLSM DA (species=2)'] - df['Tb obs resc. CLSM (species=2)'])
    mark1 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=1)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=2)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    df['avg_12'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=1)']) + np.nanmean(df['Tb misfit delta (species=2)']))
    mark3 = ax1b.plot(df.loc[mask, 'avg_12'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-7., 7.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    # ax1b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax1b.axes.xaxis.set_ticklabels([])

    gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1], hspace=0)
    ax0 = fig.add_subplot(gs01[0])
    mark1 = ax0.plot(df.loc[mask, 'Tb obs resc. CLSM (species=3)'], marker='o', markersize=2.9, linewidth=0,
                     color=col_obs, label="SMOS")
    mark2 = ax0.plot(df.loc[mask, 'Tb obs resc. CLSM (species=4)'], marker='o', markersize=2.9, linewidth=0,
                     color=col_obs, label="_")
    mark3 = ax0.plot(df.loc[mask, 'Tb fcst CLSM DA (species=3)'], marker='o', markersize=1.8, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax0.plot(df.loc[mask, 'Tb fcst CLSM DA (species=4)'], marker='o', markersize=1.8, linewidth=0,
                     color=col_CLSM, label="_")
    ax0.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([215., 270.])
    plt.grid(True)
    plt.ylabel('Tb V-pol. (K)')
    ax0.legend(ncol=2, loc='lower left')
    ax0.axes.xaxis.set_ticklabels([])
    ax0.annotate('(b)', xy=(x_anno, get_axis_limits(ax0)[1]), annotation_clip=False)

    ax0b = fig.add_subplot(gs01[1])
    mark1 = ax0b.plot(df.loc[mask, 'Tb obs resc. PCLSM (species=3)'], marker='o', markersize=2.9, linewidth=0,
                      color=col_obs, label="SMOS")
    mark2 = ax0b.plot(df.loc[mask, 'Tb obs resc. PCLSM (species=4)'], marker='o', markersize=2.9, linewidth=0,
                      color=col_obs, label="_")
    mark3 = ax0b.plot(df.loc[mask, 'Tb fcst PCLSM DA (species=3)'], marker='o', markersize=1.8, linewidth=0,
                      color=col_PCLSM, label="PEATCLSM")
    mark4 = ax0b.plot(df.loc[mask, 'Tb fcst PCLSM DA (species=4)'], marker='o', markersize=1.8, linewidth=0,
                      color=col_PCLSM, label="_")
    ax0b.set_xlim(pd.Timestamp(datelim1),
                  pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([215., 270.])
    plt.grid(True)
    plt.ylabel('Tb V-pol. (K)')
    ax0b.legend(ncol=2, loc='lower left')
    ax0b.axes.xaxis.set_ticklabels([])

    ax1b = fig.add_subplot(gs01[2])
    df['Tb misfit delta (species=3)'] = np.abs(
        df['Tb fcst PCLSM DA (species=3)'] - df['Tb obs resc. PCLSM (species=3)']) - np.abs(
        df['Tb fcst CLSM DA (species=3)'] - df['Tb obs resc. CLSM (species=3)'])
    df['Tb misfit delta (species=4)'] = np.abs(
        df['Tb fcst PCLSM DA (species=4)'] - df['Tb obs resc. PCLSM (species=4)']) - np.abs(
        df['Tb fcst CLSM DA (species=4)'] - df['Tb obs resc. CLSM (species=4)'])
    mark1 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=3)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax1b.plot(df.loc[mask, 'Tb misfit delta (species=4)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    df['avg_12'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=3)']) + np.nanmean(df['Tb misfit delta (species=4)']))
    mark3 = ax1b.plot(df.loc[mask, 'avg_12'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=3)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=3)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-7., 7.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    # ax1b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    # ax0b.axes.xaxis.set_ticklabels([])
    ax1b.set_xlim(pd.Timestamp(datelim1),
                  pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)

    plt.tight_layout()
    fname = 'Tb_col_%s_row_%s_full_delta_DA' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    # Tb misfit
    Tb_misfit_CLSM_OL = df.loc[:, 'Tb fcst CLSM OL (species=2)'] - df.loc[:, 'Tb obs (species=2)']
    Tb_misfit_PCLSM_OL = df.loc[:, 'Tb fcst PCLSM OL (species=2)'] - df.loc[:, 'Tb obs (species=2)']
    print('Tb_misfit(CLSM OL): %0.2f' % (np.var(Tb_misfit_CLSM_OL)))
    print('Tb_misfit(PCLSM OL): %0.2f' % (np.var(Tb_misfit_PCLSM_OL)))

    a = np.ma.masked_invalid(df['Tb fcst PCLSM DA (species=1)'])
    b = np.ma.masked_invalid(df['Tb obs resc. PCLSM (species=1)'])
    c = np.ma.masked_invalid(df['Tb fcst CLSM DA (species=1)'])
    msk = (~a.mask & ~b.mask & ~c.mask)
    C1 = np.corrcoef(df.loc[mask & msk, 'Tb fcst CLSM DA (species=1)'],
                     df.loc[mask & msk, 'Tb obs resc. CLSM (species=1)'])
    P1 = np.corrcoef(df.loc[mask & msk, 'Tb fcst PCLSM DA (species=1)'],
                     df.loc[mask & msk, 'Tb obs resc. PCLSM (species=1)'])
    a = np.ma.masked_invalid(df['Tb fcst PCLSM DA (species=2)'])
    b = np.ma.masked_invalid(df['Tb obs resc. PCLSM (species=2)'])
    c = np.ma.masked_invalid(df['Tb fcst CLSM DA (species=2)'])
    msk = (~a.mask & ~b.mask & ~c.mask)
    C2 = np.corrcoef(df.loc[mask & msk, 'Tb fcst CLSM DA (species=2)'],
                     df.loc[mask & msk, 'Tb obs resc. CLSM (species=2)'])
    P2 = np.corrcoef(df.loc[mask & msk, 'Tb fcst PCLSM DA (species=2)'],
                     df.loc[mask & msk, 'Tb obs resc. PCLSM (species=2)'])
    a = np.ma.masked_invalid(df['Tb fcst PCLSM DA (species=3)'])
    b = np.ma.masked_invalid(df['Tb obs resc. PCLSM (species=3)'])
    c = np.ma.masked_invalid(df['Tb fcst CLSM DA (species=3)'])
    msk = (~a.mask & ~b.mask & ~c.mask)
    C3 = np.corrcoef(df.loc[mask & msk, 'Tb fcst CLSM DA (species=3)'],
                     df.loc[mask & msk, 'Tb obs resc. CLSM (species=3)'])
    P3 = np.corrcoef(df.loc[mask & msk, 'Tb fcst PCLSM DA (species=3)'],
                     df.loc[mask & msk, 'Tb obs resc. PCLSM (species=3)'])
    a = np.ma.masked_invalid(df['Tb fcst PCLSM DA (species=4)'])
    b = np.ma.masked_invalid(df['Tb obs resc. PCLSM (species=4)'])
    c = np.ma.masked_invalid(df['Tb fcst CLSM DA (species=4)'])
    msk = (~a.mask & ~b.mask & ~c.mask)
    C4 = np.corrcoef(df.loc[mask & msk, 'Tb fcst CLSM DA (species=4)'],
                     df.loc[mask & msk, 'Tb obs resc. CLSM (species=4)'])
    P4 = np.corrcoef(df.loc[mask & msk, 'Tb fcst PCLSM DA (species=4)'],
                     df.loc[mask & msk, 'Tb obs resc. PCLSM (species=4)'])
    # tp1

    datelim1 = '2010-01-01'
    datelim2 = '2020-01-01'
    mask = (df.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask1 = (df1.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df1.index <= pd.Timestamp(datelim2 + ' 00:00:00'))
    mask2 = (df2.index > pd.Timestamp(datelim1 + ' 00:00:00')) & (df2.index <= pd.Timestamp(datelim2 + ' 00:00:00'))

    x_anno = pd.Timestamp('2009-10-01 00:00:00')
    fig = plt.figure(figsize=(10, 5.8))
    fontsize = 12

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], hspace=0)
    ax0 = fig.add_subplot(gs00[0])
    condna1 = np.isnan(df['Tb obs (species=1)'].values)
    condna2 = np.isnan(df['Tb obs (species=2)'].values)
    condna3 = np.isnan(df['Tb obs (species=3)'].values)
    condna4 = np.isnan(df['Tb obs (species=4)'].values)
    condna1[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna2[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna3[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    condna4[(df.index.dayofyear > 340) | (df.index.dayofyear < 20)] = False
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb obs (species=1)'], linestyle='-', color=col_obs, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst CLSM OL (species=1)'], linestyle='-', color=col_CLSM, linewidth=1.5)
    # ax1.plot(df.loc[(condna==False) & (mask),'Tb fcst PCLSM OL (species=1)'], linestyle='-', color=col_PCLSM, linewidth=1.5)
    mark1 = ax0.plot(df.loc[mask, 'Tb obs (species=1)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax0.plot(df.loc[mask, 'Tb obs (species=2)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=1)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax0.plot(df.loc[mask, 'Tb fcst CLSM OL (species=2)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=1)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax0.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=2)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax0.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([198.,247.])
    plt.ylabel('Tb H-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax0.legend(ncol=3)
    # ax0.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.annotate('(a)', xy=(x_anno, get_axis_limits(ax0)[1]), annotation_clip=False)

    ax1b = fig.add_subplot(gs00[1], sharex=ax0)
    df2['delta tp1'] = df2['tp1 open loop (PCLSM)'] - df2['tp1 open loop (CLSM)']
    mark1 = ax1b.plot(df2.loc[mask2, 'delta tp1'], marker='o', markersize=2., linewidth=0, color='k')
    df2['avg_tp1'] = np.nanmean(df2['delta tp1'])
    mark3 = ax1b.plot(df2.loc[mask2, 'avg_tp1'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-1., 1.])
    plt.grid(True)
    plt.ylabel('$\Delta$tp1 (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    # ax1b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax1b.axes.xaxis.set_ticklabels([])

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1], hspace=0)
    ax2 = fig.add_subplot(gs01[0])
    mark1 = ax2.plot(df.loc[mask, 'Tb obs (species=3)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="SMOS")
    mark2 = ax2.plot(df.loc[mask, 'Tb obs (species=4)'], marker='o', markersize=3.5, linewidth=0, color=col_obs,
                     label="_")
    mark3 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=3)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="CLSM")
    mark4 = ax2.plot(df.loc[mask, 'Tb fcst CLSM OL (species=4)'], marker='o', markersize=2.5, linewidth=0,
                     color=col_CLSM, label="_")
    mark5 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=3)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="PEATCLSM")
    mark6 = ax2.plot(df.loc[mask, 'Tb fcst PCLSM OL (species=4)'], marker='o', markersize=1.5, linewidth=0,
                     color=col_PCLSM, label="_")
    # ax3.plot(df.loc[condna==False,'Tb fcst PCLSM DA (species='+cspec+')']-df.loc[:,'Tb obs resc. PCLSM (species='+cspec+')'],marker='o', markerfacecolor="None", linestyle='-', linewidth=0, color=col_PCLSM)
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    # plt.ylim([216.,263.])
    plt.ylabel('Tb V-pol. (K)')
    # ax1.legend([mark1,mark3,mark5],['SMOS','OL','DA'],ncol=3)
    ax2.annotate('(b)', xy=(x_anno, get_axis_limits(ax2)[1]), annotation_clip=False)
    ax2.legend(ncol=3, loc='lower left')
    ax2.set_xticks(
        [pd.Timestamp('2013-01-01 00:00:00'), pd.Timestamp('2014-01-01 00:00:00'), pd.Timestamp('2015-01-01 00:00:00'),
         pd.Timestamp('2016-01-01 00:00:00')], minor=False)
    ax2.axes.xaxis.set_ticklabels([])

    ax2b = fig.add_subplot(gs01[1])
    df['Tb misfit delta (species=3)'] = np.abs(df['Tb fcst PCLSM OL (species=3)'] - df['Tb obs (species=3)']) - np.abs(
        df['Tb fcst CLSM OL (species=3)'] - df['Tb obs (species=3)'])
    df['Tb misfit delta (species=4)'] = np.abs(df['Tb fcst PCLSM OL (species=4)'] - df['Tb obs (species=4)']) - np.abs(
        df['Tb fcst CLSM OL (species=4)'] - df['Tb obs (species=4)'])
    mark1 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=3)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    mark2 = ax2b.plot(df.loc[mask, 'Tb misfit delta (species=4)'], marker='o', markersize=2., linewidth=0, color='k',
                      label='PEATCLSM-CLSM')
    df['avg_34'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=3)']) + np.nanmean(df['Tb misfit delta (species=4)']))
    mark3 = ax2b.plot(df.loc[mask, 'avg_34'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    ax2b.set_xlim(pd.Timestamp(datelim1),
                  pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)
    plt.ylim([-6., 6.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) (K)')
    # ax2b.legend([mark1],ncol=1, loc='lower left')
    # ax2b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)

    plt.tight_layout()
    fname = 'Tb_col_%s_row_%s_full_tp1' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    x_anno = pd.Timestamp('2009-10-01 00:00:00')
    fig = plt.figure(figsize=(10, 5.8))
    fontsize = 12

    ax1 = plt.subplot(411)
    df['Tb misfit delta (species=1)'] = np.abs(df['Tb fcst PCLSM OL (species=1)'] - df['Tb obs (species=1)']) - np.abs(
        df['Tb fcst CLSM OL (species=1)'] - df['Tb obs (species=1)'])
    df['Tb misfit delta (species=2)'] = np.abs(df['Tb fcst PCLSM OL (species=2)'] - df['Tb obs (species=2)']) - np.abs(
        df['Tb fcst CLSM OL (species=2)'] - df['Tb obs (species=2)'])
    mark1 = ax1.plot(df.loc[mask, 'Tb misfit delta (species=1)'], marker='o', markersize=2., linewidth=0, color='k',
                     label='PEATCLSM-CLSM')
    mark2 = ax1.plot(df.loc[mask, 'Tb misfit delta (species=2)'], marker='o', markersize=2., linewidth=0, color='k',
                     label='PEATCLSM-CLSM')
    df['avg_12'] = 0.5 * (np.nanmean(df['Tb misfit delta (species=1)']) + np.nanmean(df['Tb misfit delta (species=2)']))
    mark3 = ax1.plot(df.loc[mask, 'avg_12'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    # mark1 = ax1b.plot(df.loc[mask,'Tb residual CLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_CLSM)
    # mark2 = ax1b.plot(df.loc[mask,'Tb residual PCLSM (species=1)'], marker='o', markersize=3. , linewidth=0, color=col_PCLSM)
    plt.ylim([-6., 6.])
    plt.grid(True)
    plt.ylabel('$\Delta$abs(O-F) H-pol. (K)')
    # ax1b.legend(mark1, 'PEATCLSM-CLSM', loc='lower left')
    # ax1b.set_xticks([pd.Timestamp('2013-01-01 00:00:00'),pd.Timestamp('2014-01-01 00:00:00'),pd.Timestamp('2015-01-01 00:00:00'),pd.Timestamp('2016-01-01 00:00:00')],minor=False)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)

    ax2 = plt.subplot(412)
    df1['delta tp1'] = df1['tp1 open loop (PCLSM)'] - df1['tp1 open loop (CLSM)']
    mark1 = ax2.plot(df1.loc[mask1, 'delta tp1'], marker='o', markersize=2., linewidth=0, color='k')
    df1['avg_tp1'] = np.nanmean(df1['delta tp1'])
    mark3 = ax2.plot(df1.loc[mask1, 'avg_tp1'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    plt.ylim([-1., 1.])
    plt.grid(True)
    plt.ylabel('$\Delta$tp1 (K)')
    ax2.axes.xaxis.set_ticklabels([])
    ax2.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)

    ax3 = plt.subplot(413)
    df1['delta sfmc'] = df1['sfmc open loop (PCLSM)'] - df1['sfmc open loop (CLSM)']
    mark1 = ax3.plot(df1.loc[mask1, 'delta sfmc'], marker='o', markersize=2., linewidth=0, color='k')
    df1['avg_sfmc'] = np.nanmean(df1['delta sfmc'])
    mark3 = ax3.plot(df1.loc[mask1, 'avg_sfmc'], marker='.', markersize=0., linewidth=1.0, color='r', label='avg')
    plt.ylim([-0.2, 0.2])
    plt.grid(True)
    plt.ylabel('$\Delta$sfmc (-)')
    ax3.axes.xaxis.set_ticklabels([])
    ax3.set_xlim(pd.Timestamp(datelim1),
                 pd.Timestamp(datelim2))  # , linewidth=0, color = ['k', col_CLSM, col_PCLSM], markersize=2.0)

    plt.tight_layout()
    fname = 'res_tp1_sfmc_col_%s_row_%s' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    #### illustrate calibration and scaling
    cspec = '1'
    df['doy'] = df.index.dayofyear
    df['Year'] = df.index.year
    df_uncalib = df.copy()
    df_uncalib['Tb fcst CLSM OL (species=' + cspec + ')'] = 204. + 0.81 * (
            df['Tb fcst CLSM OL (species=' + cspec + ')'] - 207.)
    df_uncalib['Tb fcst PCLSM OL (species=' + cspec + ')'] = 209. + 0.87 * (
            df['Tb fcst CLSM OL (species=' + cspec + ')'] - 212.)

    hFig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(231)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs (species=' + cspec + ')'], '-', color=col_obs,
                 linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df_uncalib.loc[cyearcond, 'Tb fcst CLSM OL (species=' + cspec + ')'], '-',
                 color=col_CLSM, linewidth=0.5)
    ax1.axes.xaxis.set_ticklabels([])
    plt.ylim([190, 250])
    ax2 = plt.subplot(234)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs (species=' + cspec + ')'], '-', color=col_obs,
                 linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df_uncalib.loc[cyearcond, 'Tb fcst PCLSM OL (species=' + cspec + ')'], '-',
                 color=col_PCLSM, linewidth=0.5)
    plt.ylim([190, 250])

    ax1 = plt.subplot(232)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs (species=' + cspec + ')'], '-', color=col_obs,
                 linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb fcst CLSM OL (species=' + cspec + ')'], '-',
                 color=col_CLSM, linewidth=0.5)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    plt.ylim([190, 250])
    ax2 = plt.subplot(235)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs (species=' + cspec + ')'], '-', color=col_obs,
                 linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb fcst PCLSM OL (species=' + cspec + ')'], '-',
                 color=col_PCLSM, linewidth=0.5)
    ax2.axes.yaxis.set_ticklabels([])
    plt.ylim([190, 250])

    ax1 = plt.subplot(233)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs resc. CLSM (species=' + cspec + ')'], '-',
                 color=col_obs, linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb fcst CLSM OL (species=' + cspec + ')'], '-',
                 color=col_CLSM, linewidth=0.5)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    plt.ylim([190, 250])
    ax2 = plt.subplot(236)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb obs resc. PCLSM (species=' + cspec + ')'], '-',
                 color=col_obs, linewidth=0.5)
    for cyear in np.unique(df['Year']):
        condna = np.isnan(df['Tb obs (species=' + cspec + ')'].values)
        cyearcond = (df['Year'].values == cyear) & (condna == 0)
        plt.plot(df.loc[cyearcond, 'doy'], df.loc[cyearcond, 'Tb fcst PCLSM OL (species=' + cspec + ')'], '-',
                 color=col_PCLSM, linewidth=0.5)
    ax2.axes.yaxis.set_ticklabels([])
    plt.ylim([190, 250])

    fname = 'scaling_species=' + cspec + '_col_%s_row_%s' % (col, row)
    fname_long = os.path.join(outpath, exp1, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()

    plt.tight_layout()

    # emissivity
    # ax4 = plt.subplot(414)
    # ax4.plot(df.loc[:,'Tb obs (species=2)']/df2.loc[:,'tp1 open loop (PCLSM)'],marker='*', linestyle='-', color='k')
    # ax4.plot(df.loc[:,'Tb fcst CLSM OL (species=2)']/df2.loc[:,'tp1 open loop (CLSM)'],marker='o', markerfacecolor="None", linestyle='-', color=col_CLSM)
    # ax4.plot(df.loc[:,'Tb fcst PCLSM OL (species=2)']/df2.loc[:,'tp1 open loop (PCLSM)'],marker='o', markerfacecolor="None", linestyle='-', color=col_PCLSM)
    # plt.ylabel('e [-]')

    # Tb misfit
    Tb_misfit_CLSM_OL = df.loc[:, 'Tb fcst CLSM OL (species=2)'] - df.loc[:, 'Tb obs (species=2)']
    Tb_misfit_PCLSM_OL = df.loc[:, 'Tb fcst PCLSM OL (species=2)'] - df.loc[:, 'Tb obs (species=2)']
    print('Tb_misfit(CLSM OL): %0.2f' % (np.var(Tb_misfit_CLSM_OL)))
    print('Tb_misfit(PCLSM OL): %0.2f' % (np.var(Tb_misfit_PCLSM_OL)))


def plot_scatter_daily_output(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp)
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    fontsize = 12
    # get poros for col row
    cond = (poros > 0.65) & (lons > 19.0) & (lons < 20.3) & (lats > 2.0) & (lats < 2.5)
    # cond = (poros>0.65) & (lons>65.0) & (lons<66.3) & (lats>60.0) & (lats<62.5)
    # tropics Taka1 114.05956;-2.31735
    # cond = (poros>0.65) & (lons>114.0) & (lons<114.12) & (lats>-2.38) & (lats<-2.25)
    cond = np.array([np.where(cond)[0][0], np.where(cond)[1][0]])
    plt.figure(figsize=(19, 8))

    ax1 = plt.subplot(231)
    plt.plot(io.timeseries['sfmc'][:, cond[0], cond[1]].values, io.timeseries['zbar'][:, cond[0], cond[1]].values, '.')
    plt.ylabel('zbar [m]')
    plt.xlabel('sfmc [-]')

    ax1 = plt.subplot(232)
    plt.plot(io.timeseries['rzmc'][:, cond[0], cond[1]].values, io.timeseries['zbar'][:, cond[0], cond[1]].values, '.')
    plt.ylabel('zbar [m]')
    plt.xlabel('rzmc [-]')

    ax2 = plt.subplot(233)
    plt.plot(io.timeseries['evap'][:, cond[0], cond[1]], io.timeseries['zbar'][:, cond[0], cond[1]], '.')
    plt.ylabel('zbar [m]')
    plt.xlabel('evap [-]')

    # ax2 = plt.subplot(233)
    # plt.plot((1-io.timeseries['ar1'][:,cond[0],cond[1]]-io.timeseries['ar2'][:,cond[0],cond[1]]),io.timeseries['zbar'][:,cond[0],cond[1]],'.')
    # plt.ylabel('zbar [m]')
    # plt.xlabel('ar4 [-]')

    ax2 = plt.subplot(234)
    plt.plot(io.timeseries['eveg'][:, cond[0], cond[1]], io.timeseries['zbar'][:, cond[0], cond[1]], '.')
    plt.ylabel('zbar [m]')
    plt.xlabel('eveg [-]')

    ax2 = plt.subplot(235)
    plt.plot(io.timeseries['esoi'][:, cond[0], cond[1]], io.timeseries['zbar'][:, cond[0], cond[1]], '.')
    plt.ylabel('zbar [m]')
    plt.xlabel('esoi [-]')

    ax2 = plt.subplot(236)
    plt.plot(io.timeseries['zbar'][:, cond[0], cond[1]],
             io.timeseries['eveg'][:, cond[0], cond[1]] / io.timeseries['esoi'][:, cond[0], cond[1]], '.')
    plt.xlabel('zbar [m]')
    plt.ylabel('eveg/esoi [-]')

    # my_title = 'lon=%s, lat=%s poros=%.2f' % (lon, lat, siteporos)
    # plt.title(my_title, fontsize=fontsize+2)
    plt.tight_layout()
    fname = 'scatter_test_IN'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_timeseries_daily_compare(exp1, exp2, domain, root, outpath):
    outpath = os.path.join(outpath, exp1)
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp1, domain=domain, root=root)
    io2 = LDAS_io('daily', exp=exp2, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io2.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io2.grid.tilecoord.j_indg.values, io2.grid.tilecoord.i_indg.values] = catparam['poros'].values

    fontsize = 12
    # get poros for col row
    cond = (poros > 0.65) & (lons > 19.0) & (lons < 20.3) & (lats > 2.0) & (lats < 2.5)
    # cond = (poros>0.65) & (lons>65.0) & (lons<66.3) & (lats>60.0) & (lats<62.5)
    # tropics Taka1 114.05956;-2.31735
    # cond = (poros>0.65) & (lons>114.0) & (lons<114.12) & (lats>-2.38) & (lats<-2.25)
    cond = np.array([np.where(cond)[0][0], np.where(cond)[1][0]])
    plt.figure(figsize=(19, 8))

    ax1 = plt.subplot(311)
    plt.plot(io.timeseries['time'], io.timeseries['catdef'][:, cond[0], cond[1]].values, 'b.')
    plt.plot(io2.timeseries['time'], io2.timeseries['catdef'][:, cond[0], cond[1]].values, 'r.')
    plt.ylabel('time')
    plt.xlabel('catdef [mm]')

    ax1 = plt.subplot(312)
    plt.plot(io.timeseries['time'], io.timeseries['rzmc'][:, cond[0], cond[1]].values, 'b.')
    plt.plot(io2.timeseries['time'], io2.timeseries['rzmc'][:, cond[0], cond[1]].values, 'r.')
    plt.ylabel('time')
    plt.xlabel('rzmc [mm]')

    ax1 = plt.subplot(313)
    plt.plot(io.timeseries['time'], io.timeseries['sfmc'][:, cond[0], cond[1]].values, 'b.')
    plt.plot(io2.timeseries['time'], io2.timeseries['sfmc'][:, cond[0], cond[1]].values, 'r.')
    plt.ylabel('time')
    plt.xlabel('sfmc [mm]')

    plt.tight_layout()
    fname = 'timeseries_test_IN1IN2'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_timeseries_fsw_change(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp)
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    fontsize = 12
    # get poros for col row
    # cond = (poros>0.65) & (lons>19.0) & (lons<20.3) & (lats>2.0) & (lats<2.5)
    cond = (poros > 0.65) & (lons > 65.0) & (lons < 66.3) & (lats > 60.0) & (lats < 62.5)
    # tropics Taka1 114.05956;-2.31735
    # cond = (poros>0.65) & (lons>114.0) & (lons<114.12) & (lats>-2.38) & (lats<-2.25)
    cond = np.array([np.where(cond)[0][0], np.where(cond)[1][0]])

    fsw = io.timeseries['fsw_change'][:, cond[0], cond[1]].values * 24 * 60 * 60
    fsw_min = np.min(np.cumsum(fsw))

    plt.figure(figsize=(19, 8))

    ax1 = plt.subplot(311)
    plt.plot(io.timeseries['time'], io.timeseries['zbar'][:, cond[0], cond[1]].values, 'b.')
    plt.ylabel('zbar [mm]')

    ax1 = plt.subplot(312)
    plt.plot(io.timeseries['time'], fsw, 'b.')
    plt.ylabel('fsw_change [mm]')

    ax1 = plt.subplot(313)
    plt.plot(io.timeseries['time'], np.cumsum(fsw), 'b.')
    plt.ylabel('cumulative fsw_change [mm]')

    plt.tight_layout()
    fname = 'timeseries_fsw_change'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=300)
    plt.close()

    plt.figure(figsize=(19, 8))
    ax1 = plt.subplot(121)
    plt.plot(1000 * io.timeseries['zbar'][:, cond[0], cond[1]].values, np.cumsum(fsw) - fsw_min, '.')
    plt.ylabel('fsw relative to absolute minimum [mm]')
    plt.xlabel('-zbar [mm]')
    ax1 = plt.subplot(122)
    plt.plot(1000 * io.timeseries['zbar'][:, cond[0], cond[1]].values, np.cumsum(fsw) - fsw_min, '.')
    plt.ylabel('fsw relative to absolute minimum [mm]')
    plt.xlabel('-zbar [mm]')
    plt.xlim([-50, 0])
    plt.ylim([0, 50])
    fname = 'fsw_change_crossplot'
    fname_long = os.path.join(outpath, fname + '.png')
    plt.savefig(fname_long, dpi=150)
    plt.close()


def plot_map_cum_fsw_change_for_same_wtd(exp, domain, root, outpath):
    outpath = os.path.join(outpath, exp)
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    fontsize = 12
    # get poros for col row
    # cond = (poros>0.65) & (lons>19.0) & (lons<20.3) & (lats>2.0) & (lats<2.5)
    cond = (poros > 0.65)
    # tropics Taka1 114.05956;-2.31735
    # cond = (poros>0.65) & (lons>114.0) & (lons<114.12) & (lats>-2.38) & (lats<-2.25)
    # cond = np.array([np.where(cond)[0][0],np.where(cond)[1][0]])

    fsw_map = np.zeros([np.shape(lons)[0], np.shape(lons)[1]])
    io.images['fsw_change'][0, :, :]
    fsw_error = 0
    for clon in range(np.shape(lons)[1]):
        print('clon: ', clon)
        for clat in range(np.shape(lons)[0]):
            if cond[clat, clon] == True:
                fsw = io.timeseries['fsw_change'][:, clat, clon].values * 24 * 60 * 60
                fsw1 = fsw[135:330]
                fsw2 = fsw[(1825 + 135):(1825 + 330)]
                zbar = io.timeseries['zbar'][:, clat, clon].values
                zbar1 = zbar[135:330]
                zbar2 = zbar[(1825 + 135):(1825 + 330)]
                fsw_error = 0
                for z1 in range(np.size(zbar1)):
                    cdif = np.abs(zbar2 - zbar1[z1])
                    if np.min(cdif) < 0.001:
                        z2 = np.argmin(cdif)
                        fsw_error = fsw1[z1] - fsw2[z2]
                        break
                if fsw_error == 0:
                    for z1 in range(np.size(zbar1)):
                        cdif = np.abs(zbar2 - zbar1[z1])
                        if np.min(cdif) < 0.002:
                            z2 = np.argmin(cdif)
                            fsw_error = fsw1[z1] - fsw2[z2]
                            break
                fsw_map[clat, clon] = fsw_error

    print("now plot")

    varname = 'fsw_error'
    plot_title = varname
    if varname == 'fsw_error':
        plot_title = 'average annual fsw_error [mm]'
    fname = varname
    # plot_title='zbar [m]'
    cmin = None
    cmax = None

    fsw_map[fsw_map == 0] = np.nan

    figure_single_default(data=fsw_map / 5.5, lons=lons, lats=lats, cmin=cmin, cmax=cmax, llcrnrlat=llcrnrlat,
                          urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, outpath=outpath, exp=exp, fname=fname,
                          plot_title=plot_title)
    # m2 = io.timeseries.std(axis=0)


if __name__ == '__main__':
    plot_ismn_statistics()
