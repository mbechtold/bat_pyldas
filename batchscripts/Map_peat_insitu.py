import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
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
from scipy.interpolate import interp2d
from validation_good_practice.ancillary import metrics
import sys
from scipy import stats
import pymannkendall as mk
import copy

# set all the paths to read in data
root = '/staging/leuven/stg_00024/OUTPUT/michelb/'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/TROPICS/sm_sensitivity_test'
exp = 'PEATREV_PEATMAPHWSD'
domain = 'SMAP_EASEv2_M09'

# red in the actual data
lsm = LDAS_io('catparam', exp=exp, domain=domain, root=root)
params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
[lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
tc = lsm.grid.tilecoord
tg = lsm.grid.tilegrids
poros[tc.j_indg.values, tc.i_indg.values] = params['poros'].values

# generated nc file stored in working directory under this name
result_file = '/scratch/leuven/324/vsc32460/output/TROPICS/SM_E_correlationmap_drained.nc'

tags = ['pearsonR']

[lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
params = LDAS_io(exp=exp, domain=domain, root=root).read_params('catparam')
poros = np.full(lons.shape, np.nan)
tc = lsm.grid.tilecoord
tg = lsm.grid.tilegrids
poros[tc.j_indg.values, tc.i_indg.values] = params['poros'].values

# lons1D = np.unique(lsm.grid.tilecoord['com_lon'].values)
# lats1D = np.unique(lsm.grid.tilecoord['com_lat'].values)[::-1]

species = ObsFcstAna.timeseries['species'].values

ds = ncfile_init(result_file, lats[:, 0], lons[0, :], species, tags)

# [col, row] = get_M09_ObsFcstAna(ObsFcstAna,lon,lat)
# [col, row] = get_M09_ObsFcstAna(ObsFcstAna,lons.min()+2,lats.min()+2)
# col = col%4

for row in range(poros.shape[0]):
    print("row: " + str(row))
    for col in range(poros.shape[1]):
        if poros[row, col] > 0.6:
            for i_spc, spc in enumerate(species):
                [col_obs, row_obs] = get_M09_ObsFcstAna(ObsFcstAna, col, row, lonlat=False)
                ts_sfmc = lsm.read_ts('sfmc', col, row, lonlat=False)
                ts_tp1 = lsm.read_ts('tp1', col, row, lonlat=False)
                ts_obsobs = ObsFcstAna.read_ts('obs_obs', col_obs, row_obs, species=i_spc + 1, lonlat=False)
                ts_obsobs.name = 'obsobs'

                df = pd.concat((ts_sfmc, ts_tp1, ts_obsobs), axis=1)
                ts_emissivity = df['obsobs'] / df['tp1']

                df = pd.concat((ts_sfmc, ts_emissivity), axis=1)
                if not np.isnan(df.corr().values[0, 1]):
                    ds.variables['pearsonR'][row, col, i_spc] = df.corr().values[0, 1]

                # pearsonr([0]['obs_obs'][i_spc] - [0]['obs_fcst'][i_spc]).mean(dim='time').values
                # tmp = [0]['obs_obs'][i_spc].values
                # np.place(tmp, ~np.isnan(tmp), 1.)
                # np.place(tmp, np.isnan(tmp), 0.)
                # ds['pearsonR'][:, :, i_spc] = tmp.sum(axis=2)

# plt.imshow(-1.0*ds['pearsonR'][:,:,2])
# data = np.full(lons.shape, np.nan)
# data[tc.j_indg.values, tc.i_indg.values] =
for i_spc, spc in enumerate(species):
    data = np.ma.masked_invalid(ds['pearsonR'][:, :, i_spc])
    # tmp_data = obs_M09_to_M36(data)
    fname = 'R_eSM_sp' + str(i_spc)
    figpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Drained/DA_sensitivity'
    # drained zoom
    latmin = -1.5
    latmax = 0.5
    lonmin = 102.2
    lonmax = 104.2

    # natural zoom
    # latmin = -3.9
    # latmax = -1.9
    # lonmin = 113.1
    # lonmax = 115.1
    cmin = -0.85
    cmax = -0.25
    [data_zoom, lons_zoom, lats_zoom, llcrnrlat_zoom, urcrnrlat_zoom, llcrnrlon_zoom, urcrnrlon_zoom] = figure_zoom(
        data, lons, lats, latmin, latmax, lonmin, lonmax)
    figure_single_default_zoom(data=data_zoom, lons=lons_zoom, lats=lats_zoom, cmin=cmin, cmax=cmax,
                               llcrnrlat=llcrnrlat_zoom, urcrnrlat=urcrnrlat_zoom,
                               llcrnrlon=llcrnrlon_zoom, urcrnrlon=urcrnrlon_zoom, outpath=figpath, exp=exp,
                               fname=fname + '_zoom_' + figpath[52:59],
                               plot_title='R (-), ' + fname + ' ,zoom ,' + figpath[52:59], cmap='jet')













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
frac_peatland = np.nansum(np.all(np.vstack((frac_cell > 0.5, params['poros'].values > 0.8)), axis=0)) / np.nansum(
    params['poros'].values > 0.01)
frac_peatland_less5 = np.nansum(
    np.all(np.vstack((frac_cell > 0.95, params['poros'].values > 0.8)), axis=0)) / np.nansum(
    params['poros'].values > 0.8)
# set more than 5% open water grid cells to 1.
params[param].values[np.all(np.vstack((frac_cell < 0.95, params['poros'].values > 0.8)), axis=0)] = 1.
params[param].values[params['poros'].values < 0.8] = np.nan
params[param].values[np.all(np.vstack((params['poros'].values < 0.95, params['poros'].values > 0.8)), axis=0)] = 0.0
img = np.full(lons.shape, np.nan)
img[tc.j_indg.values, tc.i_indg.values] = params[param].values
data = np.ma.masked_invalid(img)
fname = '01b_' + param
cmin = 0
cmax = 1
title = 'Peatland distribution'
# open figure
figsize = (0.85 * 13, 0.85 * 10)
fontsize = 13
f = plt.figure(num=None, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
cmap = matplotlib.colors.ListedColormap([[255. / 255, 193. / 255, 7. / 255], [30. / 255, 136. / 255, 229. / 255]])
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
m.readshapefile('/data/leuven/317/vsc31786/gis/permafrost/permafrost_boundary', 'permafrost_boundary', linewidth=1.3,
                color=(0. / 255, 0. / 255, 0. / 255))
# load peatland sites
sites = pd.read_csv('/data/leuven/317/vsc31786/FIG_tmp/00DA/20190228_M09/cluster_radius_2_bog.txt', sep=',')
# http://lagrange.univ-lyon1.fr/docs/matplotlib/users/colormapnorms.html
# lat=48.
# lon=51.0
x, y = m(sites['Lon'].values, sites['Lat'].values)
if np.mean(lats) > 40:
    m.plot(x, y, '.', color=(0. / 255, 0. / 255, 0. / 255), markersize=12, markeredgewidth=1.5, mfc='none')
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
