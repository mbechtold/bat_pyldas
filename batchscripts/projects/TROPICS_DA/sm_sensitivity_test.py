#!/usr/bin/env python

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *
from bat_pyldas.functions import *
import getpass
import numpy as np
import logging
import pandas as pd
from scipy.stats import pearsonr
from netCDF4 import Dataset
from pyldas.interface import LDAS_io




# sebastian added plot for wtd vs sfmc
root = '/staging/leuven/stg_00024/OUTPUT/sebastiana'
exp = 'INDONESIA_M09_PEATCLSMTN_v01'
domain = 'SMAP_EASEv2_M09'
outpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Natural/'

if '/Drained' in outpath:
    style = ['m.']
elif '/Natural' in outpath:
    style = ['g.']

lsm = LDAS_io('daily', exp=exp, domain=domain, root=root)

# natural
lon=138.6
lat=-5.1
lon = 144.7
lat = -6.8

#sebangau forest
lon = 114.2
lat = -3
# Drained Indonesia
lon=103
lat=-0.75
[col, row] = lsm.grid.lonlat2colrow(lon, lat, domain=True)


sfmc = lsm.read_ts('sfmc', col, row, lonlat=False)
zbar = lsm.read_ts('zbar', col, row, lonlat=False)

df = pd.concat((sfmc,zbar),axis=1)

fig = plt.figure(figsize=(20,10))
fontsize = 18
ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1, fig=None)
df['sfmc'].plot(ax=ax1,style=style, markersize=5, fontsize=fontsize+8, linewidth=2)
plt.ylabel('Surface moisture content (vol/vol)', fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

ax0 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1, fig=None)
df['zbar'].plot(ax=ax0,style=style, markersize=5, fontsize=fontsize+8, linewidth=2)
plt.ylabel('Water table depth (m)', fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

fname = 'zbarandsfmc_' +outpath[52:59]
fname_long = os.path.join(outpath + fname + '.png')
plt.savefig(fname_long, dpi=300)

fig1 = plt.figure(figsize=(18,18))
df.plot(y='sfmc', x='zbar', style=style, fontsize= fontsize+4, linewidth=2)
plt.ylabel('Surface moisture content (vol/vol)', fontsize=14)
plt.xlabel('Water Table depth (m)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

fname = 'zbar-sfmc_' +outpath[52:59]
fname_long = os.path.join(outpath + fname + '.png')
plt.savefig(fname_long, dpi=300)




root='/staging/leuven/stg_00024/OUTPUT/michelb/TROPICS'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/TROPICS/sm_sensitivity_test'
exp='INDONESIA_M09_PEATCLSMTN_v01_SMOSfw'
domain='SMAP_EASEv2_M09'

lsm = LDAS_io('inst', exp=exp, domain=domain, root=root)
ObsFcstAna = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)

# natural
lon=138.6
lat=-5.1
lon = 144.7
lat = -6.8

#sebangau forest
lon = 114.2
lat = -3
# Drained Indonesia
#lon=103
#lat=-0.75

#[col, row] = lsm.grid.lonlat2colrow(lon, lat, domain=True)
[col, row] = get_M09_ObsFcstAna(ObsFcstAna,lon,lat)
plt.imshow(lsm.images['sfmc'][1000,:,:])

#ObsFcstAna.timeseries['obs_obs'][:,1,row,col]
#plt.hist(lsm.timeseries['sfmc'][0:1000,row,col])

ts_sfmc = lsm.read_ts('sfmc', col, row, lonlat=False)
ts_tp1 = lsm.read_ts('tp1', col, row, lonlat=False)
ts_lai = lsm.read_ts('lai', col, row, lonlat=False)
ts_obsobs_ha = ObsFcstAna.read_ts('obs_obs', col, row, species=1, lonlat=False)
ts_obsobs_ha.name = 'obsobs_ha'
ts_obsobs_hd = ObsFcstAna.read_ts('obs_obs', col, row, species=2, lonlat=False)
ts_obsobs_hd.name = 'obsobs_hd'
ts_obsobs_va = ObsFcstAna.read_ts('obs_obs', col, row, species=3, lonlat=False)
ts_obsobs_va.name = 'obsobs_va'
ts_obsobs_vd = ObsFcstAna.read_ts('obs_obs', col, row, species=4, lonlat=False)
ts_obsobs_vd.name = 'obsobs_vd'

df = pd.concat((ts_sfmc,ts_tp1,ts_lai,ts_obsobs_ha,ts_obsobs_hd,ts_obsobs_va,ts_obsobs_vd),axis=1)
ts_emissivity_ha = df['obsobs_ha']/df['tp1']
ts_emissivity_ha.name='emissivity_ha'
ts_emissivity_hd = df['obsobs_hd']/df['tp1']
ts_emissivity_hd.name='emissivity_hd'
ts_emissivity_va = df['obsobs_va']/df['tp1']
ts_emissivity_va.name='emissivity_va'
ts_emissivity_vd = df['obsobs_vd']/df['tp1']
ts_emissivity_vd.name='emissivity_vd'
# MPDI
ts_mpdi = (df['obsobs_va']-df['obsobs_ha'])/(df['obsobs_va']+df['obsobs_ha'])
ts_mpdi.name = 'mpdi'

df = pd.concat((ts_sfmc,ts_tp1,ts_lai,ts_obsobs_ha,ts_obsobs_hd,ts_obsobs_va,ts_obsobs_vd,
                ts_emissivity_ha,ts_emissivity_hd,ts_emissivity_va,ts_emissivity_vd,
                ts_mpdi),axis=1)

fig = plt.figure(figsize=(22,7))
fontsize = 18
ax1 = plt.subplot(211)
df['sfmc'].plot(ax=ax1,style=['g.'], markersize=5, fontsize=fontsize+12, linewidth=2)
plt.ylabel('SM [vol.]', fontsize=fontsize +13)
plt.xlabel(" ")
plt.yticks(ticks=[0.1, 0.3, 0.5, 0.7, 0.9], labels=['0.1', '0.3', '0.5', '0.7', '0.9'])
ax2 = plt.subplot(212)
df['emissivity_hd'].plot(ax=ax2,style=['gx'], markersize=6,fontsize=fontsize+12, linewidth=2)
plt.ylabel('e (H-pol) [-]', fontsize=fontsize +13)
plt.xlabel(" ")
plt.yticks(ticks=[0.74, 0.79, 0.84, 0.89, 0.94], labels=['0.74', '0.79', '0.84', '0.89', '0.94'])
figpath = '/data/leuven/324/vsc32460/FIG/in_situ_comparison/IN/Natural/DA_sensitivity/'
fname = 'Timeseries_DA_' +figpath[52:59]
fname_long = os.path.join(figpath + fname + '.png')
plt.savefig(fname_long, dpi=300)

fig = plt.figure(figsize=(19,7))
fontsize = 12
ax1 = plt.subplot(511)
df['sfmc'].plot(ax=ax1,style=['o'], fontsize=fontsize, linewidth=2)
plt.ylabel('sm [vol.]')
ax2 = plt.subplot(512)
df['obsobs_ha'].plot(ax=ax2,style=['x'], fontsize=fontsize, linewidth=2)
plt.ylabel('Tb obs [vol.]')
ax3 = plt.subplot(513)
df['emissivity_ha'].plot(ax=ax3,style=['x'], fontsize=fontsize, linewidth=2)
plt.ylabel('emissivity ha [-]')
ax4 = plt.subplot(514)
df['mpdi'].plot(ax=ax4,style=['x'], fontsize=fontsize, linewidth=2)
plt.ylabel('mpdi [-]') 
ax5 = plt.subplot(515)
df['lai'].plot(ax=ax5,style=['x'], fontsize=fontsize, linewidth=2)
plt.ylabel('lai [m2/m2]')


fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(221)
plt.plot(df['emissivity_ha'],df['sfmc'],'.')
plt.xlabel('e_ha')
ax2 = plt.subplot(222)
plt.plot(df['emissivity_va'],df['sfmc'],'.')
plt.xlabel('e_va')
ax3 = plt.subplot(223)
plt.plot(df['mpdi'],df['sfmc'],'.')
plt.xlabel('mpdi')
plt.show()

