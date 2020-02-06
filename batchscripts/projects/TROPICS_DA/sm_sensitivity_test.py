#!/usr/bin/env python

import os
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from bat_pyldas.plotting import *
from bat_pyldas.functions import *
import getpass

root='/scratch/leuven/317/vsc31786/output/TROPICS'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/TROPICS/sm_sensitivity_test'
exp='INDONESIA_M09_PEATCLSMTD_v01_SMOSfw'
domain='SMAP_EASEv2_M09'

lsm = LDAS_io('inst', exp=exp, domain=domain, root=root)
ObsFcstAna = LDAS_io('ObsFcstAna', exp=exp, domain=domain, root=root)

# natural
lon=138.6
lat=-5.1
lon = 144.7
lat = -6.8
lon = 113.74
lat = -2.68
# Drained Indonesia
lon=100.3
lat=1.8

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

plt.close(fig)


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

