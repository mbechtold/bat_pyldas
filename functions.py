# collections of functions to work with LDAS output over peatlands
# Michel Bechtold
# November 2019, KU Leuven

import os
import sys
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
from pathlib import Path
from pyldas.functions import find_files
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


# SA added
def read_et_data(insitu_path, mastertable_filename, exp, domain, root):
    """read master table of ET in csv format (; separated), use mastertable to select coordinates
    if coordinates are within extent and on peat select in situ ET and store both in dataframe for use in plotting"""

    folder_general_et = insitu_path + '/ET/'
    # set insitu path and read in mastertable ET
    filenames = find_files(folder_general_et, mastertable_filename)
    if isinstance(find_files(folder_general_et, mastertable_filename), str):
        master_table = pd.read_csv(filenames, sep=';')
    else:
        for filename in filenames:
            if filename.endswith('csv'):
                master_table = pd.read_csv(filename, sep=';')
                continue
            else:
                logging.warning(
                    "some files, maybe swp files, exist that start with master table searchstring, not per se the one loading (WTD/ET)")

    # load LDAS structure data in daily timesteps
    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    # check porosity of coordinate tile, if smaller than 0.6 it i not a peat pixel
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    first_site = True

    for i, site_ID in enumerate(master_table.iloc[:, 0]):

        # If site is on the blacklist (contains bad data), or if "comparison" =  0, then don include that site in the dataframe.
        if master_table['comparison_yes'][i] == 0:
            continue

        # to select only the drained of only the natural ones to calculate skill metrics!

        # if master_table['drained_U=uncertain'][i] == 'D':
        #    continue

        # if master_table['drained_U=uncertain'][i] == 'U':
        #    continue

        # Get lat lon from master table for site.
        lon = master_table['lon'][i]

        # sebastian added   # to skip a site if there is no lon or lat when all are used instead of only the comparison =1 ones.
        # if np.isnan(lon):
        #    continue

        lat = master_table['lat'][i]

        # Get porosity for site lon lat.
        # Get M09 rowcol with data.
        col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
        # Get poros for col row. + Check whether site is in domain, if not 'continue' with next site
        try:
            siteporos = poros[row, col]
        except:
            print(site_ID + " not in domain.")
            continue

        if siteporos <= 0.60:
            # If the porosity of the site is 0.6 or lower the site is not classified as peatland in the model.
            print(site_ID + " not on a peatland grid cell.")
            continue

        folder_et = insitu_path + '/ET/et/'
        folder_ee = insitu_path + '/ET/ee/'
        folder_br = insitu_path + '/ET/br/'
        folder_rn = insitu_path + '/ET/rn/'
        folder_sh = insitu_path + '/ET/sh/'
        folder_le = insitu_path + '/ET/le/'
        folder_Tair = insitu_path + '/ET/tair/'
        folder_vpd = insitu_path + '/ET/vpd/'
        folder_wtd = insitu_path + '/WTD/'

        # load csv file for et
        try:
            if isinstance(find_files(folder_et, site_ID), str):
                filename_et = find_files(folder_et, site_ID)
                print(filename_et)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_et = f
        except:
            print(site_ID + " does not have a ET csv file with data.")
            continue
        # check for empty path
        if len(filename_et) < 10 or filename_et.endswith('csv') != True or filename_et.count('WTD') >= 1:
            print("checking ... " + site_ID + " " + filename_et)
            print("some other reason for no data for " + site_ID + " in " + filename_et)
            continue

        # load csv file of ee
        try:
            if isinstance(find_files(folder_ee, site_ID), str):
                filename_ee = find_files(folder_ee, site_ID)
                print(filename_ee)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_ee = f
        except:
            print(site_ID + " does not have a EE csv file with data.")
            continue
            # check for empty path
        if len(filename_ee) < 10 or filename_ee.endswith('csv') != True or filename_ee.count('WTD') >= 1:
            print("checking ... " + site_ID + " " + filename_ee)
            print("some other reason for no data for " + site_ID + " in " + filename_ee)

        # load csv file of br
        try:
            if isinstance(find_files(folder_br, site_ID), str):
                filename_br = find_files(folder_br, site_ID)
                print(filename_br)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_br = f
        except:
            print(site_ID + " does not have a BR csv file with data.")
            continue
            # check for empty path
        if len(filename_br) < 10 or filename_br.endswith('csv') != True or filename_br.count('WTD') >= 1:
            print("checking ... " + site_ID + " " + filename_br)
            print("some other reason for no data for " + site_ID + " in " + filename_br)

        # load csv file of rn
        try:
            if isinstance(find_files(folder_rn, site_ID), str):
                filename_rn = find_files(folder_rn, site_ID)
                print(filename_rn)

            else:
                flist = find_files(folder_rn, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_rn = f
        except:
            print(site_ID + " does not have a RN csv file with data.")
            continue
            # check for empty path
        if len(filename_rn) < 10 or filename_rn.endswith('csv') != True or filename_rn.count('WTD') >= 1:
            print("checking ... " + site_ID + " " + filename_rn)
            print("some other reason for no data for " + site_ID + " in " + filename_rn)

        # load csv file of tair
        try:
            if isinstance(find_files(folder_Tair, site_ID), str):
                filename_Tair = find_files(folder_Tair, site_ID)
                print(filename_Tair)

            else:
                flist = find_files(folder_Tair, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_rn = f

            # check for empty path
            if len(filename_Tair) < 10 or filename_Tair.endswith('csv') != True or filename_Tair.count('WTD') >= 1:
                print("checking ... " + site_ID + " " + filename_Tair)
                print("some other reason for no data for " + site_ID + " in " + filename_Tair)

        except:
            print(site_ID + " does not have a Tair csv file with data.")

        # load csv file of sh
        try:
            if isinstance(find_files(folder_sh, site_ID), str):
                filename_sh = find_files(folder_sh, site_ID)
                print(filename_sh)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_sh = f
                # check for empty path
            if len(filename_sh) < 10 or filename_sh.endswith('csv') != True or filename_sh.count('WTD') >= 1:
                print("checking ... " + site_ID + " " + filename_sh)
                print("some other reason for no data for " + site_ID + " in " + filename_sh)

        except:
            print(site_ID + " does not have a SH csv file with data.")

        # load csv file of le
        try:
            if isinstance(find_files(folder_le, site_ID), str):
                filename_le = find_files(folder_le, site_ID)
                print(filename_le)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_le = f

                # check for empty path
            if len(filename_le) < 10 or filename_le.endswith('csv') != True or filename_le.count('WTD') >= 1:
                print("checking ... " + site_ID + " " + filename_le)
                print("some other reason for no data for " + site_ID + " in " + filename_le)

        except:
            print(site_ID + " does not have a LE csv file with data.")

        # load csv file of wtd
        try:
            if isinstance(find_files(folder_wtd, site_ID), str):
                filename_wtd = find_files(folder_wtd, site_ID)
                print(filename_wtd)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_wtd = f
                # check for empty path
            if len(filename_wtd) < 10 or filename_wtd.endswith('csv') != True:
                print("checking ... " + site_ID + " " + filename_wtd)
                print("some other reason for no data for " + site_ID + " in " + filename_wtd)

        except:
            print(site_ID + " does not have a WTD csv file with data.")

        # load csv file of vpd
        try:
            if isinstance(find_files(folder_vpd, site_ID), str):
                filename_vpd = find_files(folder_vpd, site_ID)
                print(filename_vpd)
            else:
                flist = find_files(folder_vpd, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_vpd = f
                # check for empty path
            if len(filename_wtd) < 10 or filename_vpd.endswith('csv') != True:
                print("checking ... " + site_ID + " " + filename_vpd)
                print("some other reason for no data for " + site_ID + " in " + filename_vpd)

        except:
            print(site_ID + " does not have a vpd csv file with data.")

            # cumbersome first site, next sites ... to be simplified ...
        if first_site == True:

            # Load in situ ET data
            print("reading ... " + filename_et)
            et_obs = pd.read_csv(filename_et)
            if et_obs.shape[1] == 1:
                print("ET csv file with semicolon ...")
                et_obs = pd.read_csv(filename_et, sep=';')
            et_obs.columns = ['time', site_ID]
            et_obs['time'] = pd.to_datetime(et_obs['time'])
            et_obs = et_obs.set_index('time')

            # Load model ET data.
            et_mod = io.read_ts('evap', lon, lat, lonlat=True)
            eveg_mod = io.read_ts('eveg', lon, lat, lonlat=True)
            esoi_mod = io.read_ts('esoi', lon, lat, lonlat=True)
            eint_mod = io.read_ts('eint', lon, lat, lonlat=True)
            zbar_mod = io.read_ts('zbar', lon, lat, lonlat=True)
            AR1 = io.read_ts('ar1', lon, lat, lonlat=True)
            AR2 = io.read_ts('ar2', lon, lat, lonlat=True)
            AR4 = 1 - AR1 - AR2
            sfmc = io.read_ts('sfmc', lon, lat, lonlat=True)
            rzmc = io.read_ts('rzmc', lon, lat, lonlat=True)
            srfexc = io.read_ts('srfexc', lon, lat, lonlat=True)
            rzexc = io.read_ts('rzexc', lon, lat, lonlat=True)
            catdef = io.read_ts('catdef', lon, lat, lonlat=True)
            Qair = io.read_ts('Qair', lon, lat, lonlat=True)
            Wind = io.read_ts('Wind', lon, lat, lonlat=True)

            # Check if overlapping data.
            df_check = pd.concat((et_obs, et_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False

            # Load in situ EE data
            print("reading ... " + filename_ee)
            ee_obs = pd.read_csv(filename_ee)
            if ee_obs.shape[1] == 1:
                print("EE csv file with semicolon ...")
                ee_obs = pd.read_csv(filename_ee, sep=';')
            ee_obs.columns = ['time', site_ID]
            ee_obs['time'] = pd.to_datetime(ee_obs['time'])
            ee_obs = ee_obs.set_index('time')

            # Load in situ RN data
            print("reading ... " + filename_rn)
            rn_obs = pd.read_csv(filename_rn)
            if rn_obs.shape[1] == 1:
                print("RN csv file with semicolon ...")
                rn_obs = pd.read_csv(filename_rn, sep=';')
            rn_obs.columns = ['time', site_ID]
            rn_obs['time'] = pd.to_datetime(rn_obs['time'])
            rn_obs = rn_obs.set_index('time')

            # Load in situ Tair data
            try:
                print("reading ... " + filename_Tair)
                Tair_obs = pd.read_csv(filename_Tair)
                if Tair_obs.shape[1] == 1:
                    print("Tair csv file with semicolon ...")
                    Tair_obs = pd.read_csv(filename_Tair, sep=';')
                Tair_obs.columns = ['time', site_ID]
                Tair_obs['time'] = pd.to_datetime(Tair_obs['time'])
                Tair_obs = Tair_obs.set_index('time')
            except:
                Tair_obs = et_obs

            # Load in situ LE data
            try:
                print("reading ... " + filename_le)
                le_obs = pd.read_csv(filename_le)
                if le_obs.shape[1] == 1:
                    print("LE csv file with semicolon ...")
                    le_obs = pd.read_csv(filename_le, sep=';')
                le_obs.columns = ['time', site_ID]
                le_obs['time'] = pd.to_datetime(le_obs['time'])
                le_obs = le_obs.set_index('time')
            except:
                le_obs = et_obs

            # Load model EE data.
            try:
                le_mod = io.read_ts('lhflux', lon, lat, lonlat=True)
                le_mod = pd.DataFrame(le_mod)
                swi_mod = io.read_ts('SWdown', lon, lat, lonlat=True)
                lwi_mod = io.read_ts('LWdown', lon, lat, lonlat=True)
                swo_mod = io.read_ts('swup', lon, lat, lonlat=True)
                lwo_mod = io.read_ts('lwup', lon, lat, lonlat=True)
                rn_mod = pd.concat([lwi_mod, swi_mod, swo_mod, lwo_mod], axis=1)
                rn_mod = rn_mod['LWdown'] + rn_mod['SWdown'] - rn_mod['swup'] - rn_mod['lwup']
                ee_mod = le_mod['lhflux'] / rn_mod[1]
            except:
                print('not all model data is available')

            # Load in situ BR data
            print("reading ... " + filename_br)
            br_obs = pd.read_csv(filename_br)
            if br_obs.shape[1] == 1:
                print("BR csv file with semicolon ...")
                br_obs = pd.read_csv(filename_br, sep=';')
            br_obs.columns = ['time', site_ID]
            br_obs['time'] = pd.to_datetime(br_obs['time'])
            br_obs = br_obs.set_index('time')

            # Load in situ SH data
            print("reading ... " + filename_sh)
            sh_obs = pd.read_csv(filename_sh)
            if sh_obs.shape[1] == 1:
                print("SH csv file with semicolon ...")
                sh_obs = pd.read_csv(filename_sh, sep=';')
            sh_obs.columns = ['time', site_ID]
            sh_obs['time'] = pd.to_datetime(sh_obs['time'])
            sh_obs = sh_obs.set_index('time')

            # Load model BR data.
            le_mod = io.read_ts('lhflux', lon, lat, lonlat=True)
            le_mod = pd.DataFrame(le_mod)
            sh_mod = io.read_ts('shflux', lon, lat, lonlat=True)
            sh_mod = pd.DataFrame(sh_mod)
            br_mod = sh_mod['shflux'] / le_mod['lhflux']

            # Load in situ WTD data
            print("reading ... " + filename_wtd)
            wtd_obs = pd.read_csv(filename_wtd)
            if wtd_obs.shape[1] == 1:
                print("SH csv file with semicolon ...")
                wtd_obs = pd.read_csv(filename_wtd, sep=';')
            wtd_obs.columns = ['time', site_ID]
            wtd_obs['time'] = pd.to_datetime(wtd_obs['time'])
            wtd_obs = wtd_obs.set_index('time')

            # Load model ETpot calculation data
            ghflux_mod = io.read_ts('ghflux', lon, lat, lonlat=True)
            Psurf_mod = io.read_ts('Psurf', lon, lat, lonlat=True)
            Tair_mod = io.read_ts('Tair', lon, lat, lonlat=True)

            # Load in situ vpd data
            try:
                print("reading ... " + filename_vpd)
                vpd_obs = pd.read_csv(filename_vpd)
                if vpd_obs.shape[1] == 1:
                    print("vpd csv file with semicolon ...")
                    vpd_obs = pd.read_csv(filename_vpd, sep=';')
                vpd_obs.columns = ['time', site_ID]
                vpd_obs['time'] = pd.to_datetime(vpd_obs['time'])
                vpd_obs = vpd_obs.set_index('time')
            except:
                vpd_obs = et_obs


        else:
            # Load in situ ET data
            print("reading ... " + filename_et)
            et_obs_tmp = pd.read_csv(filename_et)
            if et_obs_tmp.shape[1] == 1:
                print("ET csv file with semicolon ...")
                et_obs_tmp = pd.read_csv(filename_et, sep=';')
            et_obs_tmp.columns = ['time', site_ID]
            et_obs_tmp['time'] = pd.to_datetime(et_obs_tmp['time'])
            et_obs_tmp = et_obs_tmp.set_index('time')

            # Load model data of ET (and components).
            et_mod_tmp = io.read_ts('evap', lon, lat, lonlat=True)
            eveg_mod_tmp = io.read_ts('eveg', lon, lat, lonlat=True)
            esoi_mod_tmp = io.read_ts('esoi', lon, lat, lonlat=True)
            eint_mod_tmp = io.read_ts('eint', lon, lat, lonlat=True)
            zbar_mod_tmp = io.read_ts('zbar', lon, lat, lonlat=True)
            AR1_tmp = io.read_ts('ar1', lon, lat, lonlat=True)
            AR2_tmp = io.read_ts('ar2', lon, lat, lonlat=True)
            AR4_tmp = 1 - AR1_tmp - AR2_tmp
            sfmc_tmp = io.read_ts('sfmc', lon, lat, lonlat=True)
            rzmc_tmp = io.read_ts('rzmc', lon, lat, lonlat=True)
            srfexc_tmp = io.read_ts('srfexc', lon, lat, lonlat=True)
            rzexc_tmp = io.read_ts('rzexc', lon, lat, lonlat=True)
            catdef_tmp = io.read_ts('catdef', lon, lat, lonlat=True)
            Qair_tmp = io.read_ts('Qair', lon, lat, lonlat=True)
            Wind_tmp = io.read_ts('Wind', lon, lat, lonlat=True)

            # Check if overlaping data.
            df_check_et = pd.concat((et_obs_tmp, et_mod_tmp), axis=1)
            no_overlap_et = pd.isnull(df_check_et).any(axis=1)

            if False in no_overlap_et.values:
                et_obs = pd.concat((et_obs, et_obs_tmp), axis=1)
                et_mod = pd.concat((et_mod, et_mod_tmp), axis=1)

            df_check_eveg = pd.concat((et_obs_tmp, eveg_mod_tmp), axis=1)
            no_overlap_eveg = pd.isnull(df_check_eveg).any(axis=1)

            if False in no_overlap_eveg.values:
                eveg_mod = pd.concat((eveg_mod, eveg_mod_tmp), axis=1)

            df_check_esoi = pd.concat((et_obs_tmp, esoi_mod_tmp), axis=1)
            no_overlap_esoi = pd.isnull(df_check_esoi).any(axis=1)

            if False in no_overlap_esoi.values:
                esoi_mod = pd.concat((esoi_mod, esoi_mod_tmp), axis=1)

            df_check_eint = pd.concat((et_obs_tmp, eint_mod_tmp), axis=1)
            no_overlap_eint = pd.isnull(df_check_eint).any(axis=1)

            if False in no_overlap_eint.values:
                eint_mod = pd.concat((eint_mod, eint_mod_tmp), axis=1)

            df_check_zbar = pd.concat((et_obs_tmp, zbar_mod_tmp), axis=1)
            no_overlap_zbar = pd.isnull(df_check_zbar).any(axis=1)

            if False in no_overlap_zbar.values:
                zbar_mod = pd.concat((zbar_mod, zbar_mod_tmp), axis=1)

            df_check_ar1 = pd.concat((AR1, AR1_tmp), axis=1)
            no_overlap_ar1 = pd.isnull(df_check_ar1).any(axis=1)

            if False in no_overlap_ar1.values:
                AR1 = pd.concat((AR1, AR1_tmp), axis=1)

            df_check_ar2 = pd.concat((AR2, AR2_tmp), axis=1)
            no_overlap_ar2 = pd.isnull(df_check_ar2).any(axis=1)

            if False in no_overlap_ar2.values:
                AR2 = pd.concat((AR2, AR2_tmp), axis=1)

            df_check_ar4 = pd.concat((AR4, AR4_tmp), axis=1)
            no_overlap_ar4 = pd.isnull(df_check_ar4).any(axis=1)

            if False in no_overlap_ar4.values:
                AR4 = pd.concat((AR4, AR4_tmp), axis=1)

            df_check_sfmc = pd.concat((sfmc, sfmc_tmp), axis=1)
            no_overlap_sfmc = pd.isnull(df_check_sfmc).any(axis=1)

            if False in no_overlap_sfmc.values:
                sfmc = pd.concat((sfmc, sfmc_tmp), axis=1)

            df_check_rzmc = pd.concat((rzmc, rzmc_tmp), axis=1)
            no_overlap_rzmc = pd.isnull(df_check_rzmc).any(axis=1)

            if False in no_overlap_rzmc.values:
                rzmc = pd.concat((rzmc, rzmc_tmp), axis=1)

            df_check_rzexc = pd.concat((rzexc, rzexc_tmp), axis=1)
            no_overlap_rzexc = pd.isnull(df_check_rzexc).any(axis=1)

            if False in no_overlap_rzexc.values:
                rzexc = pd.concat((rzexc, rzexc_tmp), axis=1)

            df_check_sfexc = pd.concat((srfexc, srfexc_tmp), axis=1)
            no_overlap_sfexc = pd.isnull(df_check_sfexc).any(axis=1)

            if False in no_overlap_sfexc.values:
                srfexc = pd.concat((srfexc, srfexc_tmp), axis=1)

            df_check_catdef = pd.concat((catdef, catdef_tmp), axis=1)
            no_overlap_catdef = pd.isnull(df_check_catdef).any(axis=1)

            if False in no_overlap_catdef.values:
                catdef = pd.concat((catdef, catdef_tmp), axis=1)

            df_check_Qair = pd.concat((Qair, Qair_tmp), axis=1)
            no_overlap_Qair = pd.isnull(df_check_Qair).any(axis=1)

            if False in no_overlap_Qair.values:
                Qair = pd.concat((Qair, Qair_tmp), axis=1)

            df_check_Wind = pd.concat((Wind, Wind_tmp), axis=1)
            no_overlap_Wind = pd.isnull(df_check_Wind).any(axis=1)

            if False in no_overlap_Wind.values:
                Wind = pd.concat((Wind, Wind_tmp), axis=1)

            # Load in situ EE data
            print("reading ... " + filename_ee)
            ee_obs_tmp = pd.read_csv(filename_ee)
            if ee_obs_tmp.shape[1] == 1:
                print("EE csv file with semicolon ...")
                ee_obs_tmp = pd.read_csv(filename_ee, sep=';')
            ee_obs_tmp.columns = ['time', site_ID]
            ee_obs_tmp['time'] = pd.to_datetime(ee_obs_tmp['time'])
            ee_obs_tmp = ee_obs_tmp.set_index('time')

            # Load in situ RN data
            print("reading ... " + filename_rn)
            rn_obs_tmp = pd.read_csv(filename_rn)
            if rn_obs_tmp.shape[1] == 1:
                print("RN csv file with semicolon ...")
                rn_obs_tmp = pd.read_csv(filename_rn, sep=';')
            rn_obs_tmp.columns = ['time', site_ID]
            rn_obs_tmp['time'] = pd.to_datetime(rn_obs_tmp['time'])
            rn_obs_tmp = rn_obs_tmp.set_index('time')

            # Load in situ Tair data
            print("reading ... " + filename_Tair)
            Tair_obs_tmp = pd.read_csv(filename_Tair)
            if Tair_obs_tmp.shape[1] == 1:
                print("Tair csv file with semicolon ...")
                Tair_obs_tmp = pd.read_csv(filename_Tair, sep=';')
            Tair_obs_tmp.columns = ['time', site_ID]
            Tair_obs_tmp['time'] = pd.to_datetime(Tair_obs_tmp['time'])
            Tair_obs_tmp = Tair_obs_tmp.set_index('time')

            # Load in situ LE data
            print("reading ... " + filename_le)
            le_obs_tmp = pd.read_csv(filename_le)
            if le_obs_tmp.shape[1] == 1:
                print("LE csv file with semicolon ...")
                le_obs_tmp = pd.read_csv(filename_le, sep=';')
            le_obs_tmp.columns = ['time', site_ID]
            le_obs_tmp['time'] = pd.to_datetime(le_obs_tmp['time'])
            le_obs_tmp = le_obs_tmp.set_index('time')

            # Load model EE data.
            try:
                le_mod_tmp = io.read_ts('lhflux', lon, lat, lonlat=True)
                le_mod_tmp = pd.DataFrame(le_mod_tmp)
                swi_mod_tmp = io.read_ts('SWdown', lon, lat, lonlat=True)
                lwi_mod_tmp = io.read_ts('LWdown', lon, lat, lonlat=True)
                swo_mod_tmp = io.read_ts('swup', lon, lat, lonlat=True)
                lwo_mod_tmp = io.read_ts('lwup', lon, lat, lonlat=True)
                rn_mod_tmp = pd.concat([lwi_mod_tmp, swi_mod_tmp, swo_mod_tmp, lwo_mod_tmp], axis=1)
                rn_mod_tmp = rn_mod_tmp['LWdown'] + rn_mod_tmp['SWdown'] - rn_mod_tmp['swup'] - rn_mod_tmp['lwup']
                rn_mod_tmp = pd.DataFrame(rn_mod_tmp)
                ee_mod_tmp = le_mod_tmp['lhflux'] / rn_mod_tmp[0]
                ee_mod_tmp = pd.DataFrame(ee_mod_tmp)

                # Check if overlapping data in ee
                df_check_ee = pd.concat((ee_obs_tmp, ee_mod_tmp), axis=1)
                no_overlap_ee = pd.isnull(df_check_ee).any(axis=1)

                if False in no_overlap_ee.values:
                    ee_obs = pd.concat((ee_obs, ee_obs_tmp), axis=1)
                    ee_mod = pd.concat((ee_mod, ee_mod_tmp), axis=1)

                # Check if overlapping data in rn
                df_check_rn = pd.concat((rn_obs_tmp, rn_mod_tmp), axis=1)
                no_overlap_rn = pd.isnull(df_check_rn).any(axis=1)

                if False in no_overlap_rn.values:
                    rn_obs = pd.concat((rn_obs, rn_obs_tmp), axis=1)
                    rn_mod = pd.concat((rn_mod, rn_mod_tmp), axis=1)

                # Check if overlapping data in le
                df_check_le = pd.concat((le_obs_tmp, le_mod_tmp), axis=1)
                no_overlap_le = pd.isnull(df_check_le).any(axis=1)

                if False in no_overlap_le.values:
                    le_obs = pd.concat((le_obs, le_obs_tmp), axis=1)
                    le_mod = pd.concat((le_mod, le_mod_tmp), axis=1)
            except:
                print('not all model data is available')

            # Load in situ BR data
            print("reading ... " + filename_br)
            br_obs_tmp = pd.read_csv(filename_br)
            if br_obs_tmp.shape[1] == 1:
                print("BR csv file with semicolon ...")
                br_obs_tmp = pd.read_csv(filename_br, sep=';')
            br_obs_tmp.columns = ['time', site_ID]
            br_obs_tmp['time'] = pd.to_datetime(br_obs_tmp['time'])
            br_obs_tmp = br_obs_tmp.set_index('time')

            # Load in situ sh data
            print("reading ... " + filename_sh)
            sh_obs_tmp = pd.read_csv(filename_sh)
            if sh_obs_tmp.shape[1] == 1:
                print("SH csv file with semicolon ...")
                sh_obs_tmp = pd.read_csv(filename_sh, sep=';')
            sh_obs_tmp.columns = ['time', site_ID]
            sh_obs_tmp['time'] = pd.to_datetime(sh_obs_tmp['time'])
            sh_obs_tmp = sh_obs_tmp.set_index('time')

            # Load model BR data.
            le_mod_tmp = io.read_ts('lhflux', lon, lat, lonlat=True)
            le_mod_tmp = pd.DataFrame(le_mod_tmp)
            sh_mod_tmp = io.read_ts('shflux', lon, lat, lonlat=True)
            sh_mod_tmp = pd.DataFrame(sh_mod_tmp)
            br_mod_tmp = sh_mod_tmp['shflux'] / le_mod_tmp['lhflux']

            # Check if overlapping data in br.
            df_check_br = pd.concat((br_obs_tmp, br_mod_tmp), axis=1)
            no_overlap_br = pd.isnull(df_check_br).any(axis=1)

            if False in no_overlap_br.values:
                br_obs = pd.concat((br_obs, br_obs_tmp), axis=1)
                br_mod = pd.concat((br_mod, br_mod_tmp), axis=1)

            # Check if overlapping data in sh.
            df_check_sh = pd.concat((sh_obs_tmp, sh_mod_tmp), axis=1)
            no_overlap_sh = pd.isnull(df_check_sh).any(axis=1)

            if False in no_overlap_sh.values:
                sh_obs = pd.concat((sh_obs, sh_obs_tmp), axis=1)
                sh_mod = pd.concat((sh_mod, sh_mod_tmp), axis=1)

            # Load in situ sh data
            print("reading ... " + filename_wtd)
            wtd_obs_tmp = pd.read_csv(filename_wtd)
            if wtd_obs_tmp.shape[1] == 1:
                print("WTD csv file with semicolon ...")
                wtd_obs_tmp = pd.read_csv(filename_wtd, sep=';')
            wtd_obs_tmp.columns = ['time', site_ID]
            wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'])
            wtd_obs_tmp = wtd_obs_tmp.set_index('time')

            # Check if overlapping data in sh.
            df_check_wtd = pd.concat((wtd_obs_tmp, et_mod_tmp), axis=1)
            no_overlap_wtd = pd.isnull(df_check_wtd).any(axis=1)

            if False in no_overlap_wtd.values:
                wtd_obs = pd.concat((wtd_obs, wtd_obs_tmp), axis=1)

            # Load model ETpot calculation data
            ghflux_mod_tmp = io.read_ts('ghflux', lon, lat, lonlat=True)
            Psurf_mod_tmp = io.read_ts('Psurf', lon, lat, lonlat=True)
            Tair_mod_tmp = io.read_ts('Tair', lon, lat, lonlat=True)

            df_check_ETo = pd.concat((et_obs_tmp, ghflux_mod_tmp), axis=1)
            no_overlap_ghflux = pd.isnull(df_check_ETo).any(axis=1)

            if False in no_overlap_ghflux.values:
                ghflux_mod = pd.concat((ghflux_mod, ghflux_mod_tmp), axis=1)
                Psurf_mod = pd.concat((Psurf_mod, Psurf_mod_tmp), axis=1)

            # Check if overlapping data in Tair.
            df_check_Tair = pd.concat((Tair_obs_tmp, Tair_mod_tmp), axis=1)
            no_overlap_Tair = pd.isnull(df_check_Tair).any(axis=1)

            if False in no_overlap_Tair.values:
                Tair_obs = pd.concat((Tair_obs, Tair_obs_tmp), axis=1)
                Tair_mod = pd.concat((Tair_mod, Tair_mod_tmp), axis=1)

            # Load in situ vpd data
            print("reading ... " + filename_vpd)
            vpd_obs_tmp = pd.read_csv(filename_vpd)
            if vpd_obs_tmp.shape[1] == 1:
                print("vpd csv file with semicolon ...")
                vpd_obs_tmp = pd.read_csv(filename_vpd, sep=';')
            vpd_obs_tmp.columns = ['time', site_ID]
            vpd_obs_tmp['time'] = pd.to_datetime(vpd_obs_tmp['time'])
            vpd_obs_tmp = vpd_obs_tmp.set_index('time')

            # Check if overlapping data in vpd.
            df_check_vpd = pd.concat((vpd_obs_tmp, et_mod_tmp), axis=1)
            no_overlap_vpd = pd.isnull(df_check_vpd).any(axis=1)

            if False in no_overlap_vpd.values:
                vpd_obs = pd.concat((vpd_obs, vpd_obs_tmp), axis=1)

    et_mod = pd.DataFrame(et_mod)
    et_mod.columns = et_obs.columns

    eveg_mod = pd.DataFrame(eveg_mod)
    eveg_mod.columns = et_obs.columns

    esoi_mod = pd.DataFrame(esoi_mod)
    esoi_mod.columns = et_obs.columns

    eint_mod = pd.DataFrame(eint_mod)
    eint_mod.columns = et_obs.columns

    zbar_mod = pd.DataFrame(zbar_mod)
    zbar_mod.columns = et_obs.columns

    et_mod_copy = et_mod.copy()
    for col in et_mod_copy:
        et_mod_copy[col].values[:] = 0

    try:
        ee_mod = pd.DataFrame(ee_mod)
        ee_mod.columns = ee_obs.columns
    except:
        ee_mod = et_mod_copy

    br_mod = pd.DataFrame(br_mod)
    br_mod.columns = br_obs.columns

    try:
        rn_mod = pd.DataFrame(rn_mod)
        rn_mod.columns = rn_obs.columns
    except:
        rn_mod = et_mod_copy

    try:
        sh_mod = pd.DataFrame(sh_mod)
        sh_mod.columns = sh_obs.columns
    except:
        rn_mod = et_mod_copy

    le_mod = pd.DataFrame(le_mod)
    le_mod.columns = le_obs.columns

    wtd_obs = pd.DataFrame(wtd_obs)

    ghflux_mod = pd.DataFrame(ghflux_mod)
    ghflux_mod.columns = et_obs.columns
    Psurf_mod = pd.DataFrame(Psurf_mod)
    Psurf_mod.columns = et_obs.columns
    try:
        Tair_mod = pd.DataFrame(Tair_mod)
        Tair_mod.columns = Tair_obs.columns
    except:
        Tair_mod = et_mod_copy

    AR1 = pd.DataFrame(AR1)
    AR1.columns = wtd_obs.columns
    AR2 = pd.DataFrame(AR2)
    AR2.columns = wtd_obs.columns
    AR4 = pd.DataFrame(AR4)
    AR4.columns = wtd_obs.columns
    sfmc = pd.DataFrame(sfmc)
    sfmc.columns = wtd_obs.columns
    rzmc = pd.DataFrame(rzmc)
    rzmc.columns = wtd_obs.columns
    sfexc = pd.DataFrame(srfexc)
    sfexc.columns = wtd_obs.columns
    rzexc = pd.DataFrame(rzexc)
    rzexc.columns = wtd_obs.columns
    catdef = pd.DataFrame(catdef)
    catdef.columns = wtd_obs.columns
    Qair = pd.DataFrame(Qair)
    Qair.columns = wtd_obs.columns
    Wind = pd.DataFrame(Wind)
    Wind.columns = wtd_obs.columns

    return et_obs, et_mod, ee_obs, ee_mod, br_obs, br_mod, rn_obs, rn_mod, sh_obs, sh_mod, le_obs, le_mod, zbar_mod, eveg_mod, esoi_mod, eint_mod, wtd_obs, ghflux_mod, Psurf_mod, Tair_mod, Tair_obs, AR1, AR2, AR4, sfmc, rzmc, srfexc, rzexc, catdef, Qair, vpd_obs, Wind


def read_wtd_data(insitu_path, mastertable_filename, exp, domain, root):
    # read in in situ data of water table depth time series and master table in csv format
    # read out modeled time series for nearest grid cell
    #
    # Input:
    # insitu_path: basic path for all peatland in situ data 
    # exp: experiment name
    # domain: domain name
    # root: root path of simulation experiment
    # 
    # Output:
    # wtd_obs
    # wtd_mod
    # precip_obs
    # precip_mod

    """
    Function to read
    - in situ data of water table depth time series and master table in csv format
    - modeled time series for location of in situ data (corresponding grid cell is selected
    ! poros threshold 0.6 --> obs-mod pairs only for peat grid cells !

    Parameters
    ----------
    insitu_path : str
        basic path for all peatland in situ data
    exp : str
        experiment name
    domain : str
        domain name
    root : str
        root path of simulation experiment

    Returns
    -------
    data : pd.DataFrame
        Content of the fortran binary file
    wtd_obs : pd.DataFrame
        wtd in situ time series
    wtd_mod : pd.DataFrame
        wtd model time series
    precip_obs : pd.DataFrame
        precip time series (meteo station at site, only sipalaga at the moment)
    """

    filenames = find_files(insitu_path, mastertable_filename)
    if isinstance(find_files(insitu_path, mastertable_filename), str):
        master_table = pd.read_csv(filenames, sep=';')
        if master_table.shape[1] == 1:
            master_table = pd.read_csv(filenames, sep=',')
    else:
        for filename in filenames:
            if filename.endswith('csv'):
                master_table = pd.read_csv(filename, sep=';')
                if master_table.shape[1] == 1:
                    master_table = pd.read_csv(filenames, sep=',')
                continue
            else:
                logging.warning("some files, maybe swp files, exist that start with master table searchstring !")

    io = LDAS_io('daily', exp=exp, domain=domain, root=root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)

    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values

    first_site = True
    whitelist_coordinates = []

    for i, site_ID in enumerate(master_table.iloc[:, 0]):

        # If site is on the blacklist (contains bad data), or if "comparison" =  0, then don include that site in the dataframe.
        if master_table['comparison_yes'][i] == 0:
            continue

 #sebastian added       #to select only the drained of only the natural ones to calculate skill metrics!
        if master_table['drained_U=uncertain'][i] == 'N':
            continue

        #if master_table['drained_U=uncertain'][i] == 'U':
        #    continue

        # Get lat lon from master table for site.
        lon = master_table['lon'][i]

        # sebastian added   # to skip a site if there is no lon or lat when all are used instead of only the comparison =1 ones.
        # if np.isnan(lon):
        #    continue

        lat = master_table['lat'][i]

        # Get porosity for site lon lat.
        # Get M09 rowcol with data.
        col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
        # Get poros for col row. + Check whether site is in domain, if not 'continue' with next site
        try:
            siteporos = poros[row, col]
        except:
            print(site_ID + " not in domain.")
            continue

        if siteporos <= 0.60:
            # If the porosity of the site is 0.6 or lower the site is not classified as peatland in the model.
            print(site_ID + " not on a peatland grid cell.")
            continue

        folder_p = insitu_path + '/Rainfall/'

        try:
            if isinstance(find_files(insitu_path, site_ID), str):
                filename_wtd = find_files(insitu_path, site_ID)
                print(filename_wtd)
            else:
                flist = find_files(insitu_path, site_ID)
                for f in flist:
                    # 000_all_wtd added for northern peatlands, data management to be harmonized
                    if ((f.count('aily') >= 1) or (f.count('000_all_wtd/') >= 1)) and f.endswith('csv'):
                        filename_wtd = f
        except:
            print(site_ID + " does not have a WTD csv file with data.")
            continue
        # check for empty path
        if len(filename_wtd) < 10 or filename_wtd.endswith('csv') != True or filename_wtd.count('Rainfall') >= 1:
            print("checking ... " + site_ID + " " + filename_wtd)
            print("some other reason for no data for " + site_ID + " in " + filename_wtd)
            continue

        # same for precipitation data
        try:
            if isinstance(find_files(folder_p, site_ID), str):
                filename_precip = find_files(folder_p, site_ID)
                print(filename_precip)
            else:
                flist = find_files(folder_p, site_ID)
                for f in flist:
                    if f.count('aily') >= 1:
                        filename_precip = f
                # check for empty path
            if len(filename_precip) < 10 or filename_precip.endswith('csv') != True or filename_precip.count(
                    'WTD') >= 1:
                print("checking ... " + site_ID + " " + filename_precip)
                print("some other reason for no data for " + site_ID + " in " + filename_precip)
                continue
        except:
            filename_precip = ''
            print(site_ID + " does not have a precipitation csv file with data.")

        # to obtain the tile coordinates for whitlist runs

        # j = io.grid.lonlat2tilenum(lon,lat)
        # site_coordinate = io.grid.tilecoord['tile_id'][j]
        # whitelist_coordinates.append(site_coordinate)

        # cumbersome first site, next sides ... to be simplified ...
        if first_site == True:
            # Load in situ data.
            # wtd:
            print("reading ... " + filename_wtd)
            wtd_obs = pd.read_csv(filename_wtd)
            if wtd_obs.shape[1] == 1:
                print("WTD csv file with semicolon ...")
                wtd_obs = pd.read_csv(filename_wtd, sep=';')
            wtd_obs.columns = ['time', site_ID]
            wtd_obs['time'] = pd.to_datetime(wtd_obs['time'])
            wtd_obs = wtd_obs.set_index('time')
            try:
                # Precipitation:
                precip_obs = pd.read_csv(filename_precip)
                precip_obs.columns = ['time', site_ID]
                precip_obs['time'] = pd.to_datetime(precip_obs['time'])
                precip_obs = precip_obs.set_index('time')
            except:
                precip_obs = wtd_obs.copy()
                precip_obs[:] = -9999

            # Load model wtd data.
            wtd_mod = io.read_ts('zbar', lon, lat, lonlat=True)
            # Check if overlapping data.
            df_check = pd.concat((wtd_obs, wtd_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False

            # Load model precip data.
            try:
                precip_mod = io.read_ts('Rainf', lon, lat, lonlat=True)
            except:
                precip_mod = wtd_mod.deepcopy()
                precip_mod[site_ID] = -15

            # Check if overlapping data.
            df_check = pd.concat((precip_obs, precip_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False

            # Load model sfmc rzmc.
            sfmc_mod = io.read_ts('sfmc', lon, lat, lonlat=True)
            rzmc_mod = io.read_ts('rzmc', lon, lat, lonlat=True)
            srfexc = io.read_ts('srfexc', lon, lat, lonlat=True)
            rzexc = io.read_ts('rzexc', lon, lat, lonlat=True)
            catdef = io.read_ts('catdef', lon, lat, lonlat=True)

            df_check = pd.concat((wtd_obs, sfmc_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False
            df_check = pd.concat((wtd_obs, rzmc_mod), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False
            # to end firs_site
            first_site = False

        else:
            # Load in situ data.
            # wtd:
            print("reading ... " + filename_wtd)
            wtd_obs_tmp = pd.read_csv(filename_wtd, sep=',')
            if wtd_obs_tmp.shape[1] == 1:
                print("WTD csv file with semicolon ...")
                wtd_obs_tmp = pd.read_csv(filename_wtd, sep=';')
            wtd_obs_tmp.columns = ['time', site_ID]
            # MB: pandas failed to recognice datetime pattern automatically ..
            if wtd_obs_tmp.loc[0, 'time'].__len__() <= 11:
                if wtd_obs_tmp.loc[0, 'time'].split('/')[0].__len__() <= 2:
                    # Year in the end
                    try:
                        wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'], format='%d/%m/%Y')
                    except:
                        wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'], format='%m/%d/%Y')
                else:
                    wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'])
            else:
                if wtd_obs_tmp.loc[0, 'time'].split('/')[0].__len__() <= 2:
                    # Year in the end
                    wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'], format='%d/%m/%Y %H:%M')
                else:
                    wtd_obs_tmp['time'] = pd.to_datetime(wtd_obs_tmp['time'])

            wtd_obs_tmp = wtd_obs_tmp.set_index('time')
            try:
                # Precipitation:
                precip_obs_tmp = pd.read_csv(filename_precip)
                precip_obs_tmp.columns = ['time', site_ID]
                precip_obs_tmp['time'] = pd.to_datetime(precip_obs_tmp['time'])
                precip_obs_tmp = precip_obs_tmp.set_index('time')
            except:
                precip_obs_tmp = wtd_obs_tmp.copy()
                precip_obs_tmp[:] = -9999

            # temporarly added this to go around an error
            # precip_obs_tmp[site_ID] = -1

            # Load model data.

            wtd_mod_tmp = io.read_ts('zbar', lon, lat, lonlat=True)

            # Check if overlaping data.
            # if site_ID.startswith('Taka1_Palangkraya_da'):
            #    print('s')
            print(site_ID)
            df_check_wtd = pd.concat((wtd_obs_tmp, wtd_mod_tmp), axis=1)
            no_overlap_wtd = pd.isnull(df_check_wtd).any(axis=1)

            try:
                # Load model precip data.
                precip_mod_tmp = io.read_ts('Rainf', lon, lat, lonlat=True)
            except:
                precip_mod_tmp = wtd_mod_tmp.copy()
                precip_mod_tmp[site_ID] = -15

            # Check if overlapping data.
            # df_check = pd.concat((precip_obs_tmp, precip_mod_tmp), axis=1)
            df_check = pd.concat((precip_obs_tmp.loc[~precip_obs_tmp.index.duplicated(keep='first')], precip_mod_tmp),
                                 axis=1)
            no_overlap_precip = pd.isnull(df_check).any(axis=1)

            # Load model sfmc rzmc.
            sfmc_mod_tmp = io.read_ts('sfmc', lon, lat, lonlat=True)
            rzmc_mod_tmp = io.read_ts('rzmc', lon, lat, lonlat=True)
            srfexc_tmp = io.read_ts('srfexc', lon, lat, lonlat=True)
            rzexc_tmp = io.read_ts('rzexc', lon, lat, lonlat=True)
            catdef_tmp = io.read_ts('catdef', lon, lat, lonlat=True)

            df_check = pd.concat((wtd_obs_tmp, sfmc_mod_tmp), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False
            df_check = pd.concat((wtd_obs_tmp, rzmc_mod_tmp), axis=1)
            no_overlap = pd.isnull(df_check).any(axis=1)
            if False in no_overlap.values:
                first_site = False

            if False in no_overlap_wtd.values:
                first_site = False

            if False in no_overlap_wtd.values:
                wtd_obs = pd.concat((wtd_obs, wtd_obs_tmp), axis=1)
                wtd_mod = pd.concat((wtd_mod, wtd_mod_tmp), axis=1)

            if False in no_overlap_precip.values:
                # precip_obs = pd.concat((precip_obs, precip_obs_tmp), axis=1)
                precip_obs = pd.concat((precip_obs_tmp.loc[~precip_obs_tmp.index.duplicated(keep='first')], precip_obs),
                                       axis=1)
                precip_mod = pd.concat((precip_mod_tmp.loc[~precip_mod_tmp.index.duplicated(keep='first')], precip_mod),
                                       axis=1)

            if False in no_overlap.values:
                rzmc_mod = pd.concat((rzmc_mod, rzmc_mod_tmp), axis=1)
                sfmc_mod = pd.concat((sfmc_mod, sfmc_mod_tmp), axis=1)
                srfexc = pd.concat((srfexc, srfexc_tmp), axis=1)
                rzexc = pd.concat((rzexc, rzexc_tmp), axis=1)
                catdef = pd.concat((catdef, catdef_tmp), axis=1)

    wtd_mod = pd.DataFrame(wtd_mod)
    wtd_mod.columns = wtd_obs.columns

    precip_mod = pd.DataFrame(precip_mod)
    precip_mod.columns = precip_obs.columns

    rzmc_mod = pd.DataFrame(rzmc_mod)
    rzmc_mod.columns = wtd_obs.columns

    sfmc_mod = pd.DataFrame(sfmc_mod)
    sfmc_mod.columns = wtd_obs.columns

    srfexc = pd.DataFrame(srfexc)
    srfexc.columns = wtd_obs.columns
    rzexc = pd.DataFrame(rzexc)
    rzexc.columns = wtd_obs.columns
    catdef = pd.DataFrame(catdef)
    catdef.columns = wtd_obs.columns

    # Sebastian added   #to create the csv files for Susan Page
    # wtd_mod_export_csv = wtd_mod.to_csv (r'/data/leuven/317/vsc31786/peatland_data/tropics/WTD/Sipalaga/processed/WTD/model_WTD/model_WTD_Natural.csv', index = True, header=True)
    # whitelist_coordinates = pd.DataFrame(whitelist_coordinates)
    # whitelist_coordinates.to_csv(r'/data/leuven/317/vsc31786/projects/TROPICS/WHITELIST_M09_PEATCLSMTN_v01/whitelist/SMAP_EASEv2_M09/whitelisttilescongo.csv',index=False, header=False)

    return wtd_obs, wtd_mod, precip_obs, precip_mod, sfmc_mod, rzmc_mod, srfexc, rzexc, catdef


#################################################################
#################################################################
# functions used for Remote Sensing of Environment data assimilation paper
# to be cleaned up after publication

def ncfile_init(fname, lats, lons, species, tags):
    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['species', 'lat', 'lon']
    dimvals = [species, lats, lons]
    chunksizes = [1, len(lats), len(lons)]
    dtypes = ['uint8', 'float32', 'float32']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=(dim,), chunksizes=(chunksize,), zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        if (tag.find('innov') != -1) | (tag.find('obsvar') != -1) | (tag.find('fcstvar') != -1) | (
                tag.find('R_mean') != -1) | (tag.find('obs_mean') != -1):
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[1:], chunksizes=chunksizes[1:], fill_value=-9999.,
                              zlib=True)

    ds.variables['lon'].setncatts({'long_name': 'longitude', 'units': 'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude', 'units': 'degrees_north'})

    return ds


def ncfile_init_incr(fname, dimensions, tags):
    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat', 'lon']
    dimvals = [dimensions['lat'], dimensions['lon']]
    chunksizes = [len(dimvals[0]), len(dimvals[1])]
    dtypes = ['float32', 'float32']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=(dim,), chunksizes=(chunksize,), zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)

    ds.variables['lon'].setncatts({'long_name': 'longitude', 'units': 'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude', 'units': 'degrees_north'})

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
        if dim in ['lon', 'lat']:
            chunksize = len(dimensions[dim])
        else:
            chunksize = 1
        chunksizes.append(chunksize)

        dtype = dimensions[dim].dtype
        ds.createDimension(dim, len(dimensions[dim]))
        ds.createVariable(dim, dtype,
                          dimensions=(dim,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds.variables[dim][:] = dimensions[dim]

    # Coordinate attributes following CF-conventions
    if "time" in ds.variables:
        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})
    ds.variables['lon'].setncatts({'long_name': 'longitude',
                                   'units': 'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude',
                                   'units': 'degrees_north'})

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

    dims = ['lat', 'lon', 'pentad', 'AD']
    dimvals = [lats, lons, pentad, AD]
    chunksizes = [len(lats), len(lons), 1, 1]
    dtypes = ['float32', 'float32', 'uint8', 'uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim] = dimval

    for tag in tags:
        ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)

    return ds


def ncfile_init_multiruns(fname, lats, lons, runs, species, tags):
    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat', 'lon', 'run', 'species']
    dimvals = [lats, lons, runs, species]
    chunksizes = [len(lats), len(lons), 1, 1]
    dtypes = ['float32', 'float32', 'uint8', 'uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim] = dimval

    for tag in tags:
        if tag.find('innov') != -1:
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[0:-1], chunksizes=chunksizes[0:-1], fill_value=-9999.,
                              zlib=True)

    return ds


def bin2nc_scaling(exp, domain, root, outputpath, scalepath='', scalename=''):
    angle = 40
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'scaling.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('ObsFcstAna', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    pentad = np.arange(1, 74)
    AD = np.arange(0, 2)
    if scalepath == '':
        scalepath = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalepath'].split()[0]
        scalename = io.read_nml('ensupd')['ens_upd_inputs']['obs_param_nml'][20]['scalename'][5:].split()[0]

    print('scalepath')
    print(scalepath)
    print(scalename)
    tags = ['m_mod_H_%2i' % angle, 'm_mod_V_%2i' % angle, 'm_obs_H_%2i' % angle, 'm_obs_V_%2i' % angle]

    ds = ncfile_init_scaling(result_file, lats[:, 0], lons[0, :], pentad, AD, tags)

    for i_AscDes, AscDes in enumerate(list(["A", "D"])):
        for i_pentad in np.arange(1, 74):
            logging.info('pentad %i' % (i_pentad))
            fname = scalepath + scalename + '_' + AscDes + '_p' + "%02d" % (i_pentad,) + '.bin'
            tmp = io.read_scaling_parameters(fname=fname)

            res = tmp[['lon', 'lat', 'm_mod_H_%2i' % angle, 'm_mod_V_%2i' % angle, 'm_obs_H_%2i' % angle,
                       'm_obs_V_%2i' % angle]]
            res.replace(-9999., np.nan, inplace=True)

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_mod_H_%2i' % angle][:, :, i_pentad - 1, i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_mod_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_mod_V_%2i' % angle][:, :, i_pentad - 1, i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_H_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_obs_H_%2i' % angle][:, :, i_pentad - 1, i_AscDes] = data

            img = np.full(lons.shape, np.nan)
            img[tc.j_indg.values, tc.i_indg.values] = res['m_obs_V_%2i' % angle].values
            img = obs_M09_to_M36(img)
            data = np.ma.masked_invalid(img)
            ds['m_obs_V_%2i' % angle][:, :, i_pentad - 1, i_AscDes] = data

    ds.close()


def filter_diagnostics_evaluation(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)
    # cvars = DA_innov.timeseries.data_vars
    # cond = DA_innov.timeseries['obs_assim']==-1
    # tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    # np.place(tmp, np.isnan(tmp), 1.)
    # np.place(tmp, tmp==False, 0.)
    # np.place(tmp, tmp==-1, 1.)
    # DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    # cvars = ['obs_obs']
    # for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    # del cond1
    try:
        DA_incr = LDAS_io('incr', exp, domain, root)
        DA = 1
    except:
        DA = 0

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
    ars1 = np.full(lons.shape, np.nan)
    ars1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['ars1'].values
    ars2 = np.full(lons.shape, np.nan)
    ars2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['ars2'].values
    ars3 = np.full(lons.shape, np.nan)
    ars3[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['ars3'].values

    if DA == 1:
        tags = ['innov_mean', 'innov_var',
                'norm_innov_mean', 'norm_innov_var',
                'obs_mean',
                'obsvar_mean', 'fcstvar_mean',
                'n_valid_innov',
                'incr_catdef_mean', 'incr_catdef_var',
                'incr_rzexc_mean', 'incr_rzexc_var',
                'incr_srfexc_mean', 'incr_srfexc_var',
                'n_valid_incr']
    else:
        tags = ['innov_mean', 'innov_var',
                'norm_innov_mean', 'norm_innov_var',
                'obs_mean',
                'obsvar_mean', 'fcstvar_mean',
                'n_valid_innov']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    # lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    if DA == 1:
        tmp = DA_incr.timeseries['srfexc'][:, :, :].values
        np.place(tmp, tmp == 0., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    mask_scaling_old = 0
    if mask_scaling_old == 1:
        mask_innov = np.full_like(DA_innov.timeseries['obs_obs'][:, 0, :, :].values, False)
        if DA == 1:
            mask_incr = np.full_like(DA_incr.timeseries['catdef'][:, :, :].values, False)
        n_innov_nyr = np.zeros([DA_innov.timeseries.dims['lat'], DA_innov.timeseries.dims['lon']])
        for j, yr in enumerate(np.unique(pd.Series(DA_innov.timeseries['time']).dt.year)):
            n_innov_yr = np.zeros([DA_innov.timeseries.dims['lat'], DA_innov.timeseries.dims['lon']])
            for i_spc, spc in enumerate(species):
                logging.info('species %i' % (i_spc))
                obs_obs_tmp = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values
                # np.place(obs_obs_tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
                tmp = obs_obs_tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :].values
                np.place(tmp, tmp == 0., np.nan)
                np.place(tmp, tmp == -9999., np.nan)
                np.place(tmp, ~np.isnan(tmp), 1.)
                n_innov_tmp = np.nansum(tmp[pd.Series(DA_innov.timeseries['time']).dt.year == yr, :, :], axis=0)
                np.place(n_innov_tmp, n_innov_tmp == 0., np.nan)
                n_innov_tmp = obs_M09_to_M36(n_innov_tmp)
                np.place(n_innov_tmp, np.isnan(n_innov_tmp), 0.)
                n_innov_yr = n_innov_yr + n_innov_tmp
            n_innov_yr = n_innov_yr / 2.0  # H-pol and V-pol count as one obs
            cond = (n_innov_yr < 3) & (n_innov_yr > 0)
            n_innov_nyr[n_innov_yr >= 3] = n_innov_nyr[n_innov_yr >= 3] + 1
            cmask = np.repeat(cond[np.newaxis, :, :],
                              np.where((pd.Series(DA_innov.timeseries['time']).dt.year == yr))[0].__len__(), axis=0)
            mask_innov[pd.Series(DA_innov.timeseries['time']).dt.year == yr, :, :] = cmask
            if DA == 1:
                cmask = np.repeat(cond[np.newaxis, :, :],
                                  np.where((pd.Series(DA_incr.timeseries['time']).dt.year == yr))[0].__len__(), axis=0)
                mask_incr[pd.Series(DA_incr.timeseries['time']).dt.year == yr, :, :] = cmask
        del cmask, n_innov_yr, cond, tmp, obs_obs_tmp
        cond2 = n_innov_nyr <= 1
        cmask = np.repeat(cond2[np.newaxis, :, :],
                          np.where((pd.Series(DA_innov.timeseries['time']).dt.year != -9999))[0].__len__(), axis=0)
        np.place(mask_innov, cmask, True)
        if DA == 1:
            cmask = np.repeat(cond2[np.newaxis, :, :],
                              np.where((pd.Series(DA_incr.timeseries['time']).dt.year != -9999))[0].__len__(), axis=0)
            np.place(mask_incr, cmask, True)

    # DA_innov2 = LDAS_io('ObsFcstAna',exp+'_DA', domain, root)
    # if DA==0:
    #    OL_mask = np.in1d(DA_innov.timeseries['time'].values, DA_innov2.timeseries['time'].values)
    #    DA_mask = np.in1d(DA_innov2.timeseries['time'].values, DA_innov.timeseries['time'].values)

    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        tmp = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values
        # np.place(tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
        # if DA==0:
        #    # take valid data mask from what has been assimilated in the _DA run
        #    tmp = np.where(DA_innov2.timeseries['obs_assim'][DA_mask,i_spc,:,:]!=-1,np.nan,tmp)
        # else:
        if DA == 1:
            tmp = np.where(DA_innov.timeseries['obs_assim'][:, i_spc, :, :] != -1, np.nan, tmp)
        if mask_scaling_old == 1:
            np.place(tmp, mask_innov, np.nan)
        ds['obs_mean'][i_spc, :, :] = (DA_innov.timeseries['obs_obs'][:, i_spc, :, :]).mean(dim='time',
                                                                                            skipna=True).values
        ds['obsvar_mean'][i_spc, :, :] = (DA_innov.timeseries['obs_obsvar'][:, i_spc, :, :]).mean(dim='time',
                                                                                                  skipna=True).values
        ds['fcstvar_mean'][i_spc, :, :] = (DA_innov.timeseries['obs_fcstvar'][:, i_spc, :, :]).mean(dim='time',
                                                                                                    skipna=True).values
        # if DA==0:
        #    ds['innov_mean'][i_spc,:,:] = (tmp - DA_innov.timeseries['obs_fcst'][OL_mask,i_spc,:,:]).mean(dim='time',skipna=True).values
        #    ds['innov_var'][i_spc,:,:] = (tmp - DA_innov.timeseries['obs_fcst'][OL_mask,i_spc,:,:]).var(dim='time',skipna=True).values
        #    ds['norm_innov_mean'][i_spc,:,:] = ((tmp - DA_innov.timeseries['obs_fcst'][OL_mask,i_spc,:,:]) /
        #                                        np.sqrt(DA_innov.timeseries['obs_obsvar'][OL_mask,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][OL_mask,i_spc,:,:])).mean(dim='time',skipna=True).values
        #    ds['norm_innov_var'][i_spc,:,:] = ((tmp - DA_innov.timeseries['obs_fcst'][OL_mask,i_spc,:,:]) /
        #                                       np.sqrt(DA_innov.timeseries['obs_obsvar'][OL_mask,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][OL_mask,i_spc,:,:])).var(dim='time',skipna=True).values
        # else:
        ds['innov_mean'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]).mean(dim='time',
                                                                                                     skipna=True).values
        ds['innov_var'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]).var(dim='time',
                                                                                                   skipna=True).values
        ds['norm_innov_mean'][i_spc, :, :] = ((tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]) /
                                              np.sqrt(DA_innov.timeseries['obs_obsvar'][:, i_spc, :, :] +
                                                      DA_innov.timeseries['obs_fcstvar'][:, i_spc, :, :])).mean(
            dim='time', skipna=True).values
        ds['norm_innov_var'][i_spc, :, :] = ((tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]) /
                                             np.sqrt(DA_innov.timeseries['obs_obsvar'][:, i_spc, :, :] +
                                                     DA_innov.timeseries['obs_fcstvar'][:, i_spc, :, :])).var(
            dim='time', skipna=True).values
        np.place(tmp, tmp == 0., np.nan)
        np.place(tmp, tmp == -9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc, :, :] = tmp.sum(axis=0)
        del tmp

    if mask_scaling_old == 1:
        del mask_innov
    if DA == 1:
        np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
        np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
        np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
        if mask_scaling_old == 1:
            np.place(DA_incr.timeseries['catdef'].values, mask_incr, np.nan)
            np.place(DA_incr.timeseries['rzexc'].values, mask_incr, np.nan)
            np.place(DA_incr.timeseries['srfexc'].values, mask_incr, np.nan)
        # map from daily to hourly
        # cmap = np.floor(np.linspace(0,io.timeseries['time'].size,incr.size+1))
        ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time', skipna=True).values
        ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time', skipna=True).values
        ndays = io.timeseries['time'].size
        n3hourly = DA_incr.timeseries['time'].size
        # overwrite if exp does not contain CLSM
        if exp.find("_CLSM") < 0:
            for row in range(DA_incr.timeseries['catdef'].shape[1]):
                logging.info('PCLSM row %i' % (row))
                for col in range(DA_incr.timeseries['catdef'].shape[2]):
                    if poros[row, col] > 0.6:
                        catdef1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                               data=io.timeseries['catdef'][:, row, col].values)
                        catdef1 = catdef1.resample('3H').max().interpolate()
                        ar1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                           data=io.timeseries['ar1'][:, row, col].values)
                        ar1 = ar1.resample('3H').max().interpolate()
                        incr = pd.DataFrame(index=DA_incr.timeseries['time'].values,
                                            data=DA_incr.timeseries['catdef'][:, row, col].values)
                        # incr = DA_incr.timeseries['catdef'][:,row,col]
                        df = pd.concat((catdef1, incr, ar1), axis=1)
                        df.columns = ['catdef1', 'incr', 'ar1']
                        catdef0 = df['catdef1'] - df['incr']
                        catdef1 = df['catdef1']
                        ar1 = df['ar1']
                        incr = df['incr']
                        zbar1 = -1.0 * (np.sqrt(0.000001 + catdef1 / bf1[row, col]) - bf2[row, col])
                        zbar0 = -1.0 * (np.sqrt(0.000001 + catdef0 / bf1[row, col]) - bf2[row, col])
                        ar0 = (1.0 + ars1[row, col] * catdef0) / (
                                    1.0 + ars2[row, col] * catdef0 + ars3[row, col] * catdef0 ** 2.0)
                        incr_ar1 = -(zbar1 - zbar0) * 1000 * (0.5 * (
                                    ar0 + ar1))  # incr surface water storage but expressed as deficit change, so '-'
                        # data2 = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,0,0].values)
                        # cdata = pd.concat(data1,data2)
                        # zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                        # incr = DA_incr.timeseries['catdef'][:,row,col]
                        # catdef = np.full_like(incr,np.nan)
                        # for i in range(n3hourly):
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
        ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time', skipna=True).values
        ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time', skipna=True).values
        ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time', skipna=True).values
        ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time', skipna=True).values

    ds.close()


def filter_diagnostics_evaluation_OL_with_rescaled_obs(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics_OL_with_rescaled_obs.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)
    # cvars = DA_innov.timeseries.data_vars
    # cond = DA_innov.timeseries['obs_assim']==-1
    # tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    # np.place(tmp, np.isnan(tmp), 1.)
    # np.place(tmp, tmp==False, 0.)
    # np.place(tmp, tmp==-1, 1.)
    # DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    # cvars = ['obs_obs']
    # for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    # del cond1

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['innov_mean', 'innov_var',
            'n_valid_innov']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    # lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    # DA_innov2 = LDAS_io('ObsFcstAna',exp+'_DA', domain, root)
    # OL_mask = np.in1d(DA_innov.timeseries['time'].values, DA_innov2.timeseries['time'].values)
    # DA_mask = np.in1d(DA_innov2.timeseries['time'].values, DA_innov.timeseries['time'].values)

    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        tmp = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values
        # np.place(tmp,DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan)
        # tmp = np.where(DA_innov.timeseries['obs_assim'][:,i_spc,:,:]!=-1,np.nan,tmp)
        ds['innov_mean'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]).mean(dim='time',
                                                                                                     skipna=True).values
        ds['innov_var'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]).var(dim='time',
                                                                                                   skipna=True).values
        np.place(tmp, tmp == 0., np.nan)
        np.place(tmp, tmp == -9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc, :, :] = tmp.sum(axis=0)
    del tmp

    ds.close()


def filter_diagnostics_evaluation_R(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics_R_nonrescaled_obs.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['R_mean']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    # lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    # DA_innov2 = LDAS_io('ObsFcstAna',exp+'_DA', domain, root)
    # OL_mask = np.in1d(DA_innov.timeseries['time'].values, DA_innov2.timeseries['time'].values)
    # DA_mask = np.in1d(DA_innov2.timeseries['time'].values, DA_innov.timeseries['time'].values)

    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        for r, crow in enumerate(DA_innov.timeseries['lat']):
            print('row: ' + str(r))
            if np.isnan(np.nanmean(DA_innov.timeseries['obs_obs'][:, i_spc, r, :])):
                continue
            for c, ccol in enumerate(DA_innov.timeseries['lon']):
                a = DA_innov.timeseries['obs_obs'][:, i_spc, r, c]
                b = DA_innov.timeseries['obs_fcst'][:, i_spc, r, c]
                a = np.ma.masked_invalid(a)
                b = np.ma.masked_invalid(b)
                msk = (~a.mask & ~b.mask)
                R = np.ma.corrcoef(a[msk], b[msk])[0, 1]
                if np.ma.is_masked(R) == False:
                    ds['R_mean'][i_spc, r, c] = np.ma.corrcoef(a[msk], b[msk])[0, 1]

    ds.close()


def filter_diagnostics_evaluation_gs(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics_gs.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)
    # growing season
    gs = (pd.to_datetime(DA_innov.timeseries['time'].values).month > 7) & (
                pd.to_datetime(DA_innov.timeseries['time'].values).month < 10)
    # cvars = DA_innov.timeseries.data_vars
    # cond = DA_innov.timeseries['obs_assim']==-1
    # tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    # np.place(tmp, np.isnan(tmp), 1.)
    # np.place(tmp, tmp==False, 0.)
    # np.place(tmp, tmp==-1, 1.)
    # DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    # cvars = ['obs_obs']
    # for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    # del cond1
    DA_incr = LDAS_io('incr', exp, domain, root)
    gs_incr = (pd.to_datetime(DA_incr.timeseries['time'].values).month > 7) & (
                pd.to_datetime(DA_incr.timeseries['time'].values).month < 10)

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['innov_mean', 'innov_var',
            'norm_innov_mean', 'norm_innov_var',
            'obsvar_mean', 'fcstvar_mean',
            'n_valid_innov',
            'incr_catdef_mean', 'incr_catdef_var',
            'incr_rzexc_mean', 'incr_rzexc_var',
            'incr_srfexc_mean', 'incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    # lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][gs_incr, :, :].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    mask_scaling_old = 0
    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        tmp = DA_innov.timeseries['obs_obs'][gs, i_spc, :, :].values
        np.place(tmp, DA_innov.timeseries['obs_assim'][gs, i_spc, :, :] != -1, np.nan)
        if mask_scaling_old == 1:
            np.place(tmp, mask_innov, np.nan)
        ds['obsvar_mean'][i_spc, :, :] = (DA_innov.timeseries['obs_obsvar'][gs, i_spc, :, :]).mean(dim='time',
                                                                                                   skipna=True).values
        ds['fcstvar_mean'][i_spc, :, :] = (DA_innov.timeseries['obs_fcstvar'][gs, i_spc, :, :]).mean(dim='time',
                                                                                                     skipna=True).values
        ds['innov_mean'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][gs, i_spc, :, :]).mean(dim='time',
                                                                                                      skipna=True).values
        ds['innov_var'][i_spc, :, :] = (tmp - DA_innov.timeseries['obs_fcst'][gs, i_spc, :, :]).var(dim='time',
                                                                                                    skipna=True).values
        ds['norm_innov_mean'][i_spc, :, :] = ((tmp - DA_innov.timeseries['obs_fcst'][gs, i_spc, :, :]) /
                                              np.sqrt(DA_innov.timeseries['obs_obsvar'][gs, i_spc, :, :] +
                                                      DA_innov.timeseries['obs_fcstvar'][gs, i_spc, :, :])).mean(
            dim='time', skipna=True).values
        ds['norm_innov_var'][i_spc, :, :] = ((tmp - DA_innov.timeseries['obs_fcst'][gs, i_spc, :, :]) /
                                             np.sqrt(DA_innov.timeseries['obs_obsvar'][gs, i_spc, :, :] +
                                                     DA_innov.timeseries['obs_fcstvar'][gs, i_spc, :, :])).var(
            dim='time', skipna=True).values

        np.place(tmp, tmp == 0., np.nan)
        np.place(tmp, tmp == -9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc, :, :] = tmp.sum(axis=0)
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
    # cmap = np.floor(np.linspace(0,io.timeseries['time'].size,incr.size+1))
    ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'][gs_incr, :, :].mean(dim='time', skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'][gs_incr, :, :].var(dim='time', skipna=True).values
    gs_daily = (pd.to_datetime(io.timeseries['time'].values).month > 7) & (
                pd.to_datetime(io.timeseries['time'].values).month < 10)
    ndays = io.timeseries['time'][gs_daily].size
    n3hourly = DA_incr.timeseries['time'].size
    # overwrite if exp does not contain CLSM
    if exp.find("_CLSM") < 0:
        for row in range(DA_incr.timeseries['catdef'].shape[1]):
            logging.info('PCLSM row %i' % (row))
            for col in range(DA_incr.timeseries['catdef'].shape[2]):
                if poros[row, col] > 0.6:
                    catdef1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                           data=io.timeseries['catdef'][:, row, col].values)
                    catdef1 = catdef1.resample('3H').max().interpolate()
                    ar1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                       data=io.timeseries['ar1'][:, row, col].values)
                    ar1 = ar1.resample('3H').max().interpolate()
                    incr = pd.DataFrame(index=DA_incr.timeseries['time'].values,
                                        data=DA_incr.timeseries['catdef'][:, row, col].values)
                    # incr = DA_incr.timeseries['catdef'][:,row,col]
                    df = pd.concat((catdef1, incr, ar1), axis=1)
                    df.columns = ['catdef1', 'incr', 'ar1']
                    catdef2 = df['catdef1'] + df['incr']
                    catdef1 = df['catdef1']
                    ar1 = df['ar1']
                    incr = df['incr']
                    zbar1 = -1.0 * (np.sqrt(0.000001 + catdef1 / bf1[row, col]) - bf2[row, col])
                    zbar2 = -1.0 * (np.sqrt(0.000001 + catdef2 / bf1[row, col]) - bf2[row, col])
                    incr_ar1 = -(
                                zbar2 - zbar1) * 1000 * ar1  # incr surface water storage but expressed as deficit change, so '-'
                    # data2 = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,0,0].values)
                    # cdata = pd.concat(data1,data2)
                    # zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                    # incr = DA_incr.timeseries['catdef'][:,row,col]
                    # catdef = np.full_like(incr,np.nan)
                    # for i in range(n3hourly):
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
                    ds['incr_catdef_mean'][row, col] = incr_total[gs_incr, :, :].mean(skipna=True)
                    ds['incr_catdef_var'][row, col] = incr_total[gs_incr, :, :].var(skipna=True)
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'][gs_incr, :, :].mean(dim='time', skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'][gs_incr, :, :].var(dim='time', skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'][gs_incr, :, :].mean(dim='time', skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'][gs_incr, :, :].var(dim='time', skipna=True).values

    ds.close()


def anomaly_JulyAugust_zbar(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + '/anomaly_JulyAugust_zbar.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('daily', exp, domain, root)
    # growing season
    gs = (pd.to_datetime(io.timeseries['time'].values).month > 6) & (
                pd.to_datetime(io.timeseries['time'].values).month < 9)
    cgs = (pd.to_datetime(io.timeseries['time'].values) > '2012-07-01') & (
                pd.to_datetime(io.timeseries['time'].values) < '2012-08-31')
    cday = pd.to_datetime(io.timeseries['time'].values) == '2012-07-21'
    # get poros grid and bf1 and bf2 parameters
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['anomaly_zbar_21July', 'anomaly_zbar_JulyAugust', 'anomaly_tws_JulyAugust']

    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    # lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]
    dimensions = OrderedDict([('lat', lats), ('lon', lons)])
    ds = ncfile_init_incr(result_file, dimensions, tags)
    # clons = np.all(np.vstack((lons>60.,lons<90)),axis=0)
    # clats = np.all(np.vstack((lats>50.,lats<90)),axis=0)
    # ds['anomaly_zbar_21July'][clats, clons] = io.images['zbar'][cday,clats,clons].values - io.images['zbar'][gs,clats,clons].mean(axis=0).values
    # ds['anomaly_zbar_JulyAugust'][clats, clons] = io.images['zbar'][cgs,clats,clons].mean(axis=0).values - io.images['zbar'][gs,clats,clons].mean(axis=0).values
    # ds['anomaly_tws_JulyAugust'][clats, clons] = -io.images['catdef'][cgs,clats,clons].mean(axis=0).values - (-1.0*io.images['catdef'][gs,clats,clons].mean(axis=0).values)
    ds['anomaly_zbar_21July'][:, :] = io.images['zbar'][cday, :, :].values - io.images['zbar'][gs, :, :].mean(
        axis=0).values
    ds['anomaly_zbar_JulyAugust'][:, :] = io.images['zbar'][cgs, :, :].mean(axis=0).values - io.images['zbar'][gs, :,
                                                                                             :].mean(axis=0).values
    ds['anomaly_tws_JulyAugust'][:, :] = -io.images['catdef'][cgs, :, :].mean(axis=0).values - (
                -1.0 * io.images['catdef'][gs, :, :].mean(axis=0).values)

    ds.close()


def filter_diagnostics_evaluation_incr(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)
    # cvars = DA_innov.timeseries.data_vars
    # cond = DA_innov.timeseries['obs_assim']==-1
    # tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    # np.place(tmp, np.isnan(tmp), 1.)
    # np.place(tmp, tmp==False, 0.)
    # np.place(tmp, tmp==-1, 1.)
    # DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim']==-1)
    # cvars = ['obs_obs']
    # for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    # del cond1
    DA_incr = LDAS_io('incr', exp, domain, root)

    # get poros grid and bf1 and bf2 parameters
    io = LDAS_io('daily', exp, domain, root)
    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    catparam = io.read_params('catparam')
    poros = np.full(lons.shape, np.nan)
    poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
    bf1 = np.full(lons.shape, np.nan)
    bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
    bf2 = np.full(lons.shape, np.nan)
    bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values

    tags = ['innov_mean', 'innov_var',
            'norm_innov_mean', 'norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean', 'incr_catdef_var',
            'incr_rzexc_mean', 'incr_rzexc_var',
            'incr_srfexc_mean', 'incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    # lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]

    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][:, :, :].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    mask_innov = np.full_like(DA_innov.timeseries['obs_obs'][:, 0, :, :].values, False)
    mask_incr = np.full_like(DA_incr.timeseries['catdef'][:, :, :].values, False)
    mask_scaling_old = 0
    if mask_scaling_old == 1:
        n_innov_nyr = np.zeros([DA_innov.timeseries.dims['lat'], DA_innov.timeseries.dims['lon']])
        for j, yr in enumerate(np.unique(pd.Series(DA_innov.timeseries['time']).dt.year)):
            n_innov_yr = np.zeros([DA_innov.timeseries.dims['lat'], DA_innov.timeseries.dims['lon']])
            for i_spc, spc in enumerate(species):
                logging.info('species %i' % (i_spc))
                obs_obs_tmp = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values
                np.place(obs_obs_tmp, DA_innov.timeseries['obs_assim'][:, i_spc, :, :] != -1, np.nan)
                tmp = obs_obs_tmp - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :].values
                np.place(tmp, tmp == 0., np.nan)
                np.place(tmp, tmp == -9999., np.nan)
                np.place(tmp, ~np.isnan(tmp), 1.)
                n_innov_tmp = np.nansum(tmp[pd.Series(DA_innov.timeseries['time']).dt.year == yr, :, :], axis=0)
                np.place(n_innov_tmp, n_innov_tmp == 0., np.nan)
                n_innov_tmp = obs_M09_to_M36(n_innov_tmp)
                np.place(n_innov_tmp, np.isnan(n_innov_tmp), 0.)
                n_innov_yr = n_innov_yr + n_innov_tmp
            n_innov_yr = n_innov_yr / 2.0  # H-pol and V-pol count as one obs
            cond = (n_innov_yr < 3) & (n_innov_yr > 0)
            n_innov_nyr[n_innov_yr >= 3] = n_innov_nyr[n_innov_yr >= 3] + 1
            cmask = np.repeat(cond[np.newaxis, :, :],
                              np.where((pd.Series(DA_innov.timeseries['time']).dt.year == yr))[0].__len__(), axis=0)
            mask_innov[pd.Series(DA_innov.timeseries['time']).dt.year == yr, :, :] = cmask
            cmask = np.repeat(cond[np.newaxis, :, :],
                              np.where((pd.Series(DA_incr.timeseries['time']).dt.year == yr))[0].__len__(), axis=0)
            mask_incr[pd.Series(DA_incr.timeseries['time']).dt.year == yr, :, :] = cmask
        del cmask, n_innov_yr, cond, tmp, obs_obs_tmp
        cond2 = n_innov_nyr <= 1
        cmask = np.repeat(cond2[np.newaxis, :, :],
                          np.where((pd.Series(DA_innov.timeseries['time']).dt.year != -9999))[0].__len__(), axis=0)
        np.place(mask_innov, cmask, True)
        cmask = np.repeat(cond2[np.newaxis, :, :],
                          np.where((pd.Series(DA_incr.timeseries['time']).dt.year != -9999))[0].__len__(), axis=0)
        np.place(mask_incr, cmask, True)

    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))
        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values,
                 DA_innov.timeseries['obs_assim'][:, i_spc, :, :] != -1, np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values, mask_innov, np.nan)
        # ds['innov_mean'][i_spc,:,:] = (DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]).mean(dim='time',skipna=True).values
        ds['innov_var'][i_spc, :, :] = (
                    DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values - DA_innov.timeseries['obs_fcst'][:, i_spc, :,
                                                                            :]).var(dim='time', skipna=True).values
        # ds['norm_innov_mean'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
        #                                          np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).mean(dim='time',skipna=True).values
        # ds['norm_innov_var'][i_spc,:,:] = ((DA_innov.timeseries['obs_obs'][:,i_spc,:,:].values - DA_innov.timeseries['obs_fcst'][:,i_spc,:,:]) /
        #                                         np.sqrt(DA_innov.timeseries['obs_obsvar'][:,i_spc,:,:] + DA_innov.timeseries['obs_fcstvar'][:,i_spc,:,:])).var(dim='time',skipna=True).values

        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values,
                 DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values == 0., np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values,
                 DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values == -9999., np.nan)
        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values,
                 ~np.isnan(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values), 1.)
        np.place(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values,
                 np.isnan(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values), 0.)
        DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values = np.where(
            np.isnan(DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values), 0,
            DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values)
        ds['n_valid_innov'][i_spc, :, :] = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values.sum(axis=0)

    del mask_innov
    np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
    np.place(DA_incr.timeseries['catdef'].values, mask_incr, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, mask_incr, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, mask_incr, np.nan)
    # map from daily to hourly
    # cmap = np.floor(np.linspace(0,io.timeseries['time'].size,incr.size+1))
    # ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time',skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time', skipna=True).values
    ndays = io.timeseries['time'].size
    n3hourly = DA_incr.timeseries['time'].size
    # overwrite if exp does not contain CLSM
    if exp.find("_CLSM") < 0:
        for row in range(DA_incr.timeseries['catdef'].shape[1]):
            logging.info('PCLSM row %i' % (row))
            for col in range(DA_incr.timeseries['catdef'].shape[2]):
                if poros[row, col] > 0.6:
                    catdef1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                           data=io.timeseries['catdef'][:, row, col].values)
                    catdef1 = catdef1.resample('3H').max().interpolate()
                    ar1 = pd.DataFrame(index=io.timeseries['time'].values + pd.Timedelta('12 hours'),
                                       data=io.timeseries['ar1'][:, row, col].values)
                    ar1 = ar1.resample('3H').max().interpolate()
                    incr = pd.DataFrame(index=DA_incr.timeseries['time'].values,
                                        data=DA_incr.timeseries['catdef'][:, row, col].values)
                    # incr = DA_incr.timeseries['catdef'][:,row,col]
                    df = pd.concat((catdef1, incr, ar1), axis=1)
                    df.columns = ['catdef1', 'incr', 'ar1']
                    catdef2 = df['catdef1'] + df['incr']
                    catdef1 = df['catdef1']
                    ar1 = df['ar1']
                    incr = df['incr']
                    zbar1 = -1.0 * (np.sqrt(0.000001 + catdef1 / bf1[row, col]) - bf2[row, col])
                    zbar2 = -1.0 * (np.sqrt(0.000001 + catdef2 / bf1[row, col]) - bf2[row, col])
                    incr_ar1 = -(
                                zbar2 - zbar1) * 1000 * ar1  # incr surface water storage but expressed as deficit change, so '-'
                    # data2 = pd.DataFrame(index=DA_incr.timeseries['time'].values, data=DA_incr.timeseries['catdef'][:,0,0].values)
                    # cdata = pd.concat(data1,data2)
                    # zbar = -1.0*(np.sqrt(0.000001+catdef/bf1[row,col])-bf2[row,col])
                    # incr = DA_incr.timeseries['catdef'][:,row,col]
                    # catdef = np.full_like(incr,np.nan)
                    # for i in range(n3hourly):
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
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time', skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time', skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time', skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time', skipna=True).values

    ds.close()


def daily_stats(exp, domain, root, outputpath, stat, param='daily'):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'daily_' + stat + '.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io(param, exp, domain, root)
    cvars = io.timeseries.data_vars

    [lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon] = setup_grid_grid_for_plot(io)
    try:
        catparam = io.read_params('catparam')
        poros = np.full(lons.shape, np.nan)
        poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
        poros[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['poros'].values
        bf1 = np.full(lons.shape, np.nan)
        bf1[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf1'].values
        bf2 = np.full(lons.shape, np.nan)
        bf2[io.grid.tilecoord.j_indg.values, io.grid.tilecoord.i_indg.values] = catparam['bf2'].values
    except:
        print('catparam not yet read for nc files smaller than model run')
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    # lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]
    dimensions = OrderedDict([('lat', lats), ('lon', lons)])

    # cvars = ['total_water']
    ds = ncfile_init_incr(result_file, dimensions, cvars)
    for i, cvar in enumerate(cvars):
        logging.info('cvar %s' % (cvar))
        if stat == 'mean':
            ds[cvar][:, :] = io.timeseries[cvar][:, :, :].mean(dim='time', skipna=True).values
        elif stat == 'std':
            ds[cvar][:, :] = io.timeseries[cvar][:, :, :].std(dim='time', skipna=True).values

    x, y = get_cdf_integral_xy()
    if exp.find("_CLSM") < 0 & exp.find("GLOB_M36_7Thv_TWS_FOV0_M2") < 0:
        for row in range(io.timeseries['catdef'].shape[1]):
            for col in range(io.timeseries['catdef'].shape[2]):
                if poros[row, col] > 0.6:
                    S_ar1 = np.interp(io.timeseries['zbar'][:, row, col].values, x, y)
                    catdef = io.timeseries['catdef'][:, row, col].values
                    catdef_total = -S_ar1 * 1000.0 + catdef
                    total_water = catdef_total + io.timeseries['rzexc'][:, row, col].values + io.timeseries['srfexc'][:,
                                                                                              row, col].values
                    if stat == 'mean':
                        ds['catdef'][row, col] = np.nanmean(catdef_total)
                        ds['total_water'][row, col] = np.nanmean(total_water)
                    elif stat == 'std':
                        ds['catdef'][row, col] = np.nanstd(catdef_total)
                        ds['total_water'][row, col] = np.nanstd(total_water)

    ds.close()


def ensstd_stats(exp, domain, root, outputpath, stat):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'ensstd_' + stat + '.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    io = LDAS_io('ensstd', exp, domain, root)
    cvars = io.timeseries.data_vars

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
    # lons = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max()+1)]
    # lats = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max()+1)]
    lons = lons[0, :]
    lats = lats[:, 0]
    dimensions = OrderedDict([('lat', lats), ('lon', lons)])

    cvars = ['sfmc', 'rzmc', 'catdef', 'total_water', 'tp1', 'tsurf']
    # cvars = ['catdef','total_water']
    cvars = ['zbar']
    ds = ncfile_init_incr(result_file, dimensions, cvars)
    for i, cvar in enumerate(cvars):
        logging.info('cvar %s' % (cvar))
        if cvar == 'total_water':
            if stat == 'mean':
                ds[cvar][:, :] = (io.timeseries['catdef'][:, :, :] + io.timeseries['srfexc'][:, :, :] + io.timeseries[
                                                                                                            'rzexc'][:,
                                                                                                        :, :]).mean(
                    dim='time', skipna=True).values
            elif stat == 'std':
                ds[cvar][:, :] = (io.timeseries['catdef'][:, :, :] + io.timeseries['srfexc'][:, :, :] + io.timeseries[
                                                                                                            'rzexc'][:,
                                                                                                        :, :]).std(
                    dim='time', skipna=True).values
        else:
            if stat == 'mean':
                ds[cvar][:, :] = io.timeseries[cvar][:, :, :].mean(dim='time', skipna=True).values
            elif stat == 'std':
                ds[cvar][:, :] = io.timeseries[cvar][:, :, :].std(dim='time', skipna=True).values

    x, y = get_cdf_integral_xy()
    if (exp.find("_CLSM") < 0) & (cvars[0] != 'zbar'):
        for row in range(io.timeseries['catdef'].shape[1]):
            for col in range(io.timeseries['catdef'].shape[2]):
                if poros[row, col] > 0.6:
                    catdef = io.timeseries['catdef'][:, row, col].values
                    zbar = -1.0 * (np.sqrt(0.000001 + catdef / bf1[row, col]) - bf2[row, col])
                    S_ar1 = np.interp(zbar, x, y)
                    catdef_total = -S_ar1 * 1000.0 + catdef
                    total_water = catdef_total + io.timeseries['rzexc'][:, row, col].values + io.timeseries['srfexc'][:,
                                                                                              row, col].values
                    if stat == 'mean':
                        ds['catdef'][row, col] = np.nanmean(catdef_total)
                        ds['total_water'][row, col] = np.nanmean(total_water)
                    elif stat == 'std':
                        ds['catdef'][row, col] = np.nanstd(catdef_total)
                        ds['total_water'][row, col] = np.nanstd(total_water)

    ds.close()


def get_M09_ObsFcstAna(io, lon, lat, latlon=True):
    # get M09 row col with data from 4x4 square
    if latlon == True:
        col, row = io.grid.lonlat2colrow(lon, lat, domain=True)
    else:
        col = lon
        row = lat
    # TODO: check whether this is the right corner of the 4 possible M09 grid cells
    dcols = [-1, 0, 1, 2]
    drows = [-1, 0, 1, 2]
    cfind = False
    for icol, dcol in enumerate(dcols):
        for icol, drow in enumerate(drows):
            ts_obs = io.read_ts('obs_obs', np.min([col + dcol, io.images['lon'].size - 1]),
                                np.min([row + drow, io.images['lat'].size - 1]), species=2, lonlat=False)
            if np.isnan(ts_obs.max(skipna=True)) == False:
                cfind = True
                break
        if cfind == True:
            break
    if cfind == False:
        dcol = 0
        drow = 0
    col = col + dcol
    row = row + drow
    return col, row


def normal_distribution_function(x):
    # x1 = -1
    # x2 = 0
    # res, err = quad(normal_distribution_function, x1, x2)
    # print('Normal Distribution (mean,std):',cmean,cstd)
    # print('Integration bewteen {} and {} --> '.format(x1,x2),res)
    # cmean = 0.0
    # cstd = 0.11  # PEAT-CLSM value
    value = stats.norm.cdf(x, 0.0, 0.11)
    return value


def get_cdf_integral_xy():
    x = np.linspace(-2.0, 1.0, 1000)
    y = np.full_like(x, np.nan)
    for i, cx in enumerate(x):
        res, err = quad(normal_distribution_function, -2, x[i])
        y[i] = res
    return x, y


def filter_diagnostics_evaluation_old(exp, domain, root, outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    result_file = outputpath + 'filter_diagnostics.nc'
    os.remove(result_file) if os.path.exists(result_file) else None

    DA_innov = LDAS_io('ObsFcstAna', exp, domain, root)
    # cvars = DA_innov.timeseries.data_vars
    # cond = DA_innov.timeseries['obs_assim']==-1
    # tmp = DA_innov.timeseries['obs_assim'][:,:,:,:].values
    # np.place(tmp, np.isnan(tmp), 1.)
    # np.place(tmp, tmp==False, 0.)
    # np.place(tmp, tmp==-1, 1.)
    DA_innov.timeseries['obs_obs'] = DA_innov.timeseries['obs_obs'].where(DA_innov.timeseries['obs_assim'] == -1)
    # cvars = ['obs_obs']
    # for i,cvar in enumerate(cvars):
    #    if cvar != 'obs_assim':
    #        DA_innov.timeseries[cvar] = DA_innov.timeseries[cvar]*tmp
    #        #cond0 = DA_innov.timeseries['obs_assim'].values[:,0,:,:]!=False
    #        #DA_innov.timeseries[cvar].values[:,0,:,:][cond0] = DA_innov.timeseries[cvar].where(cond1)

    # del cond1
    DA_incr = LDAS_io('incr', exp, domain, root)

    tags = ['innov_mean', 'innov_var',
            'norm_innov_mean', 'norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean', 'incr_catdef_var',
            'incr_rzexc_mean', 'incr_rzexc_var',
            'incr_srfexc_mean', 'incr_srfexc_var',
            'n_valid_incr']

    tc = DA_innov.grid.tilecoord
    tg = DA_innov.grid.tilegrids
    lons = DA_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max() + 1)]
    lats = DA_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max() + 1)]
    species = DA_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, species, tags)

    tmp = DA_incr.timeseries['srfexc'][:, :, :].values
    np.place(tmp, tmp == 0., np.nan)
    np.place(tmp, ~np.isnan(tmp), 1.)
    np.place(tmp, np.isnan(tmp), 0.)
    ds['n_valid_incr'][:, :] = tmp.sum(axis=0)

    for i_spc, spc in enumerate(species):
        logging.info('species %i' % (i_spc))

        ds['innov_mean'][i_spc, :, :] = (
                    DA_innov.timeseries['obs_obs'][:, i_spc, :, :] - DA_innov.timeseries['obs_fcst'][:, i_spc, :,
                                                                     :]).mean(dim='time', skipna=True).values
        ds['innov_var'][i_spc, :, :] = (
                    DA_innov.timeseries['obs_obs'][:, i_spc, :, :] - DA_innov.timeseries['obs_fcst'][:, i_spc, :,
                                                                     :]).var(dim='time', skipna=True).values
        ds['norm_innov_mean'][i_spc, :, :] = (
                    (DA_innov.timeseries['obs_obs'][:, i_spc, :, :] - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]) /
                    np.sqrt(
                        DA_innov.timeseries['obs_obsvar'][:, i_spc, :, :] + DA_innov.timeseries['obs_fcstvar'][:, i_spc,
                                                                            :, :])).mean(dim='time', skipna=True).values
        ds['norm_innov_var'][i_spc, :, :] = (
                    (DA_innov.timeseries['obs_obs'][:, i_spc, :, :] - DA_innov.timeseries['obs_fcst'][:, i_spc, :, :]) /
                    np.sqrt(
                        DA_innov.timeseries['obs_obsvar'][:, i_spc, :, :] + DA_innov.timeseries['obs_fcstvar'][:, i_spc,
                                                                            :, :])).var(dim='time', skipna=True).values

        tmp = DA_innov.timeseries['obs_obs'][:, i_spc, :, :].values - DA_innov.timeseries['obs_fcst'][:, i_spc, :,
                                                                      :].values
        np.place(tmp, tmp == 0., np.nan)
        np.place(tmp, tmp == -9999., np.nan)
        np.place(tmp, ~np.isnan(tmp), 1.)
        np.place(tmp, np.isnan(tmp), 0.)
        ds['n_valid_innov'][i_spc, :, :] = tmp.sum(axis=0)

    np.place(DA_incr.timeseries['catdef'].values, DA_incr.timeseries['catdef'].values == 0, np.nan)
    np.place(DA_incr.timeseries['rzexc'].values, DA_incr.timeseries['rzexc'].values == 0, np.nan)
    np.place(DA_incr.timeseries['srfexc'].values, DA_incr.timeseries['srfexc'].values == 0, np.nan)
    ds['incr_catdef_mean'][:, :] = DA_incr.timeseries['catdef'].mean(dim='time', skipna=True).values
    ds['incr_catdef_var'][:, :] = DA_incr.timeseries['catdef'].var(dim='time', skipna=True).values
    ds['incr_rzexc_mean'][:, :] = DA_incr.timeseries['rzexc'].mean(dim='time', skipna=True).values
    ds['incr_rzexc_var'][:, :] = DA_incr.timeseries['rzexc'].var(dim='time', skipna=True).values
    ds['incr_srfexc_mean'][:, :] = DA_incr.timeseries['srfexc'].mean(dim='time', skipna=True).values
    ds['incr_srfexc_var'][:, :] = DA_incr.timeseries['srfexc'].var(dim='time', skipna=True).values

    ds.close()


def filter_diagnostics_evaluation_compare(exp1, exp2, domain, root, outputpath):
    # not working
    result_file = outputpath + 'filter_diagnostics.nc'

    DA_CLSM_innov = LDAS_io('ObsFcstAna', exp1, domain, root)
    DA_PEAT_innov = LDAS_io('ObsFcstAna', exp2, domain, root)

    DA_CLSM_incr = LDAS_io('incr', exp1, domain, root)
    DA_PEAT_incr = LDAS_io('incr', exp2, domain, root)

    runs = OrderedDict([(1, [DA_CLSM_innov.timeseries, DA_CLSM_incr.timeseries]),
                        (2, [DA_PEAT_innov.timeseries, DA_PEAT_incr.timeseries])])

    tags = ['innov_mean', 'innov_var',
            'norm_innov_mean', 'norm_innov_var',
            'n_valid_innov',
            'incr_catdef_mean', 'incr_catdef_var',
            'incr_rzexc_mean', 'incr_rzexc_var',
            'incr_srfexc_mean', 'incr_srfexc_var']

    tc = DA_CLSM_innov.grid.tilecoord
    tg = DA_CLSM_innov.grid.tilegrids
    lons = DA_CLSM_innov.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max() + 1)]
    lats = DA_CLSM_innov.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max() + 1)]

    species = DA_CLSM_innov.timeseries['species'].values

    ds = ncfile_init(result_file, lats, lons, runs.keys(), species, tags)

    for i_run, run in enumerate(runs):
        for i_spc, spc in enumerate(species):
            logging.info('run %i, species %i' % (i_run, i_spc))

            ds['innov_mean'][:, :, i_run, i_spc] = (
                        runs[run][0]['obs_obs'][:, i_spc, :, :] - runs[run][0]['obs_fcst'][:, i_spc, :, :]).mean(
                dim='time').values
            ds['innov_var'][:, :, i_run, i_spc] = (
                        runs[run][0]['obs_obs'][:, i_spc, :, :] - runs[run][0]['obs_fcst'][:, i_spc, :, :]).var(
                dim='time').values
            ds['norm_innov_mean'][:, :, i_run, i_spc] = (
                        (runs[run][0]['obs_obs'][:, i_spc, :, :] - runs[run][0]['obs_fcst'][:, i_spc, :, :]) /
                        np.sqrt(runs[run][0]['obs_obsvar'][:, i_spc, :, :] + runs[run][0]['obs_fcstvar'][:, i_spc, :,
                                                                             :])).mean(dim='time').values
            ds['norm_innov_var'][:, :, i_run, i_spc] = (
                        (runs[run][0]['obs_obs'][:, i_spc, :, :] - runs[run][0]['obs_fcst'][:, i_spc, :, :]) /
                        np.sqrt(runs[run][0]['obs_obsvar'][:, i_spc, :, :] + runs[run][0]['obs_fcstvar'][:, i_spc, :,
                                                                             :])).var(dim='time').values

            tmp = runs[run][0]['obs_obs'][:, i_spc, :, :].values
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


def estimate_tau(df, n_lags=60):
    """ Estimate characteristic time lengths for pd.DataFrame columns """

    # df must be already daily
    # df = in_df.copy().resample('1D').last()
    n_cols = len(df.columns)

    # calculate auto-correlation for different lags
    rho = np.full((n_cols, n_lags), np.nan)
    for lag in np.arange(n_lags):
        for i, col in enumerate(df):
            Ser_l = df[col].copy()
            Ser_l.index += pd.Timedelta(lag, 'D')
            rho[i, lag] = df[col].corr(Ser_l)

    # for i in np.arange(n_cols):
    #    plt.plot(rho[i,:],'.')

    # Fit exponential function to auto-correlations and estimate tau
    tau = np.full(n_cols, np.nan)
    for i in np.arange(n_cols):
        try:
            ind = np.where(~np.isnan(rho[i, :]))[0]
            if len(ind) > 20:
                popt = optimization.curve_fit(lambda x, a: np.exp(a * x), np.arange(n_lags)[ind], rho[i, ind],
                                              bounds=[-1., -1. / n_lags])[0]
                tau[i] = np.log(np.exp(-1.)) / popt
        except:
            # If fit doesn't converge, fall back to the lag where calculated auto-correlation actually drops below 1/e
            ind = np.where(rho[i, :] < np.exp(-1))[0]
            tau[i] = ind[0] if (len(ind) > 0) else n_lags  # maximum = # calculated lags

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
    ac = np.exp(-avg_spc_t / tau)
    avg_ac = ac.prod() ** (1. / len(ac))

    return tau, ac, avg_ac


def lag1_autocorr_from_numpy_array(data):
    tau = np.empty((data.shape[1], data.shape[2])) * np.nan
    acor_lag1 = np.empty((data.shape[1], data.shape[2])) * np.nan
    nall = data.shape[1] * data.shape[2]
    for i in np.arange(data.shape[1]):
        k = (i + 1) * data.shape[2]
        logging.info("%s/%s" % (k, nall))
        df = pd.DataFrame(data[:, i, :].values, index=data['time'].values)
        tau_tmp, ac_tmp, avg_ac_tmp = estimate_lag1_autocorr(df)
        tau[i, :] = tau_tmp
        acor_lag1[i, :] = ac_tmp
    return tau, acor_lag1


def lag1_autocorr_from_numpy_array_slow(data):
    acor_lag1 = np.empty((data.shape[1], data.shape[2])) * np.nan
    nall = data.shape[1] * data.shape[2]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            k = (i + 1) * (j + 1)
            logging.info("%s/%s" % (k, nall))
            df = data[:, i, j].to_series()
            df = pd.DataFrame(data[:, i, 0:data.shape[2]].values, index=data['time'].values)
            # df = pd.DataFrame([df]).swapaxes(0,1).dropna()
            df = df.dropna()
            acor_lag1[i, j] = estimate_lag1_autocorr(df)
    return acor_lag1


def obs_M09_to_M36(data):
    ydim_ind = np.where(np.nanmean(data, 0) > -9e30)
    indmin = 0
    indmax = np.size(ydim_ind)
    try:
        if np.size(ydim_ind[0]) > 0:
            if ydim_ind[0][0] < 3:
                indmin = 1
            if ydim_ind[0][-1] > (data.shape[1] - 3):
                indmax = np.size(ydim_ind) - 1
            try:
                data[:, ydim_ind[0][indmin:indmax] - 1] = data[:, ydim_ind[0][indmin:indmax]].values
                data[:, ydim_ind[0][indmin:indmax] + 1] = data[:, ydim_ind[0][indmin:indmax]].values
                data[:, ydim_ind[0][indmin:indmax] - 2] = data[:, ydim_ind[0][indmin:indmax]].values
                data[:, ydim_ind[0][indmin:indmax] + 2] = data[:, ydim_ind[0][indmin:indmax]].values
            except:
                data[:, ydim_ind[0][indmin:indmax] - 1] = data[:, ydim_ind[0][indmin:indmax]]
                data[:, ydim_ind[0][indmin:indmax] + 1] = data[:, ydim_ind[0][indmin:indmax]]
                data[:, ydim_ind[0][indmin:indmax] - 2] = data[:, ydim_ind[0][indmin:indmax]]
                data[:, ydim_ind[0][indmin:indmax] + 2] = data[:, ydim_ind[0][indmin:indmax]]

        xdim_ind = np.where(np.nanmean(data, 1) > -9e30)
        indmin = 0
        indmax = np.size(xdim_ind)
        if np.size(xdim_ind[0]) > 0:
            if xdim_ind[0][0] < 3:
                indmin = 1
            if xdim_ind[0][-1] > (data.shape[0] - 3):
                indmax = np.size(xdim_ind) - 1
            try:
                data[xdim_ind[0][indmin:indmax] - 1, :] = data[xdim_ind[0][indmin:indmax], :].values
                data[xdim_ind[0][indmin:indmax] + 1, :] = data[xdim_ind[0][indmin:indmax], :].values
                data[xdim_ind[0][indmin:indmax] - 2, :] = data[xdim_ind[0][indmin:indmax], :].values
                data[xdim_ind[0][indmin:indmax] + 2, :] = data[xdim_ind[0][indmin:indmax], :].values
            except:
                data[xdim_ind[0][indmin:indmax] - 1, :] = data[xdim_ind[0][indmin:indmax], :]
                data[xdim_ind[0][indmin:indmax] + 1, :] = data[xdim_ind[0][indmin:indmax], :]
                data[xdim_ind[0][indmin:indmax] - 2, :] = data[xdim_ind[0][indmin:indmax], :]
                data[xdim_ind[0][indmin:indmax] + 2, :] = data[xdim_ind[0][indmin:indmax], :]
    except:
        pass
    return data


def setup_grid_grid_for_plot(io):
    tc = io.grid.tilecoord
    tg = io.grid.tilegrids
    lons_1D = io.grid.ease_lons[tc['i_indg'].min():(tc['i_indg'].max() + 1)]
    lats_1D = io.grid.ease_lats[tc['j_indg'].min():(tc['j_indg'].max() + 1)]

    try:
        latmin = io.images['lat'].min().values.item()
        latmax = io.images['lat'].max().values.item()
        lonmin = io.images['lon'].min().values.item()
        lonmax = io.images['lon'].max().values.item()
    except:
        latmin = io.grid.ease_lats.min()
        latmax = io.grid.ease_lats.max()
        lonmin = io.grid.ease_lons.min()
        lonmax = io.grid.ease_lons.max()

    domainlons = io.grid.ease_lons[np.min(io.grid.tilecoord.i_indg):(np.max(io.grid.tilecoord.i_indg) + 1)]
    domainlats = io.grid.ease_lats[np.min(io.grid.tilecoord.j_indg):(np.max(io.grid.tilecoord.j_indg) + 1)]
    lonmin = domainlons[np.argmin(np.abs(domainlons - lonmin))]
    lonmax = domainlons[np.argmin(np.abs(domainlons - lonmax))]
    latmin = domainlats[np.argmin(np.abs(domainlats - latmin))]
    latmax = domainlats[np.argmin(np.abs(domainlats - latmax))]

    # Use grid lon lat to avoid rounding issues
    tmp_tilecoord = io.grid.tilecoord.copy()
    tmp_tilecoord['com_lon'] = io.grid.ease_lons[io.grid.tilecoord.i_indg]
    tmp_tilecoord['com_lat'] = io.grid.ease_lats[io.grid.tilecoord.j_indg]

    # Clip region based on specified coordinate boundaries
    ind_img = io.grid.tilecoord[(tmp_tilecoord['com_lon'] >= lonmin) & (tmp_tilecoord['com_lon'] <= lonmax) &
                                (tmp_tilecoord['com_lat'] <= latmax) & (tmp_tilecoord['com_lat'] >= latmin)].index
    lons_1D = domainlons[(domainlons >= lonmin) & (domainlons <= lonmax)]
    lats_1D = domainlats[(domainlats >= latmin) & (domainlats <= latmax)]
    i_offg_2 = np.where(domainlons >= lonmin)[0][0]
    j_offg_2 = np.where(domainlats <= latmax)[0][0]

    tc.i_indg -= tg.loc['domain', 'i_offg']  # col / lon
    tc.j_indg -= tg.loc['domain', 'j_offg']  # row / lat
    lons, lats = np.meshgrid(lons_1D, lats_1D)
    llcrnrlat = np.min(lats)
    urcrnrlat = np.max(lats)
    llcrnrlon = np.min(lons)
    urcrnrlon = np.max(lons)
    return lons, lats, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon


def calc_tau_and_lag1_autocor(self):
    tmp_incr = self.timeseries['catdef']
    tmp_incr = tmp_incr.where(tmp_incr != 0)
    tau, acor_lag1 = lag1_autocorr_from_numpy_array(tmp_incr)

    variables = ["tau", "acor_lag1"]

    out_file = os.path.join(self.paths.ana, '..', '..', '..', 'output_postprocessed', self.param + '_autocor.nc')

    domainlons = self.grid.ease_lons[np.min(self.grid.tilecoord.i_indg):(np.max(self.grid.tilecoord.i_indg) + 1)]
    domainlats = self.grid.ease_lats[np.min(self.grid.tilecoord.j_indg):(np.max(self.grid.tilecoord.j_indg) + 1)]

    lonmin = np.min(domainlons)
    lonmax = np.max(domainlons)
    latmin = np.min(domainlats)
    latmax = np.max(domainlats)

    # Use grid lon lat to avoid rounding issues
    tmp_tilecoord = self.grid.tilecoord.copy()
    tmp_tilecoord['com_lon'] = self.grid.ease_lons[self.grid.tilecoord.i_indg]
    tmp_tilecoord['com_lat'] = self.grid.ease_lats[self.grid.tilecoord.j_indg]

    # Clip region based on specified coordinate boundaries
    ind_img = self.grid.tilecoord[(tmp_tilecoord['com_lon'] >= lonmin) & (tmp_tilecoord['com_lon'] <= lonmax) &
                                  (tmp_tilecoord['com_lat'] <= latmax) & (tmp_tilecoord['com_lat'] >= latmin)].index
    lons = domainlons[(domainlons >= lonmin) & (domainlons <= lonmax)]
    lats = domainlats[(domainlats >= latmin) & (domainlats <= latmax)]
    i_offg_2 = np.where(domainlons >= lonmin)[0][0]
    j_offg_2 = np.where(domainlats <= latmax)[0][0]

    dimensions = OrderedDict([('lat', lats), ('lon', lons)])

    dataset = self.ncfile_init(out_file, dimensions, variables)

    dataset.variables['tau'][:, :] = tau[:, :]
    dataset.variables['acor_lag1'][:, :] = acor_lag1[:, :]
    # Save file to disk and loat it as xarray Dataset into the class variable space
    dataset.close()


def calc_anomaly(Ser, method='moving_average', output='anomaly', longterm=True, window_size=31):
    if (output == 'climatology') & (longterm is True):
        output = 'climSer'

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    climSer = pd.Series(index=xSer.index)

    if not method in ['harmonic', 'mean', 'moving_average', 'ma']:
        logging.error('Unknown method: ' + method)
        return climSer

    if longterm is True:
        if method == 'harmonic':
            clim = calc_clim_harmonic(xSer)
        if method == 'mean':
            clim = calc_clim_harmonic(xSer, n=0)
        if (method == 'moving_average') | (method == 'ma'):
            clim = calc_clim_moving_average(xSer, window_size=window_size)
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
                clim = calc_clim_moving_average(xSer[years == yr], window_size=window_size)
            climSer[years == yr] = clim[doys[years == yr]].values

    if output == 'climSer':
        return climSer

    climSer.name = xSer.name
    return xSer - climSer


def calc_clim_moving_average(Ser, window_size=31, n_min=46, return_n=False):
    """
    Calculates the mean seasonal cycle as long-term mean within a moving average window.
    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    window_size : int
        Moving Average window size
    n_min : int
        Minimum number of data points to calculate average
    return_n : boolean
        If true, the number of data points over which is averaged is returned
    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    clim_doys = np.arange(365) + 1
    clim = pd.Series(index=clim_doys)
    n_data = pd.Series(index=clim_doys)

    for doy in clim_doys:

        # Avoid artifacts at start/end of year
        tmp_doys = doys.copy()
        if doy < window_size / 2.:
            tmp_doys[tmp_doys > 365 - (np.ceil(window_size / 2.) - doy)] -= 365
        if doy > 365 - (window_size / 2. - 1):
            tmp_doys[tmp_doys < np.ceil(window_size / 2.) - (365 - doy)] += 365

        n_data[doy] = len(xSer[(tmp_doys >= doy - np.floor(window_size / 2.)) & \
                               (tmp_doys <= doy + np.floor(window_size / 2.))])

        if n_data[doy] >= n_min:
            clim[doy] = xSer[(tmp_doys >= doy - np.floor(window_size / 2.)) & \
                             (tmp_doys <= doy + np.floor(window_size / 2.))].values.mean()

    if return_n is False:
        return clim
    else:
        return clim, n_data


def resample_merra2(part=1, parts=1):
    """
    This resamples MERRA-2 data from the MERRA grid onto the EASE2 grid and stores data for each grid cell into .csv files.
    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).
    Parameters
    ----------
    part : int
        Data subset to be processed - Data can be resampled in subsets for parallelization to speed-up the processing.
    parts : int
        Number of parts in which to split the data for parallel processing.
        Per default, all data are resampled at once.
    """

    paths = Paths()

    dir_out = Path('/staging/leuven/stg_00024/OUTPUT/michelb/merra2')
    if not dir_out.exists():
        dir_out.mkdir()

    path = paths.merra2_raw
    files07 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20070[78]*nc4')))
    files08 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20080[78]*nc4')))
    files09 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20090[78]*nc4')))
    files10 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20100[78]*nc4')))
    files11 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20110[78]*nc4')))
    files12 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20120[78]*nc4')))
    files13 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20130[78]*nc4')))
    files14 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20140[78]*nc4')))
    files15 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20150[78]*nc4')))
    files16 = np.array(sorted(path.rglob('*tavg1_2d_lfo_Nx.20160[78]*nc4')))
    files = np.concatenate([files07, files08, files09, files10, files11, files12, files13, files14, files15, files16])
    ds = xr.open_mfdataset(files)
    return ds


class Paths(object):
    """ This class contains the paths where data are stored and results should be written to."""

    def __init__(self):
        self.result_root = Path('/work/validation_good_practice')

        self.data_root = Path('/data_sets')

        self.lut = self.data_root / 'EASE2_grid' / 'grid_lut.csv'

        self.ascat = self.data_root / 'ASCAT'
        self.smos = self.data_root / 'SMOS'
        self.smap = self.data_root / 'SMAP'
        self.merra2 = Path('/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing')
        self.ismn = self.data_root / 'ISMN'

        # ASCAT raw data should be in self.ascat with the folder names matching the H SAF version number as downloaded
        self.smos_raw = self.smos / 'raw' / 'MIR_SMUDP2_nc'
        self.smap_raw = self.smos / 'raw'
        self.merra2_raw = self.merra2
        self.ismn_raw = self.ismn / 'downloaded' / 'CONUS_20100101_20190101'


if __name__ == '__main__':
    estimae_lag1_autocorr()
