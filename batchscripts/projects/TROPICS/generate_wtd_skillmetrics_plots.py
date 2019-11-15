from bat_pyldas.functions import read_wtd_data
from bat_pyldas.plotting import plot_skillmetrics_comparison_wtd

import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from validation_good_practice.ancillary import metrics


root='/scratch/leuven/317/vsc31786/output/TROPICS'
exp = 'CONGO_M09_PEATCLSMTN_v01'
domain = 'SMAP_EASEv2_M09'
outpath = '/staging/leuven/stg_00024/OUTPUT/michelb/FIG_tmp/TROPICS'
insitu_path = '/data/leuven/317/vsc31786/peatland_data'
mastertable_filename = 'WTD_TROPICS_MASTER_TABLE.csv'

wtd_obs, wtd_mod, precip_obs = read_wtd_data(insitu_path, mastertable_filename, exp, domain, root)

plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, precip_obs, exp, outpath)
