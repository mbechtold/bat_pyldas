from bat_pyldas.functions import read_wtd_data
from bat_pyldas.plotting import plot_skillmetrics_comparison_wtd

import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from validation_good_practice.ancillary import metrics


root='/staging/leuven/stg_00024/OUTPUT/hugor/output'
exp = 'INDONESIA_M09_v01_spinup2'
domain = 'SMAP_EASEv2_M09'
outpath = '/vsc-hard-mounts/leuven-data/329/vsc32924/figures_temp'

insitu_path = '/vsc-hard-mounts/leuven-data/329/vsc32924/in-situ_data'

wtd_obs, wtd_mod = read_wtd_data(insitu_path, exp, domain, root)

plot_skillmetrics_comparison_wtd(wtd_obs, wtd_mod, exp, outpath)
