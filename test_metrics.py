
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from validation_good_practice.ancillary import metrics

date_today = dt.date.today()
ndays = 90

# first site
site1 = pd.Series(np.random.randn(ndays))
df = pd.DataFrame({'date': [date_today + timedelta(days=x) for x in range(ndays)], 'site1': site1})
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# then move on with for loop, here example for adding site 2
site2 = pd.Series(np.random.randn(ndays))
df_tmp = pd.DataFrame({'date': [date_today + timedelta(days=x) for x in range(ndays)], 'site2': site2})
df_tmp = df_tmp.set_index('date')

df = pd.concat((df, df_tmp), axis=1)

res = metrics.bias(df)

print(res)

