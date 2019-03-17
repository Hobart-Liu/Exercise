import numpy as np
import pandas as pd

rec_a = [1, np.nan, np.nan, 1, np.nan, np.nan, np.nan]
rec_b = [np.nan, np.nan, 1, 1, np.nan, np.nan, np.nan]
datetime = pd.date_range(start='2015-01-01', periods=7, freq='D', name='datetime')

df1 = pd.DataFrame(rec_a, columns = ['maint'], index=datetime)
df1['id'] = 'a'
df2 = pd.DataFrame(rec_b, columns = ['maint'], index=datetime)
df2['id'] = 'b'
df = pd.concat((df1, df2), axis=0)

df['days'] = df.index.where(df['maint'].eq(1))
df['days'] = (df.index - df.groupby('id')['days'].ffill()).fillna(pd.Timedelta(0)).dt.days


exit()

df = df.reset_index()
a = df['maint'].eq(1).groupby(df['id']).cumsum()
s = df['datetime'].sub(df.groupby(['id',a])['datetime'].transform('first')).dt.days
df['new'] = np.where(a != 0, s, 0)
print (df)

print(df)