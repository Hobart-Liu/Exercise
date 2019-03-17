import pandas as pd

idx1 = [
    pd.Timestamp(2015, 1, 4, 0),
    pd.Timestamp(2015, 1, 4, 2),
]

idx2 = [
    pd.Timestamp(2015, 1, 4, 0),
    pd.Timestamp(2015, 1, 4, 12),
    pd.Timestamp(2015, 1, 4, 2),
    pd.Timestamp(2015, 1, 4, 13),
    pd.Timestamp(2015, 1, 4, 22),

]


df1 = pd.DataFrame(index = idx1)
df1['dummy'] = 1
df1.index.name='datetime'
df2 = pd.DataFrame(index=idx2)
df2['dummy'] = 2
df2.index.name='datetime'

print(df1)
df1.index = df1.index.floor('12H')
print(df1)


print(df2)
df2.index = df2.index.floor('12H')
print(df2)

df3 = pd.merge(df1, df2, how='right', left_index=True, right_index=True)
print(df3)

exit()

