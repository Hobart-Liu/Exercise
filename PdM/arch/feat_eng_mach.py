import pandas as pd
import numpy as np

df = pd.read_csv('data/pdm_machines_data.csv', encoding='utf-8')
repl_index = {'model1':(0, 0, 0, 1), 'model2':(0, 0, 1, 0), 'model3':(0, 1, 0, 0), 'model4':(1, 0, 0, 0)}
df['model_encoded'] = df['model'].map(repl_index)
df.to_csv('out_mach_feat.csv')
print("Check null value")
print(df.isnull().any())
print(df.head(20))