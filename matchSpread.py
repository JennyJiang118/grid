import pandas as pd
import numpy as np
import sys

sys.path.insert(0,'.')
future_path = 'data/IC_IF/IF/IF1903-1812.csv'
index_path = 'data/SH/IF300-2019.csv'
future = pd.read_csv(future_path)
index = pd.read_csv(index_path)

data = pd.merge(future, index, on='datetime',how='inner')

#%%

for i in range(241,len(data)):
    data.loc[i,'index_volatility'] = np.std(data.loc[i-240:i,'open'])

#%%
data = data.dropna(axis=0,how='any').reset_index(drop=True)

#%%
data['volatility_a'] = data['index_volatility'] - data['index_volatility'].shift(1)
data = data.dropna(axis=0,how='any').reset_index(drop=True)

#%%

data.to_csv('data/index_volatility/IF1903-1812.csv',index=False)
