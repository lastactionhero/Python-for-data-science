#%%
import pandas as pd
import seaborn as sns
import numpy as np
from pandas_datareader import data, wb
import datetime 
#%%
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2020, 1, 1)
# %%
# Bank of America
BAC = data.DataReader("BAC", 'yahoo', start, end)
# CitiGroup
C = data.DataReader("C", 'yahoo', start, end)
# Goldman Sachs
GS = data.DataReader("GS", 'yahoo', start, end)
# JPMorgan Chase
JPM = data.DataReader("JPM", 'yahoo', start, end)
# Morgan Stanley
MS = data.DataReader("MS", 'yahoo', start, end)
# Wells Fargo
WFC = data.DataReader("WFC", 'yahoo', start, end)
#%%
BAC.head()
# %%
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# %%
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],keys=tickers, names=['Ticker','Date'])
bank_stocks.head()
# %%
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
# %%
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

#%%
bank_stocks.head()


# %%
# Max close values of stock
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
#%%
bank_stocks['BAC']['Close'].pct_change()
# %%
returns = pd.DataFrame()
for tick in tickers:
    returns[tick + ' Return']= bank_stocks[tick]['Close'].pct_change()

# %%
returns.head()

# %%
sns.pairplot(returns)

# %%
# Min and max gain dates
returns.idxmax()

# %%
returns.idxmin()

# %%
#riskiest stocks : Calculate standard deviation
returns.std()

# %%
returns.loc['2015-01-01':'2015-12-31'].std()

# %%
# Morgaan stanley dist plot for 2019 values
sns.set_style('darkgrid')
sns.set_palette('inferno')
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'], bins=100)

# %%
#moving 30 days avg
BAC['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean().plot(label='30 day average')
BAC['Close'].loc['2008-01-01':'2008-12-31'].plot(label='BAC Close')
# %%
# use sns to plot thisngs
BAC['Moving average 30 days']=  BAC['Close'].rolling(window=30).mean()
# %%
BAC.tail()
# %%
df2019 = BAC.loc['2019-01-01':'2019-12-31']
sns.lineplot(data=df2019, x=df2019.index, y='Moving average 30 days')
sns.lineplot(data=df2019, x=df2019.index, y='Close')
# %%
bank_stocks.head()

# %%
# Create a heatmap of the correlation between the stocks Close Price.
bank_stocks.xs(key='Close', axis=1, level='Stock Info')

# %%
sns.set_palette('copper')
sns.heatmap(bank_stocks.xs(key='Close', axis=1, level='Stock Info').corr())

# %%
