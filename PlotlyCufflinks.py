# Plotly is interactive visualization library and cufflinks connects it with pandas 
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# %%
init_notebook_mode(connected=True)
# %%
cf.go_offline()

# %%
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
# %%
tips.head()
# %%
iris.head()
#%%
sns.jointplot(data=iris, x='petal_length', y='petal_width', hue='species')
#%%
iris.plot()
# %%
iris.iplot()

# %%
iris.iplot(kind='scatter', x='sepal_width', y='sepal_length', mode='markers')

# %%
tips.iplot(kind='scatter', x='tip', y='total_bill', mode='markers' )

# %%
iris.iplot(kind='box')

# %%
