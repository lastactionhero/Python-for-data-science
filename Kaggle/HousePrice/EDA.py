#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df= pd.read_csv('train.csv')
# %%
df.head()
#%%
df.info()
# %%
for col in df.columns:
    print(f'column - {col} and type - {df[col].dtype}')
    if(df[col].dtype == 'object'):
       df[col] = df[col].astype('category')
       
# %%
df[df.columns[df.dtypes == 'category']]
# %%
df_analysis = pd.concat([df[df.columns[df.dtypes == 'category']], df['SalePrice']],axis=1)
# %%
df_analysis.head()
# %%
# %%
for col in df[df.columns[df.dtypes == 'category']]:
    sns.countplot(df[col])
    plt.show()
# %%
sns.distplot(df['SalePrice'],rug=True)
# %%
