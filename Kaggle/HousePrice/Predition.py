# We are done with EDA
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Now lets built model and start with replacing categorical columns with dummies
#%%
df = pd.read_csv('train.csv')
# %%
df.describe()
#%%
df.mode(axis=0)
#%%
df.head()
#%%
df['MasVnrType']
#%%
# list null columns with more than one null value
df[df.columns[df.isnull().any()]].isnull().sum()*100/1460
#%%
# Drop columns who has more than 15% null values
df.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],inplace=True,axis=1)
#%% 
# columns with na 
nacolumns= df.columns[df.isnull().any()]
#%%
df[nacolumns]
#%%
# look at the null column data
for col in nacolumns:
    sns.countplot(x=col,data=df)
    plt.show()
#%%
# based on graph replace null should follow logic below
# ['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
#       'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType',
#       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
# MasVnrType - remove
# MasVnrArea - replace by mean
# BsmtQual = 'TA' (ideally should be derived)
# BsmtCond - 'TA'
# BsmtExposure - NO
# BsmtFinType1 - Unf
# BsmtFinType2 - Unf
# Electrical - SBrkr
# GarageType - Attchd
# GarageYrBlt - mode
# GarageFinish - Unf
# GarageQual - TA
# GarageCond - TA
#%%
df.drop('MasVnrType',inplace=True, axis=1)
#%%
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(),inplace=True)
#%%
# replace null values with median. this is not recommended. we should analyze each null value column with SalesPrice.
df['BsmtQual'].fillna('TA',inplace=True)
#%%
df['BsmtCond'].fillna('TA',inplace=True)
#%%
df['BsmtExposure'].fillna('NO',inplace=True)
#%%
df['GarageYrBlt'].mode()
df['GarageYrBlt'].fillna(2005 ,inplace=True)
#%%
df['BsmtFinType1'].fillna('Unf',inplace=True)
df['BsmtFinType2'].fillna('Unf',inplace=True)
df['Electrical'].fillna('SBrkr',inplace=True)
df['GarageType'].fillna('Attchd',inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode() ,inplace=True)
df['GarageFinish'].fillna('Unf' ,inplace=True)
df['GarageQual'].fillna('TA' ,inplace=True)
df['GarageCond'].fillna('TA' ,inplace=True)

#%%
df.columns[df.dtypes=='object']
# %%
df[df.columns[df.dtypes=='object']]
# %%
# get dummies for the object type columns
final = pd.get_dummies(df,columns=df.columns[df.dtypes=='object'],drop_first=True)
#%%
final.head()
#%%
X= final.drop('SalePrice',  axis=1)
y= final['SalePrice']
# %%
from sklearn.model_selection import train_test_split
#%%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
#%%
# scale all the columns
from sklearn.preprocessing import StandardScaler
# %%
scale = StandardScaler()
# %%
scale.fit(X_train)
# %%
scaled_df = pd.DataFrame( scale.transform(X_train))
# %%
from sklearn.decomposition import PCA
# %%
pca = PCA(n_components=10)
# %%
pca.fit(scaled_df)
# %%
pca_df = pd.DataFrame(pca.transform(scaled_df))
# %%
from sklearn.linear_model import LinearRegression
# %%
lr = LinearRegression()
# %%
lr.fit(X= scaled_df, y=y_train)
# %%
y_predict = lr.predict(X_test)
# %%
from sklearn import metrics
# %%
metrics.mean_absolute_error(y_true=y_test, y_pred=y_predict)
metrics.mean_squared_error(y_true=y_test, y_pred=y_predict)
# %%

# %%
