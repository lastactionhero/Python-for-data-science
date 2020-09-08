# load test data and try to submit it
#%%
submissiondf = pd.read_csv('test.csv')

#%%
submissiondf.describe()
# %%
submissiondf[submissiondf.columns[submissiondf.isnull().any()]].isnull().sum()

# %%
submissiondf.drop(['LotFrontage','MasVnrType','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],inplace=True,axis=1)

# %%
nacolumns = submissiondf.columns[submissiondf.isnull().any()]
# %%
for col in nacolumns:
    sns.countplot(x=col, data =df )
    plt.show()
#%%
df['MasVnrType'].mode()

#%%
# %%
"""['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
      'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional',
       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
       'GarageQual', 'GarageCond', 'SaleType'],
"""
# df['MSZoning'].fillna('RL',inplace=True)
# df['Utilities'].fillna('AllPub',inplace=True)
# df['Exterior1st'].fillna(df['Exterior1st'].mode(),inplace=True)
# df['MSZoning'].fillna('RL',inplace=True)
for col in nacolumns:
    submissiondf[col].fillna(df[col].mode()[0],inplace=True)
#%%
# need to append original dataset to get dummies with exact nummber of columns
submissiondf.count()
#%%
# we need to append original dataset so that it can create same number of dummy columns
submissiondf = pd.concat([submissiondf,df.drop('SalePrice', axis=1)],axis=0)
#%%
final = pd.get_dummies(submissiondf,columns=df.columns[df.dtypes=='object'],drop_first=True)
#%%
final = final[:1459]
#%%
final.head()
#%%
final.count()
# %%
scaled_df = pd.DataFrame( scale.transform(final))
# %%
pca_df = pd.DataFrame(pca.transform(scaled_df))
# %%
y_predict = lr.predict(final)
# %%
y_predict
#%%

ydf = pd.DataFrame(y_predict,columns=['SalePrice'])
#%%
answers = pd.concat([final['Id'],ydf], axis=1)

# %%
answers.to_csv("answers.csv", index=False)

# %%
