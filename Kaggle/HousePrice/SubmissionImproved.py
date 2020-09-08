#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# %%
# get test and train data together for feature engineering
df_train = pd.read_csv('train.csv')
# %%
#1460
df_train.describe()
# %%
df_test =  pd.read_csv('test.csv')
#%%
#1459
df_test.describe()
# %%
df= pd.concat([df_train.drop('SalePrice',axis=1),df_test], axis=0)
# %%
df.describe()
# %%
# lets get rid of some of unwanted columns
df.drop('Id',axis=1,inplace=True)
# %%
nacolumns = df.columns[df.isnull().any()]
nacolumns
# %%
# columnwise null value cound
df[nacolumns].isnull().sum()
# %%
# get percentage of distribution
df[nacolumns].isnull().sum()*100/2919.000000
#%%
# remove columns with more than 10% null values
df.drop(['LotFrontage', 'Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
# %%
nacolumns = df.columns[df.isnull().any()]
# %%
for col in nacolumns:
    if(df[col].dtype=='object'):
        print (col,df[col].mode()[0])
        df[col].fillna(df[col].mode()[0], inplace=True)
    else: 
        print (col,df[col].mean())
        df[col].fillna(df[col].mean(), inplace=True)
# %%
# try to convert all category columns to normal int columns
df = pd.get_dummies(df, drop_first=True)
df.head()

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#%%
def Preprocessing(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df=scaler.transform(df)
    ##pca = PCA(n_components=130)
    ##pca.fit(df)
    ##df = pca.transform(df)
    return df
# %%
def Model_Execution(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # since it was log normal scalled earlier, rescale it
    result = np.floor(np.expm1(y_pred))
   # result = pd.DataFrame(result)
    return result

#%%
#Original_df = df
#%%
df = Original_df
df= Preprocessing(df)
X = df[:1460]
y= np.log1p(df_train['SalePrice']) # log normal distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#%%
testdf = pd.DataFrame(df)
#%%
#%%
from sklearn import linear_model
from sklearn import metrics
import xgboost as xgb
# %%
lr=linear_model.LinearRegression()
lasso= linear_model.Lasso(alpha=0.1)
boost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
#%%
nm = type(lasso).__name__
nm
# %%
models =[lasso,boost]
result =pd.DataFrame(data=np.zeros(y_test.size))
for model in models:
    model_name = type(model).__name__
    predict_y= Model_Execution(model)
    resultdf = pd.DataFrame(predict_y,columns=[model_name])
    result = pd.concat([result, resultdf],axis=1)
    print(model_name)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,predict_y))
    print(rmse)
result.head()
# %%
submission = lasso.predict(df[1460:])

submission =pd.DataFrame(np.floor(np.expm1(submission)), columns=['SalePrice'])
# %%
submission.describe()
# %%
submission = pd.concat([df_test['Id'],submission], axis=1)
# %%
submission.to_csv("answers.csv", index=False)
# %%
