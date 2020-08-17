#%%
import pandas as pd
import seaborn as sns
import numpy as np
# %%
df = pd.read_csv("Classified Data", index_col=0)
# %%
df.head()
# %%
sns.pairplot(df)
# %%
# ## Preprocessing ##
# lets represent all the features into similar scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

# %%
# we can directly use array object returned by scaler standardscaler or we can convert it to df
scaled_df = scaler.transform(df.drop('TARGET CLASS', axis=1))
#%%
df.columns[:-1]
# %%
scaled_df = pd.DataFrame(scaled_df, columns=df.columns[:-1])
# %%
scaled_df.head()
# %%
from sklearn.model_selection import train_test_split
# %%
X=scaled_df
y=df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# %%
pred= knn.predict(X_test)
# %%
from sklearn.metrics import classification_report, confusion_matrix

# %%
print(classification_report(y_test,pred))
# %%
# understand how np array works
a=np.array([1,2,3,4])
b=np.array([2,2,3,12])
a!=b
#
# %%
np.mean(a!=b)
#%%
np.mean(np.abs(a-b))
# %%
# now lets calculate error rate and decide best number for n_neighbors parameter 
error_rate=[]
error_mean =[]
for i in np.arange(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    error_mean.append(np.mean(np.abs(pred-y_test)))
    error_rate.append(np.mean(pred!=y_test)) 
#%%
error_rate
#%%
error_mean
#%%
sns.lineplot(x=np.arange(1,40), y= error_rate)

# %%
sns.lineplot(x=np.arange(1,40), y= error_mean)
# %%
