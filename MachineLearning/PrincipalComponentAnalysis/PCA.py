#%%
import pandas as pd
import seaborn as sns
import numpy as np
# %%
from sklearn.datasets import load_breast_cancer
# %%
bs= load_breast_cancer()
# %%
bs
# %%
df = pd.DataFrame(data=bs.data, columns= bs.feature_names)
# %%
df.head()
# %%
from sklearn.preprocessing import StandardScaler
# %%
std = StandardScaler()
# %%
std.fit(df)
# %%
dfscaled = pd.DataFrame(std.transform(df))
# %%
dfscaled.head()
# %%
# PCA
from sklearn.decomposition import PCA
# %%
pca = PCA(n_components=2)
# %%
pca.fit(dfscaled)
# %%
x_PCA = pca.transform(dfscaled)
#%%
x_PCA
# %%
sns.set_palette("copper")
sns.scatterplot(x= x_PCA[:,1], y= x_PCA[:,0], hue=bs['target'])
# %%
pca.components_
#%%
componentdf = pd.DataFrame(pca.components_, columns=df.columns)
# %%
sns.set_palette("Set1")
sns.heatmap(componentdf, cmap='inferno')
# %%
# since we have reduced data, we can now do logistic regression on this reduced data
from sklearn.model_selection import train_test_split
# %%
X=x_PCA
y=bs['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
from sklearn.linear_model import LogisticRegression
# %%
lr = LogisticRegression()
# %%
lr.fit(X_train,y_train)
# %%
y_pred = lr.predict(X_test)
# %%
from sklearn.metrics import classification_report
# %%
print('\n')
print(classification_report(y_test,y_pred))
# %%
