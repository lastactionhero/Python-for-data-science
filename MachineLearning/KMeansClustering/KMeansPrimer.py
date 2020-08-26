#%%
import numpy as np
import pandas as pd
import seaborn as sns
# %%
from sklearn import datasets
# %%
iris = datasets.load_iris()
#%%
iris
# %%
irisdf = pd.DataFrame(data= iris.data,columns=iris.feature_names)
# %%
irisdf.head()
# %%
sns.pairplot(irisdf)
# %%
from sklearn.preprocessing import StandardScaler
# %%
scaler = StandardScaler()
# %%
scaler.fit(irisdf)
# %%
iris_scaled = pd.DataFrame(scaler.transform(irisdf))
# %%
from sklearn.cluster import KMeans
# %%
kmeans = KMeans(n_clusters = 3)
# %%
kmeans.fit(iris_scaled)
# %%
kmeans.labels_
# %%
iris_scaled['Label']=kmeans.labels_
# %%
sns.pairplot(data = iris_scaled, hue='Label')
# %%
irisdf['Label']= iris.target
# %%
sns.pairplot(data = irisdf, hue='Label')
# %%
