#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
college = pd.read_csv('college_data')
# %%
college.head()
# %%
college.info()
# %%
sns.pairplot(data= college, hue='Private')
# %%
sns.set_style('darkgrid')
sns.set_palette('Set1')
sns.scatterplot(data = college, x='Room.Board', y= 'Grad.Rate', hue='Private')
# %%
sns.scatterplot(data = college, x='Outstate', y= 'F.Undergrad', hue='Private')
# %%
g=sns.FacetGrid(data=college, hue='Private')
g.map(sns.distplot,'Outstate',kde=False)
# %%
g=sns.FacetGrid(data=college, col='Private')
g.map(sns.scatterplot, 'Outstate', 'F.Undergrad')
# %%
plt.figure(figsize=(50,50))
g=sns.FacetGrid(data=college, hue='Private', size=(50,50))
g.map(sns.distplot,'Grad.Rate',kde=False)
plt.tight_layout()
# %%
# Facetgrid plot with size and legends added
g=sns.FacetGrid(data=college, hue='Private', size=10)
g.map(sns.distplot,'Grad.Rate',kde=False)
g.add_legend()
# %%
# get school with highest grad rate
college['Grad.Rate'].describe()
# %%
college[college['Grad.Rate']==118]
# %%
college['Grad.Rate'][college['Grad.Rate']==118]=100
# %%
college.iloc[:,2:]
#%%
# lets do clustering
from sklearn.cluster import KMeans
# %%
km = KMeans(n_clusters=2)
# %%
km.fit(college.iloc[:,2:])
# %%
km.cluster_centers_
# %%
college['Cluster']=km.labels_
# %%
from sklearn.metrics import confusion_matrix,classification_report
#%%
# convert classification labels of private column to int
y = college['Private'].map(lambda x: 0 if x=='Yes' else 1)
# %%
print('\n')
print(classification_report(y,college['Cluster']))
# %%
