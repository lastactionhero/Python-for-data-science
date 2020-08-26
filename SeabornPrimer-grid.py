#%% 
#Import 
import seaborn as sns
import matplotlib as plt

# %%
# load iris data
iris = sns.load_dataset('iris')

# %%
iris.head()

# %%
sns.distplot(iris['sepal_length'])

# %%
# check with pairplot how data is distributed against each other
sns.pairplot(iris)

# %%
# check unique species
iris["species"].unique()

# %%
# lets look at the pairgrid. This gives you additional control over pairplot
g= sns.PairGrid(iris)
g.map_diag(sns.distplot)
#g.map_upper(sns.barplot)
g.map_lower(sns.kdeplot)
# %%
#FacetGrid
tips = sns.load_dataset("tips")
tips.head()

# %%
# Rows are smoker (yes, no) and coloums are time (dinner, lunch)
g=sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot,'total_bill')

# %%
# we can use facetgrid plot for scatterplot
g=sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.scatterplot,'total_bill','tip')

# %%
# adjust size of plots 
g=sns.FacetGrid(data=tips, col='time', row='smoker', size = 5)
g.map(sns.scatterplot,'total_bill','tip')