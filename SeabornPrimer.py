#%%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# %% [markdown]
tips= sns.load_dataset('tips')
tips.head(20)
# %%
# ##distribution plot
sns.distplot(tips['total_bill'])

# %% [markdown]
#joint graph
sns.jointplot(x='total_bill', y='tip', data=tips)

# %%
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')

# %%
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')

# %%
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')

# %%
#Pairplot - this plot takes all dataset and plot pair of numerical columns against each other
sns.pairplot(tips,hue='sex')


# %% 
sns.rugplot(tips["total_bill"])                    


# %%
# Categorical plots
#default average/mean values per group shown
sns.barplot(x='sex',y='total_bill',data=tips)
#%%
#estimator can affect grouping
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)

# %%
#count plt
sns.countplot(x='sex', data = tips)

# %%
sns.boxplot(x='day',y='total_bill', data=tips, hue='smoker')

# %%
#Violinplot - they are same as box plot but say more infomration 
sns.violinplot(x='day',y='total_bill', data=tips)

# %%
sns.violinplot(x='day',y='total_bill', data=tips, hue='sex', split=True)

# %%
sns.boxplot(x='day',y='total_bill',data=tips, hue='sex')

# %%
sns.stripplot(x='day', y='total_bill', data=tips)
# %%
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)

# %%
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex',jitter=True)

# %%
sns.swarmplot(x='day', y='total_bill', data=tips)
# %%
sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex')

# %%
sns.violinplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')

# %%
sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')

#%%
g = sns.catplot(x="sex", y="total_bill",
            hue="smoker", col="time",
                data=tips, kind="box",
                height=4, aspect=.7)
# %%
flights = sns.load_dataset('flights')
flights.head()
#%% Pivot table
pivt = flights.pivot_table(index="month", columns="year", values="passengers")
# %%
sns.heatmap(pivt)

# %%
plt.figure(figsize=(12,6))
sns.clustermap(pivt)

# %% We may need to normalize scale to better understand clusters
sns.clustermap(pivt, standard_scale=1)


# %%
