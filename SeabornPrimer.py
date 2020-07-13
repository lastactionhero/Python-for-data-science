#%%
import seaborn as sns

# %%
tips= sns.load_dataset('tips')
tips.head(20)
# %%
# distribution plot
sns.distplot(tips['total_bill'])

# %%
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
sns.pairplot(tips)


# %%
