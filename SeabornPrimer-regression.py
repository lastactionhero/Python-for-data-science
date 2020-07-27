#%%
import seaborn as sns
# %%
tips = sns.load_dataset('tips')
# %%
tips.head()
# %%
sns.pairplot(tips)

# %%
#Linear plot
sns.lmplot(data=tips, x='tip', y='total_bill', hue='time')

# %%
# Interesting, like facetgrid we can pass cols too
sns.lmplot(x='tip',y='total_bill',data=tips,col='time', row='smoker')

# %%
# we can also add hue to it
iris = sns.load_dataset('iris')
#%%
iris.head()
#%%
sns.distplot(iris['petal_length'])
# %%
def getCategory(pl):
    if(pl<2):
        return 2
    elif(pl<4):
        return 4
    elif(pl<6):
        return 6
    elif(pl<8):
        return 8
    else:
        return 0
#%%
iris['petal_category']=iris['petal_length'].apply(getCategory)
# %%
sns.lmplot(x='sepal_length',y='sepal_width',data=iris,col='species', row='petal_category')





# %%
