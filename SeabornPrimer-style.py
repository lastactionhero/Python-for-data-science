#%%
import seaborn as sns

# %%
tips = sns.load_dataset('tips')

# %%
tips.head()

# %%
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
#set style
sns.set_style('whitegrid')
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
sns.set_style('darkgrid')
sns.set_palette('inferno')
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
sns.set_style('ticks')
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
sns.set_context('poster')
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
sns.set_context('talk')
sns.barplot(data=tips, x='sex',y='total_bill')

# %%
sns.set_context('notebook')
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker')

# %%
# color map - https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker',palette='coolwarm')

# %%
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker',palette='copper')

# %%
# try with plasma, hot, inferno
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker',palette='inferno')

# %%
