#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
ecalls = pd.read_csv('911.csv')

# %%
ecalls.head()
#%%
ecalls.describe()
# %%
ecalls['title'].unique()

# %%
ecalls.groupby(by='zip')['title']  \
        .count() \
        .nlargest(10)
# %%
dfcounts = ecalls.groupby(by='zip')['title']   \
        .count() \
        .reset_index(name='count') 
#%%
dfcounts.head()        
#%%

sns.distplot(ecalls['zip'])

# %%
sns.barplot(x='zip', y='count',data=dfcounts)

# %%
ecalls['zip'].value_counts()

#%%
sns.countplot(data=ecalls,x='zip')

# %%
ecalls['Reason']=ecalls['title'].apply(lambda x:x.split(':')[0])

# %%
sns.countplot(data=ecalls, x='Reason')
#%%
ecalls['timeStamp']= ecalls['timeStamp'].apply(pd.to_datetime)
# %%
ecalls['Hour']= ecalls['timeStamp'].apply(lambda x:x.hour)
ecalls['Month'] = ecalls['timeStamp'].apply(lambda x:x.month)
ecalls['Day of Week'] = ecalls['timeStamp'].apply(lambda x:x.dayofweek)

# %%
sns.countplot(data=ecalls, x='Month', hue='Reason')

# %%
sns.countplot(data=ecalls, x='Month', hue='Reason')

# %%
# add names to day of week
weeknames = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# %%
ecalls['Day of Week']= ecalls['Day of Week'].map(weeknames)

# %%
ecalls.head()

# %%
sns.countplot(data=ecalls, x='Day of Week', hue='Reason')
#%%
sns.scatterplot(data=ecalls.reset_index(),x='Month',y='timeStamp')
#%%
callsbymonth = ecalls.groupby(by='Month').count()
callsbymonth.head()
# %%
sns.barplot(data=callsbyweek.reset_index(),x='Day of Week', y='timeStamp')

# %%
callsbyweek = ecalls.groupby(by='Day of Week').count()
# %%
callsbyweek.head()

# %%
sns.barplot(data=callsbyweek.reset_index(),x='Day of Week', y='timeStamp')

# %%
callsbyhour =  ecalls.groupby(by='Hour').count()

# %%
callsbyhour.head()

# %%
sns.barplot(data=callsbyhour.reset_index(), x='Hour', y='timeStamp')

# %%
ecalls['Date']= ecalls['timeStamp'].apply(lambda x: x.date())

# %%
callsbyday = ecalls.groupby(by='Date').count().reset_index()
#%%
callsbyday.head()
# %%
sns.lineplot(x='Date',y='timeStamp',data=callsbyday)

# %%
ecalls.head()
#%%
reasonwise= ecalls.groupby(by=['Date','Reason']).count().reset_index()
reasonwise.head()
# %%
sns.lineplot(data=reasonwise, x='Date',y='timeStamp', hue='Reason' )

# %%
heatdata = ecalls.groupby(by=['Day of Week', 'Hour']).count()['timeStamp'].unstack()
heatdata.head()
# %%
plt.figure(figsize=(12,6))
sns.heatmap(heatdata)

# %%
sns.clustermap(heatdata,standard_scale=1)

# %%