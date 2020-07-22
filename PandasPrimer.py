# %%
import pandas as pd
import numpy as np
# %%
labels=['a','b','c']
# %%
mydata = np.arange(10,13)
mydata
# %%
d={'a':10,'b':11,'c':12}
d
# %%
ser1 = pd.Series(data=mydata, index=labels)
ser2 = pd.Series(data = [100,100,100,100], index = ['a','b','c','d'])
#addition can be done as per index. if index not found, NAN result is stored for that location
ser1+ser2
# %%
# dictionaly key is converted to index and value as data
pd.Series(d)
# %%
#data frame
np.random.seed(101)
df=pd.DataFrame(np.random.randn(5,4), ['a','b','c','d','e'],['w','x','y','z'])
df
# %%
df['w']

# %%
df.w

# %%
df[['w','z']]

# %%
df['new']=df['x']+df['z']
df

# %%
df.drop('new',axis=1)
# %%
df
df.drop('new',axis=1,inplace=True)
# %%
df

# %%
#select rows
df.loc['a']

# %%
# row c
df.iloc[2]

# %%
df.loc['a','w']

# %%
df.loc[['a','b'],['x','y']]

# %%
df[df>0]

# %%
#generally used on columns
df[df['w']>0]

# %%
df[df['w']>0][['x','y']]

# %%
# use multiple conditions
# Normal 'and' operator can not work here since we are comparing series here
df[(df['w']>0) & (df['x']<0)]

# %%
# reset index - create column with current index values 
df.reset_index()

# %%
#missing value
df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})

# %%
df
# %% 
# drop rows with null or nan values
df.dropna()
# %%
# drop cols with null or nan values
df.dropna(axis=1)
# %%
# drop rows with null or nan values with more than 1 nan value
df.dropna(thresh=2)

# %%
df.fillna(value='replaced value')

# %%
df['A'].fillna(value=df['A'].mean())
# %%
df['A'].fillna(value=0)
# %%
#Group by 
data = {'City':['Seattle','Pune','Seattle','Mumbai','Pune','Seattle'],
       'Customer':['Amit','Amy','Sunny','Sam','Jack','Reacher'],
       'PurchaseAmount':[200,210,100,150,250,350]}
data
# %%
df = pd.DataFrame(data)
df
# %%
df.groupby(by='City').sum()

#%%
df.groupby("City").agg({"PurchaseAmount":pd.Series.unique})
# %%
df.groupby(by='City').describe()

# %%
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()

# %%
df[(df['col1']>2) & (df['col2']==444)]

# %%
df['col3'].apply(len)

# %%
df.sort_values(by='col3', ascending=False)

# %%
#conda install sqlalchemy
#conda install lxml
#conda intall hhml5lib
#conta install BeautifulSoup4
mydf = pd.read_csv('example')

# %%
mydf

# %%
