#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# %%
train = pd.read_csv("train.csv")

# %%
train.head()

# %%
sns.countplot(data=train,x='Pclass',hue='Survived')

# %%
sns.distplot(train['Fare'])

# %%
# check correctness of data 
sns.heatmap(data=train.isnull(),yticklabels=False, cbar=False)
# %%
sns.pairplot(data=train)
# %%
sns.countplot(data=train, x='Survived', hue='Sex')
#%%
sns.countplot(data=train, x='Survived', hue='Pclass')
# %%
sns.distplot(train['Age'].dropna())

# %%
plt.figure(figsize=(12,4))
sns.distplot(train['Fare'].dropna(), kde=False)

# %%
# We need to take care of missing data
sns.boxplot(data=train, x='Pclass', y='Age', hue='Survived')

# %%
# age can be calculated usign all other features but lets keep it simple and take average of class
AgeMean = train.groupby(by='Pclass')['Age'].mean().astype(int).astype(float)
AgeMean
# %%
AgeMean.loc[2]

# %%
train[(train['Age'].isna()) & (train['Pclass']==2)]
 
# %%
train.info()

# %%
# following code is not working, lets try creating function and calling it
train[(train['Age'].isna()) & (train['Pclass']==2)]['Age']=29.0
#%%

# %%
def SetAge(cols):
    Age = cols[0]
    Pclass= cols[1]
    if(pd.isnull(Age)):
        if(Pclass==1):
            return 38
        if(Pclass==2):
            return 29
        else:
            return 24
    else:
        return Age

# %%
train['Age']=train[['Age','Pclass']].apply(SetAge, axis=1)

# %%
train[(train['Age'].isna()) & (train['Pclass']==2)]['Age']

# %%
# check correctness of data 
sns.heatmap(data=train.isnull(),yticklabels=False, cbar=False)

# %%
# Cabin has more null values and of no use for analysis of ML
train.drop('Cabin', axis=1, inplace=True)

# %%
# We can see columns like Sex and Embarked has non numberic values. Lets convert it to numeric
Sex= pd.get_dummies(train['Sex'], drop_first=True)
Sex
# %%
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
Embarked

# %%
train = pd.concat([train, Sex, Embarked], axis=1 )
#%%
train.head()


# %%
train.drop(['Sex', 'Embarked', 'Ticket', 'Name'], axis=1,  inplace=True)
#%%
# Understand axis 
subset = train[['Fare','Age']][:3]
subset
# %%
subset.apply(lambda x : x[0]+x[1], axis=1)

# %%
