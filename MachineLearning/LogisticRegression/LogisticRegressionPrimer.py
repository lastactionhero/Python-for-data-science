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
# check correctness of data and understand null values volume
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
#following code is working
train['Age'][(train['Age'].isna()) & (train['Pclass']==2)] = 29.0
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
# We can see columns like Sex and Embarked has non numberic values. Lets convert it to numeric.
# Converting categorical data to numeric  
Sex= pd.get_dummies(train['Sex'], drop_first=True)
Sex
# %%
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
Embarked
#%%
# Try without converting Pclass to numerical first note down score than compare it after you convert it to numeric columns
# After comparison you will see that its ok to keep numerical columns unchanged 
Pclass= pd.get_dummies(train['Pclass'], drop_first=True)
Pclass
# %%
train = pd.concat([train, Sex, Embarked], axis=1 )
#%%
train.head()


# %%
train.drop(['Sex', 'Embarked', 'Ticket', 'Name','PassengerId'], axis=1,  inplace=True)
#%%
train.drop(['Pclass'], axis=1, inplace=True)
# Understand axis 
subset = train[['Fare','Age']][:3]
subset
# %%
subset.apply(lambda x : x[0]+x[1], axis=1)

# %%
# Now we have data cleaned, lets use logistic regression. 
# Step 1 create  train test split
from sklearn.model_selection import train_test_split

#%%
X= train.loc[:,train.columns[train.columns != 'Survived']]
X
# %%
y= train['Survived']
y
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# %%
# 3 step process to apply model 
from sklearn.linear_model import LogisticRegression

# %%
logmodel = LogisticRegression()

# %%
logmodel.fit(X_train, y_train)

# %%
logmodel.intercept_

# %%
logmodel.coef_

# %%
y_predict= logmodel.predict(X_test)
y_predict
# %%
#Evaluate model performance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
print(classification_report(y_test, y_predict))

# %%
# predict data in test file
test = pd.read_csv("test.csv")
test.head()

# %%
# Data cleaning and  preprocessing
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)

# %%
# Replace missing data in Age with mean by pcclass
test['Age']=test[['Age','Pclass']].apply(SetAge, axis=1)

#%%
# while evaluating model we found that there is NAN value for fare
# %%
test.isnull().any()
#%%
test[test['Fare'].isnull()]
#%%
# Replace such fare value with Pclass wise mean
test.groupby(by=['Pclass'])['Fare'].mean()
#%%
# Set this mean value of fare 
test['Fare'][test['Fare'].isnull()]=12.45
# %%
# Drop Cabin
test.drop('Cabin', axis=1, inplace=True)

# %%
Sex= pd.get_dummies(test['Sex'], drop_first=True)
Sex

# %%
Embarked = pd.get_dummies(test['Embarked'], drop_first=True)
Embarked

# %%
test = pd.concat([test, Sex, Embarked], axis=1 )

# %%
test.drop(['Sex','Embarked','Ticket', 'Name','PassengerId'],axis=1, inplace=True)

#%%
sns.pairplot(test)
# %%
test.describe()

# %%
y = logmodel.predict(test)
#%%
y
#%%
# convert it to data frame
Survived = pd.DataFrame(y,columns=['Survived'])
Survived.head()
# %%
submitdata = pd.read_csv("test.csv")

# %%
submitdata = pd.concat([submitdata,Survived],axis=1)

# %%
submitdata.head()

# %%
# Get only 2 columns out of predicted data
submitdata = submitdata[['PassengerId','Survived']]


# %%
submitdata.to_csv("SubmitData.csv", index=False)

# %%
