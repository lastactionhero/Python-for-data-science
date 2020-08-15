#%%
import pandas as pd
import numpy as np
import seaborn as sns


# %%
df = pd.read_csv('advertising.csv')

# %%
df.head()
# %%
df.info()
# %%
sns.pairplot(df)
#%%
sns.jointplot(x='Age',y='Area Income',data=df)
# %%
sns.heatmap(df.isnull())

# %%
sns.scatterplot(data=df,x='Age',y='Area Income',hue='Clicked on Ad')

# %%
sns.scatterplot(data=df,x='Age',y='Daily Time Spent on Site',hue='Clicked on Ad')


# %%
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=df,color='green')
# %%
##sns.set(style="ticks", color_codes=True)
sns.pairplot(df, hue='Clicked on Ad', palette='Set2'  )
# %%
# Lets prepare data set by preprocessing
import datetime as date
# %%
# Day of week as well as time of day migth be essential to decide on clicking of ad
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# %%
df['Dayofweek'] = df['Timestamp'].dt.dayofweek
# %%
df['Month'] = df['Timestamp'].dt.month
# %%
sns.countplot(data=df, x=df['Timestamp'].dt.hour)
#%%
sns.countplot(data=df, x='Month',hue='Clicked on Ad')
# both month and dayofweek has no major difference
#%%
# lets create 4 buckets of these 24 hours
df['hourbucket'] = np.round(pd.to_numeric(df['Timestamp'].dt.hour/8))
#df['Hourgrp']= 
# %%
sns.countplot(data=df, x='hourbucket', hue= 'Clicked on Ad')

#%%
df.drop(['Dayofweek','Month'], axis=1, inplace=True)
#%%
# lets see if country makes any difference in add click
sns.countplot(data=df, x='Country', hue= 'Clicked on Ad')
# not any difference
#%%
sns.pairplot(df, hue='Clicked on Ad')
# %%
# Lets use logistic regression
from sklearn.model_selection import train_test_split

# %%
df.drop(['Ad Topic Line','City','Country'], axis=1, inplace=True)
# %%
df.columns
# %%
X= df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male', 
       'hourbucket']]
# %%
y= df['Clicked on Ad']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
from sklearn.linear_model import LogisticRegression

# %%
# 3 steps
logmodel = LogisticRegression()
# %%
logmodel.fit(X_train, y_train)
# %%
y_pred = logmodel.predict(X_test)
# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# %%
