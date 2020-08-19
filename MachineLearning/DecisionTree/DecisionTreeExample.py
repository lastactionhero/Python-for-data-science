#%%
import pandas as pd
import seaborn as sns
import numpy as np
# %%
df = pd.read_csv('loan_data.csv')
# %%
df.head()
# %%
df.tail() 
# %%
sns.distplot(df[df['credit.policy']==1]['fico'])
# %%
sns.jointplot(x='credit.policy', y='fico', data=df)
# %%
sns.countplot(x='fico', data=df,hue='credit.policy')
# %%
df.columns
# %%
plotingdf = df[['credit.policy', 'fico', 'days.with.cr.line', 'inq.last.6mths', 'not.fully.paid']]
# %%
sns.pairplot(plotingdf, hue='not.fully.paid')
# %%
sns.set_style('darkgrid')
sns.set_palette('inferno')
sns.jointplot(x='fico', y='int.rate', data=df)
# %%
sns.lmplot(y='int.rate', x='fico', data=df, hue='credit.policy', col='not.fully.paid')
# %%
# get dummy columns for categorical columns
df_final = pd.get_dummies(data=df, columns=['purpose'], drop_first=True)
# %%
df_final.head()
# %%
from sklearn.model_selection import train_test_split
# %%
X=df_final.drop(columns='not.fully.paid', axis=1)
y=df_final['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
from sklearn.tree import DecisionTreeClassifier
# %%
dtree = DecisionTreeClassifier()
# %%
dtree.fit(X_train, y_train)
# %%
pred = dtree.predict(X_test)
# %%
from sklearn.metrics import classification_report, confusion_matrix
# %%
print('\n')
print(classification_report(y_true=y_test, y_pred=pred))
# %%
# we can see in above report, values with 1 as not.fully.paid has very less precision. lets have deep dive into data
sns.countplot(x='not.fully.paid',data=df_final)
# %%
from sklearn.ensemble import RandomForestClassifier
# %%
rnforest = RandomForestClassifier()
# %%
rnforest.fit(X_train, y_train)
# %%
rn_pred = rnforest.predict(X_test)
# %%
print('\n')
print(classification_report(y_test, rn_pred))
# %%
