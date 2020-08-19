#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df=pd.read_csv('kyphosis.csv')
# %%
df.head()
#%%
sns.set(style="ticks", color_codes=True)
# %%
# lets do some exploratory  analysis 
sns.pairplot(df, hue='Kyphosis', diag_kind='hist', palette='Set1')
# %%
sns.countplot(hue='Kyphosis', x='Start', data = df)
# %%
# lets split data
from sklearn.model_selection import train_test_split
# %%
# Excelpt first column
X= df.iloc[:,1:]
# %%
y= df['Kyphosis']
# %%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)
# %%
# Single decision tree
from sklearn.tree import DecisionTreeClassifier
# %%
dtree = DecisionTreeClassifier()
# %%
dtree.fit(X_train, y_train)
# %%
pred = dtree.predict(X_test)
# %%
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
# %%
print(classification_report(y_test,pred))
# %%
# now lets see if we can improve score using random forest
from sklearn.ensemble import RandomForestClassifier
# %%
rforest = RandomForestClassifier(n_estimators=200)
# %%
rforest.fit(X_train, y_train)
# %%
rpred = rforest.predict(X_test)
# %%
print(classification_report(y_test, rpred))
# %%
cd = ConfusionMatrixDisplay(confusion_matrix(y_test, rpred))
 
# %%
print(confusion_matrix(y_test, rpred))
# %%
cd.plot()
# %%
