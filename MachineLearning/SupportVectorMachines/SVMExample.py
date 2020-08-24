#%%
import numpy as np
import pandas as pd
import seaborn as sns
# %%
iris = sns.load_dataset('iris')
# %%
iris.head()
# %%
iris.describe()
# %%
sns.set_palette('Dark2')
sns.set_style('darkgrid')
sns.pairplot(iris,hue='species', diag_kind='auto')
# %%
dt = iris[iris['species']=='setosa'][['sepal_length', 'sepal_width']]

sns.kdeplot(data=dt)
# %%
sns.kdeplot(dt['sepal_length'],dt['sepal_width'], cmp='inferno')
# %%
# Test train split
from sklearn.model_selection import train_test_split
# %%
iris.iloc[:,-1]
# %%
X= iris.iloc[:,:-1]
y=iris.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# %%
from sklearn.svm import SVC
# %%
svc_model = SVC()
# %%
svc_model.fit(X_train, y_train)
# %%
pred= svc_model.predict(X_test)
#%%
pred
# %%
from sklearn.metrics import classification_report
# %%
print('\n')
print(classification_report(y_test,pred))
# %%
