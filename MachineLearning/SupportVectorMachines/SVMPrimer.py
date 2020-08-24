#%%
import pandas as pd
import numpy as np
import seaborn as sns
# %%
from sklearn.datasets import load_breast_cancer
# %%
cancer = load_breast_cancer()
# %%
cancer
# %%
print(cancer['DESCR'])
# %%
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
# %%
df.head()
# %%
from sklearn.model_selection import train_test_split
# %%
X=df
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
from sklearn.svm import SVC
# %%
svc = SVC()
# %%
svc.fit(X_train, y_train)
# %%
y_pred = svc.predict(X_test)
# %%
from sklearn.metrics import classification_report
# %%
print('\n')
print(classification_report(y_test,y_pred))
#%% 
svc.get_params()
# %%
# lets see what value of C and Gamma is optimal
# C decide on how much vector nodes can lie inside hyper plane. Lower value of C indicate low bias and may lead to high variance
# Gamma is constant value in the equation
from sklearn.model_selection import GridSearchCV
# %%
param_grid = {'C':[1,10,20,30,100], 'gamma' : [1,0.1,0.001,0.0001]}
# %%
grid= GridSearchCV(svc,param_grid=param_grid, verbose=10)
# %%
grid.fit(X_train, y_train)
# %%
grid.best_params_
# %%
# now predict using grid search cv
pred = grid.predict(X_test)
# %%
print('\n')
print(classification_report(y_test,pred))
# %%
