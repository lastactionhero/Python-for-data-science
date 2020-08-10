#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
# %%
dfcustomers = pd.read_csv('ecommerce customers.csv')

# %%
dfcustomers.head()

# %%
sns.pairplot(dfcustomers)
#%%
sns.lmplot(data=dfcustomers,x='Length of Membership', y='Yearly Amount Spent')
# %%
sns.countplot(data=dfcustomers, x='Avatar')

# %%
dfcustomers['Domain']= dfcustomers['Email'].apply(lambda x : x.split('@')[1])

# %%
sns.countplot(data=dfcustomers, x='Domain')

# %%
dfcustomers['Domain'].unique()

# %%
sns.heatmap(dfcustomers.corr())

# %%
from sklearn.model_selection import train_test_split

# %%
X=dfcustomers.iloc[:,3:7]

# %%
X.head()

# %%
y= dfcustomers['Yearly Amount Spent']

# %%
y.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)

# %%
from sklearn.linear_model import LinearRegression

# %%
lrmodel= LinearRegression()

# %%
lrmodel.fit(X=X_train, y=y_train)

# %%
lrmodel.intercept_

# %%
lrmodel.coef_

# %%
pred = lrmodel.predict(X_test)

# %%
sns.scatterplot(x=y_test, y=pred)

# %%
from sklearn import metrics

# %%
metrics.mean_absolute_error(y_test, pred)

# %%
metrics.mean_squared_error(y_test, pred)

# %%
np.sqrt(metrics.mean_squared_error(y_test, pred))

# %%
#residuals
sns.distplot(y_test-pred)

# %%
metrics.explained_variance_score(y_test,pred)

# %%
