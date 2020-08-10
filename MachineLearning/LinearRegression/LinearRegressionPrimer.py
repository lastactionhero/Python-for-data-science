#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
 
# %%
dfhousing = pd.read_csv("USA_Housing.csv")

# %%
dfhousing.tail()
# %%
dfhousing.head()

# %%
dfhousing.info()

# %%
dfhousing.describe()

# %%
sns.pairplot(data=dfhousing)

# %%

sns.distplot(dfhousing['Price'])

# %%
correlation = dfhousing.corr()

# %%
sns.heatmap(data=correlation, cmap='copper')

# %%
sns.heatmap(data=correlation, cmap='YlGnBu')

# %%
sns.heatmap(data=correlation)
#%%
print(dfhousing.columns)
# %%
#X= dfhousing.loc[:, dfhousing.columns != 'Price' or dfhousing.columns != 'Address']
X= dfhousing.loc[:, dfhousing.columns != 'Price']
X= X.loc[:,X.columns != 'Address']
# %%
y = dfhousing['Price']

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

# %%
from sklearn.linear_model import LinearRegression

# %%
lrmodel= LinearRegression()

# %%
lrmodel.fit(X=X_train, y=y_train)

# %%
# location of the line intercepting at y axis when x=0
lrmodel.intercept_

# %%
# Correlation coefficient
lrmodel.coef_


# %%
predictions = lrmodel.predict(X_test)

# %%
# Compare y_test with predictions i.e. actual vs predicted values using scatterplot
sns.scatterplot(x=y_test, y=predictions) 

# %%
#Residuals is difference in actual and predicted
# if its normal distribution it means model is good and correct choice for data.
sns.distplot(y_test-predictions)

# %%
# measure performance using Mean Absolute Error (MAE) or Mean Squared Error (MSE) or Root Mean Square Error
from sklearn import metrics

# %%
metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)
#%%
metrics.mean_squared_error(y_true=y_test, y_pred=predictions)

# %%
np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=predictions))

# %%
