#%%
import pandas as pd
import seaborn as sns
import numpy as np

# %%
df = pd.read_csv("KNN_Project_Data")
# %%
df.head()
# %%
# standardise data
from sklearn.preprocessing import StandardScaler
# %%
stdscalar = StandardScaler()
# %%
scaled_model = stdscalar.fit(df[:-1])
# %%
scaled_data = scaled_model.transform(df[:-1])
# %%
scaled_data
# %%
from sklearn.neighbors import KNeighborsClassifier
#%%
from sklearn.model_selection import train_test_split
# %%
X = scaled_data
y=df["TARGET CLASS"]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# %%
error =[]
for i in np.arange(1,40):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    error.append(np.mean(pred!=y_test))

error
# %%
sns.lineplot(x=np.arange(1,40), y=error)
# %%
# Select value of 15 for neighbours count
knn= KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(pred,y_test))
# %%
