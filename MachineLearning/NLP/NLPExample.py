#%%
import pandas as pd
import seaborn as sns
import numpy as np
# %%
yelp= pd.read_csv('yelp.csv')
# %%
yelp.head()
# %%
yelp.describe()
# %%
yelp.info()
# %%
yelp['message_len'] = yelp['text'].apply(len)
# %%
sns.set_style('darkgrid')
g = sns.FacetGrid(yelp,col='stars')
g.map(sns.distplot,'message_len')
# %%
sns.set_style('darkgrid')
g = sns.FacetGrid(yelp,col='stars')
g.map(sns.boxenplot,'message_len')
# %%
sns.boxplot(x='stars',y='message_len',data=yelp,palette='rainbow')
# %%
sns.countplot(x='stars',data=yelp)
# %%
yelp.groupby('stars').mean()
# %%
sns.set_palette('Set1')
sns.heatmap(yelp.groupby('stars').mean().corr(),annot=True)
# %%
reviews = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
# %%
X = reviews['text']
y = reviews['stars']
#%%
import string
from nltk.corpus import stopwords
def TextCleaning(message):
    nopunc= [char for char in message if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # remove stop word
    return[clean for clean in nopunc.split() if clean.lower() not in stopwords.words('english')]

# %%
# count vectorization
# tfidf
# classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# %%
ppl = Pipeline(
    [
        ('Vectorization', CountVectorizer(analyzer=TextCleaning)),
        ('TFIDF', TfidfTransformer()),
        ('Classify',LogisticRegression() )
    ]
)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ppl.fit(X_train, y_train)
# %%
y_pred = ppl.predict(X_test)
# %%
from sklearn.metrics import classification_report
# %%
print('\n')
print(classification_report(y_test,y_pred))
# %%
