#%%
import nltk
import pandas as pd
import seaborn as sns
# %%
nltk.download_shell()

# %%
for line in open('SMSSpamCollection'):
    print(line)
# %%
sms = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
# %%
sms.head()

# %%
sns.countplot(sms['label'])
# %%
sms.describe()
# %%
sms.groupby('label').describe()
# %%
sms['length'] = sms['message'].apply(len)
# %%
sns.distplot(sms['length'],bins=10)
# %%
sns.set_palette('Set1')
sns.pairplot(sms,hue='label',size=8)
# %%
import string
from nltk.corpus import stopwords
# %%
def TextCleaning(message):
    nopunc= [char for char in message if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # remove stop word
    return[clean for clean in nopunc.split() if clean.lower() not in stopwords.words('english')]
# %%
print(TextCleaning('This is going to be best part!! isnt it?'))
# %%
# bag of words
from sklearn.feature_extraction.text import CountVectorizer
# %%
bow_transformer = CountVectorizer(analyzer=TextCleaning).fit(sms['message'])
# %%
message4 = sms['message'][3]
message4
# %%
bow4 = bow_transformer.transform([message4])
print(bow4)
# %%
bow_transformer.get_feature_names()[9554]
# %%
bow_message= bow_transformer.transform(sms['message'])
# %%
# this matric is created as word, message1, message2... format. if word present it will have value 1 in corresponding message column
# Due to this entire matric has lots of 0 in it
bow_message.shape
# %%
# get number of non zero cells
bow_message.nnz
# %%
from sklearn.feature_extraction.text import TfidfTransformer
# %%
tfidf = TfidfTransformer().fit(bow_message)
# %%
tfidf4 = tfidf.transform(bow4)
print(tfidf4)
#%%
# %%
tfidf_messages = tfidf.transform(bow_message)
# %%
from sklearn.naive_bayes import MultinomialNB
# %%
spam_detect_model = MultinomialNB().fit(tfidf_messages,sms['label'])
# %%
spam_detect_model.predict(tfidf4)
# %%
sms.iloc[3]
# %%
from sklearn.model_selection import train_test_split
#%%
message_train, message_test, label_train, label_test = train_test_split(sms['message'],sms['label'],test_size=0.30)
#%%
from sklearn.pipeline import Pipeline
# %%
pipeline = Pipeline(
    [
        ('bow',CountVectorizer(analyzer=TextCleaning)),
        ('tfidf', TfidfTransformer()),
        ('classifier',MultinomialNB())
    ]
)
# %%
pipeline.fit(message_train,label_train)
# %%
label_predict = pipeline.predict(message_test)
# %%
from sklearn.metrics import classification_report
print(classification_report(label_predict, label_test))