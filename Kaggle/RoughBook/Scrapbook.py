#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#%%
nltk.download('all')
#%%
df = pd.read_csv('SMSSpamCollection.txt', sep='\t',names=['label','message'])
# %%
df.head()
# %%
df.shape
# %%
df['label'].value_counts()
# %%
target = df['label']
# %%
def split_tokens(message):
  message=message.lower()
  word_tokens =word_tokenize(message)
  return word_tokens
#%%
df['tokenized_message'] = df.apply(lambda row: split_tokens(row['message']),axis=1)
# %%
def split_into_lemmas(message):
    lemma = []
    lemmatizer = WordNetLemmatizer()
    for word in message:
        a=lemmatizer.lemmatize(word)
        lemma.append(a)
    return lemma
# %%
df['lemmatized_message'] = df.apply(lambda row: split_into_lemmas(row['tokenized_message']),axis=1)
# %%
df.head()

# %%
def stopword_removal(message):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    filtered_sentence = ' '.join([word for word in message if word not in stop_words])
    return filtered_sentence
# %%
df['preprocessed_message'] = df.apply(lambda row: stopword_removal(row['lemmatized_message']),axis=1)

# %%
print('Tokenized message:',df['tokenized_message'][11])
print('Lemmatized message:',df['lemmatized_message'][11])
print('preprocessed message:',df['preprocessed_message'][11])
# %%
def stopword_removal(message):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    filtered_sentence = ' '.join([word for word in message if word not in stop_words])
    return filtered_sentence
# %%
df['preprocessed_message'] = df.apply(lambda row: stopword_removal(row['lemmatized_message']),axis=1)
Training_data=pd.Series(list(df['preprocessed_message']))
Training_label=pd.Series(list(df['label']))
# %%
tf_vectorizer = CountVectorizer(ngram_range=(1, 2),min_df = (1/len(Training_label)), max_df = 0.7)
Total_Dictionary_TDM = tf_vectorizer.fit(Training_data)
message_data_TDM = Total_Dictionary_TDM.transform(Training_data)
# %%
df = pd.read_csv('training.txt',sep='\t',names=['label','message'])
# %%
df.head()
# %%
df.shape
# %%
df['label'].value_counts()
# %%
#Write your code here
import pandas as pd
import numpy as np
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5'])
print(heights_A.shape)
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
print(weights_A.dtypes)
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
print(df_A.shape)
np.random.seed(100)
heights_B = pd.Series(data= np.random.normal(loc=170,scale=25.0,size=(5)), index=['s1','s2','s3','s4','s5'])
weights_B = pd.Series(data= np.random.normal(loc=75.0,scale=12.0,size=(5)), index=['s1','s2','s3','s4','s5'])
print(heights_B.mean())
df_B=pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
print(df_B.columns)

#%%
heights_A[1:-1]
#%%
df_B.head()
# %%
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
# %%
df.shape
# %%
df.head()
# %%
s = pd.Series([99, 32, 67],list('abc'))
# %%
s
# %%
s.isin([67,32])
# %%
np.arange(10, 16)
# %%
#Write your code here
#Write your code here
import pandas as pd 
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5'])
print(heights_A[1])
print(heights_A[1:-1])
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
height=df_A['Student_height']
print(type(height))
df_s1s2=df_A.iloc[:2]
print(df_s1s2)
df_s2s5s1= df_A.loc[['s2','s5','s1']]
print(df_s2s5s1)
df_s1s4 = df_A.loc[df_A.index.str.endswith('1') | df_A.index.str.endswith('4')]
print(df_s1s4)
# %%
df_A.head()
# %%
df_s1s2=df_A.iloc[:2]
df_s1s2
# %%
df_s2s5s1= df_A.loc[['s2','s5','s1']]
df_s2s5s1
# %%
df_A.loc[df_A.index.str.endswith('1') | df_A.index.str.endswith('4')]
# %%
df = pd.DataFrame({'A':[34, 78, 54], 'B':[12, 67, 43]}, index=['r1', 'r2', 'r3'])
# %%
df.iloc[1]
# %%
df.loc['r4'] = [67, 78]
# %%
df
# %%
#Write your code here
#Write your code here
#Write your code here
import pandas as pd
import numpy as np 
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5']) 
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A.to_csv('classA.csv')

df_A2 = pd.read_csv('classA.csv')
print(df_A2)

df_A3 = pd.read_csv('classA.csv',index_col=0)
print(df_A3)
#%%
np.random.seed(100)
heights_B = pd.Series(data= np.random.normal(loc=170,scale=25.0,size=(5)), index=['s1','s2','s3','s4','s5'])
weights_B = pd.Series(data= np.random.normal(loc=75.0,scale=12.0,size=(5)), index=['s1','s2','s3','s4','s5'])
df_B=pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
df_B.to_csv('classB.csv',index=False)

df_B2 = pd.read_csv('classB.csv')
print(df_B2)

df_B3 = pd.read_csv('classB.csv', header=None)
print(df_B3)

df_B4 = pd.read_csv('classB.csv', header=None, skiprows=2)
print(df_B4)
#%%
dates = pd.date_range('2017-09-01','2017-09-15')
dates[2]
# %%
datelist = ['14-Sep-2017','9-Sep-2017']
dates_to_be_searched = pd.to_datetime(datelist)
# %%
dates_to_be_searched.isin(datelist)
# %%
arraylist = [['classA']*5 + ['classB']*5, ['s1','s2','s3','s4','s5']*2]
arraylist
# %%
pd.DataFrame(arraylist)
# %%
mi_index= pd.MultiIndex.from_arrays(arraylist,names=['first','second'])
mi_index.get_level_values(1)
# %%
#Write your code here
import pandas as pd 
dates = pd.date_range('2017-09-01','2017-09-15')
print(dates[2])
datelist = ['14-Sep-2017','9-Sep-2017']
dates_to_be_searched = pd.to_datetime(datelist)
print(dates_to_be_searched)
print(dates_to_be_searched.isin(datelist))
arraylist = [['classA']*5 + ['classB']*5, ['s1','s2','s3','s4','s5']*2]
mi_index = pd.MultiIndex.from_arrays(arraylist,names=['first','second'])
print(mi_index.get_level_values(0))
print(mi_index.get_level_values(1))
#%%
pi = pd.period_range('11-Sep-2017', '17-Sep-2017', freq='M')
pi
# %%
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5']) 
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A.loc['s3'] = np.nan
df_A.loc['s5','Student_weight'] = np.nan

# %%
df_A.dropna()
# %%
df = pd.DataFrame({'temp':pd.Series(28 + 10*np.random.randn(10)),
                   'rain':pd.Series(100 + 50*np.random.randn(10)),
                   'location':list('AAAAABBBBB')
})
df.head()

# %%
replacements = {'location':{'A':'Mumbai','B':'Pune'}}
# %%
df.replace(replacements, regex=True,inplace=True)
df
# %%
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5']) 
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})

# %%
df_A_filter1 = df_A[(df_A['Student_height']>160.0) & (df_A['Student_weight']<80.0)]
# %%
df_A_filter2= df_A.loc[df_A.index.str.endswith('5')]
df_A_filter2
# %%
df_A['Gender'] = ['M','F','M','M','F']
# %%
df_groups=df_A.groupby(by='Gender')
#%%
df_groups.groups
#%%
df_groups.mean()
# %%
df.iloc[:,lambda x:[0,2]]
# %%
df.groupby(df.index.str.len())
# %%

df = pd.DataFrame({'A':[34, 78, 54], 'B':[12, 67, 43]}, index=['r1', 'r2', 'r3'])
# %%
df[:2]
# %%
df.iloc[:2]
# %%
df = pd.DataFrame(index=[ 'r1', 'r2', 'r3', 'row4', 'row5', 'row6', 'r7', 'r8', 'r9', 'row10'], data= np.arange(0,10))
# %%
df
# %%
g = df.groupby(df.index.str.len())
#%%
g.groups
#%%
g.filter(lambda x: len(x) > 2)
# %%
s = pd.Series([89.2, 76.4, 98.2, 75.9], index=list('abcd'))
# %%
'b' in s
# %%
g = df.groupby(df.index.str.len())
g.aggregate({'A':len, 'B':np.sum})
# %%
df.iloc[1]
# %%
df.loc['r2':'r3']
# %%
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5']) 
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A['Gender'] = ['M','F','M','M','F']
s = pd.Series(name='s6',data=[165.4,82.7,'F'], index=['Student_height','Student_weight','Gender'])

df_AA = df_A.append(s)
# %%
df_A
# %%
np.random.seed(100)
heights_B = pd.Series(data= np.random.normal(loc=170,scale=25.0,size=(5)), index=['s1','s2','s3','s4','s5'])
weights_B = pd.Series(data= np.random.normal(loc=75.0,scale=12.0,size=(5)), index=['s1','s2','s3','s4','s5'])
df_B=pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
df_B.index=['s7','s8','s9','s10','s11']
df_B['Gender']  = ['F','M','F','F','M']
df_B

# %%
df= pd.concat([df_AA, df_B],axis=0)
df.reset_index()
# %%
#Write your code here
import pandas as pd 
import numpy as np 
heights_A = pd.Series(data=[176.2,158.4,167.6,156.2,161.4], index = ['s1','s2','s3','s4','s5']) 
weights_A= pd.Series(data=[85.1,90.2,76.8,80.4,78.9], index=['s1','s2','s3','s4','s5'])
df_A=pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A['Gender'] = ['M','F','M','M','F']
s = pd.Series(name='s6',data=[165.4,82.7,'F'], index=['Student_height','Student_weight','Gender'])
df_AA = df_A.append(s)
print(df_AA)

np.random.seed(100)
heights_B = pd.Series(data= np.random.normal(loc=170,scale=25.0,size=(5)), index=['s1','s2','s3','s4','s5'])
weights_B = pd.Series(data= np.random.normal(loc=75.0,scale=12.0,size=(5)), index=['s1','s2','s3','s4','s5'])
df_B=pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
df_B.index=['s7','s8','s9','s10','s11']
df_B['Gender']  = ['F','M','F','F','M']
df = pd.concat([df_AA,df_B],axis=0)
print(df)
# %%
