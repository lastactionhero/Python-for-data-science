#%%
import numpy as np
import pandas as pd
import seaborn as sns
# %%
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
# %%
df.head()
# %%
movie_titles= pd.read_csv('Movie_Id_Titles')
#%%
movie_titles.head()
# %%
# Join 2 data frames
df=pd.merge(df,movie_titles,how='inner',left_on='item_id', right_on='item_id')
# %%
sns.set_style('darkgrid')
sns.set_palette('Set1')
sns.distplot(df['rating'])
# %%
sns.countplot(df['rating'])
# %%
# top 10
df.value_counts(df['title']).head(10)
# %%
# group by most ratings
df.groupby('title')['rating'] \
    .mean()\
        .sort_values(ascending=False)
# %%
# group by number of ratings
df.groupby('title')['rating'] \
    .count()\
        .sort_values(ascending=False)
# %%
ratings = pd.DataFrame( df.groupby('title')['rating'].mean())
# %%
ratings.head()
# %%
ratings['num of ratings'] = df.groupby('title')['rating'].count()
# %%
sns.scatterplot(data=ratings, x='num of ratings', y='rating')
# %%
sns.set_style('darkgrid')
sns.set_palette('rainbow')
sns.jointplot(data=ratings,y='num of ratings', x='rating', kind='reg')

# %%
sns.distplot(ratings['rating'])
# %%
sns.distplot(ratings['num of ratings'])
# %%
# pivot table 
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
# %%
moviemat.head()
# %%
ratings.sort_values(by='num of ratings',ascending=False)
# %%
starwars = moviemat['Star Wars (1977)']
liar  = moviemat['Liar Liar (1997)']
# %%
similar_starwars = moviemat.corrwith(starwars)
# %%
similar_starwars.head()
# %%
# join on basis of index. if index are differnt then use merge
similar_starwars = pd.DataFrame(similar_starwars, columns=['corr'])
# %%
# now we have correlation score with each title. add num of ratings col
similar_starwars = similar_starwars.join(ratings)

# %%
similar_starwars[similar_starwars['num of ratings']>100].sort_values(by='corr', ascending=False)
# %%
