Python 3.7.2 (v3.7.2:9a3ffc0492, Dec 24 2018, 02:44:43) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Recommender Systems with Python

## Import Libraries

import numpy as np
import pandas as pd

## Get the Data

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

df.head()

Now let's get the movie titles:

movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

We can merge them together:

df = pd.merge(df,movie_titles,on='item_id')
df.head()

# EDA

## Visualization Imports

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

Now set the number of ratings column:

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

Now a few histograms:

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

## Recommending Similar Movies

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

Most rated movie:

ratings.sort_values('num of ratings',ascending=False).head(10)

Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

ratings.head()

Now let's grab the user ratings for those two movies:

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()

We can then use corrwith() method to get correlations between two pandas series:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

Let's clean this by removing NaN values and using a DataFrame instead of a series:

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie). 

corr_starwars.sort_values('Correlation',ascending=False).head(10)

Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

Now sort the values and notice how the titles make a lot more sense:

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

Now the same for the comedy Liar Liar:

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()   
