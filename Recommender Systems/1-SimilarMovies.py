import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('source/ml-100k/u.data', sep='\\t', names=r_cols, usecols=range(3), engine='python')
m_cols = ['movie_id', 'title']
movies = pd.read_csv('source/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), engine='python')
ratings = pd.merge(movies, ratings)
#print ratings.head()

#Now the amazing pivot_table function on a DataFrame will construct a user / movie rating matrix. Note how NaN indicates missing data - movies that specific users didn't rate.
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
#print movieRatings.head()

#Extract starWars
starWarsRatings = movieRatings['Star Wars (1977)']
#print starWarsRatings.head()

#Make the correlation and order by high correlation
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
#print similarMovies.sort_values(ascending = False)

#---------------Improve-----------

import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
# print movieStats.head()

#Let's get rid of any movies rated by fewer than 100 people, and check the top-rated ones that are left:

popularMovies = movieStats['rating']['size'] >= 100
print movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
print df.sort_values(['similarity'], ascending=False)[:15]
