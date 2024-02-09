"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
reduced_ratings_data = pd.read_csv('resources/data/Collab_ratings.csv')

# We make use of an SVD model trained on a subset of the MovieLens 40k dataset.
model=pickle.load(open('resources/models/collab_model.pkl', 'rb'))

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(chosen_movies,top_n=5):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    movie_ids = reduced_ratings_data[reduced_ratings_data['title'].isin(chosen_movies)]['movieId'].unique()

    # Find similar users who rated the chosen movies highly
    similar_users = set()
    for movie_id in movie_ids:
        movie_ratings = reduced_ratings_data[reduced_ratings_data['movieId'] == movie_id]
        high_rated_users = set(movie_ratings[movie_ratings['rating'] >= 4]['userId'])
        similar_users.update(high_rated_users)

    # Recommend movies based on what similar users have liked
    recommended_movies = []
    for user_id in similar_users:
        # Get the movies rated by similar users
        user_movies = reduced_ratings_data[(reduced_ratings_data['userId'] == user_id) & (reduced_ratings_data['rating'] >= 4)]['movieId'].unique()
        
        # Exclude movies already chosen by the user
        user_movies = set(user_movies) - set(movie_ids)
        
        # Predict ratings for the movies
        predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in user_movies]
        
        # Sort predictions by predicted ratings in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Add the top 5 recommended movies to the list with titles
        recommended_movies.extend([(movie_id, reduced_ratings_data[reduced_ratings_data['movieId'] == movie_id]['title'].iloc[0], rating) for movie_id, rating in predictions[:5]])

    # Sort recommended movies by predicted ratings in descending order
    recommended_movies.sort(key=lambda x: x[2], reverse=True)

    # Get the top 5 recommended movie titles
    top_recommended_movie_titles = [title for _, title, _ in recommended_movies[:5]]

    return top_recommended_movie_titles
