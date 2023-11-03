# Import necessary libraries
import pandas as pd
from numpy import sqrt
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

def get_rating(userid,movieid):
    """ Get a user's rating for a movie.
    Args:
        userid (int): The user's ID needs to get the rating value.
        movieid (int): The movie's ID needs to get the rating value
    Returns:
        float: Rating value of 'userid' and 'movieid'.
    """
    return (ratings.loc[(ratings.userId==userid) & (ratings.movieId == movieid),'rating'].iloc[0])

def pearson_correlation_score(user1, user2):
    """ The similarity score function is based on Pearson correlation.
    Args:
        user1, user2 (int): The IDs of two users.
    Returns:
        float: Similarity score value.
    """
    both_watch_count = []
    list_movie_user1 = ratings.loc[ratings.userId == user1, 'movieId'].to_list()
    list_movie_user2 = ratings.loc[ratings.userId == user2, 'movieId'].to_list()
    for element in list_movie_user1:
        if element in list_movie_user2:
            both_watch_count.append(element)
    if(len(both_watch_count) == 0):
        return 0

    rating_sum_1 = sum([get_rating(user1, i) for i in both_watch_count])
    avg_rating_sum_1 = rating_sum_1/len(both_watch_count)
    rating_sum_2 = sum([get_rating(user2, i) for i in both_watch_count])
    avg_rating_sum_2 = rating_sum_2/len(both_watch_count)

    numerator = sum([(get_rating(user1, i) - avg_rating_sum_1)*(get_rating(user2, i) - avg_rating_sum_2)  for i in both_watch_count])
    denominator = sqrt(sum([pow((get_rating(user1, i) - avg_rating_sum_1),2) for i in both_watch_count]))*sqrt(sum([pow((get_rating(user2, i) - avg_rating_sum_2),2) for i in both_watch_count]))
    if(denominator == 0):
        return 0
    return numerator/denominator

def distance_similarity_score(user1,user2):
    """ The similarity score function is based on distance using a cosine measure.
    Args:
        user1, user2 (int): The IDs of two users.
    Returns:
        float: Similarity score value
    """
    both_watch_count = 0
    for element in ratings.loc[ratings.userId==user1,'movieId'].tolist():
        if element in ratings.loc[ratings.userId==user2,'movieId'].tolist():
            both_watch_count += 1
    if both_watch_count == 0 :
        return 0

    rating1 = []
    rating2 = []
    for element in ratings.loc[ratings.userId==user1,'movieId'].tolist():
        if element in ratings.loc[ratings.userId==user2,'movieId'].tolist():
            rating1.append(get_rating(user1,element))
            rating2.append(get_rating(user2,element))
    return dot(rating1, rating2)/(norm(rating1)*norm(rating2))

def most_similar_user(user1, number_of_user, similarity_name):
    """ Find the k neighbors with the highest similarity.
    Args:
        user1 (int): The user's ID.
        number_of_user (int): The number of users with the highest similarity to find.
        similarity_name (str): Name of similarity measure.
    Returns:
       list[tuple]: List of users with the highest similarity. The values in the list include tuples containing similar values and user IDs.
    """
    user_ID = ratings.userId.unique().tolist()
    print(len(user_ID))

    if (similarity_name == r"pearson"):
        similarity_score = [(pearson_correlation_score(user1, user_i),user_i)  for user_i in user_ID[:] if user_i != user1]
        # The list of users is large, so we can select a specific number of users (for example: 100 or 200).

    if(similarity_name == r"cosine"):
        similarity_score = [(distance_similarity_score(user1, user_i),user_i)  for user_i in user_ID[:] if user_i != user1]

    similarity_score.sort(reverse=True)
    return similarity_score[:number_of_user]

def weighted_rating_aggregation(userid, movieid, list_similar_user):
    """ Weighted rating aggregation function. Calculate user rating value based on similar users.
    Args:
        userid (int): The user's ID.
        movieid (int): The movie's ID.
        list_similar_user (list): list of users similar to 'userid'.
    Return:
        float: User rating value after weighted rating aggregation.
    """
    sims = []
    users = []
    user_watch_movie = []
    sim_user_watch_movie = []
    rating_user_watch_movie = []
    for sim, user in list_similar_user:
        sims.append(sim)
        users.append(user)

    for user in users:
        if user in ratings.loc[ratings.movieId==movieid,'userId'].tolist():
            user_watch_movie.append(user)
            sim_user_watch_movie.append([y[0] for x,y in enumerate(list_similar_user) if y[1] == user])
    for user in user_watch_movie:
        rating = get_rating(user, movieid)
        rating_user_watch_movie.append(rating)

    sim_user_watch_movie = np.array(sim_user_watch_movie).reshape(-1)
    rating_user_watch_movie = np.array(rating_user_watch_movie).reshape(-1)
    dot_product = np.dot(sim_user_watch_movie, rating_user_watch_movie)
    sum_sim = np.sum(sim_user_watch_movie)
    # Check if the sum of sim_user_watch_movie is not zero
    if sum_sim != 0:
        movieid_rating = dot_product / sum_sim
    else:
        movieid_rating = 0.0
    return movieid_rating

def update_rating(userid, unwatched_movies, list_similar_user):
    """ Update the rating value of movies that users have not watched yet.
    Args:
        userid (int): The user's ID.
        unwatched_movies (list): List of movies the user has not watched yet
        list_similar_user (list): list of users similar to 'userid'.
    Returns:
        dict: The newly updated dictionary of rating values is stored by movie name
    """
    predicted_movie_rating = {}
    for movie in unwatched_movies:
        movie_rating = weighted_rating_aggregation(userid, movie, list_similar_user)
        if math.isnan(movie_rating) == True:
            continue
        movie_name = movies.loc[movies.movieId == movie, 'title'].iloc[0]
        predicted_movie_rating[movie_name] = movie_rating
    return predicted_movie_rating

def main():
    UserID = int(input("Enter a user id to predict their next movie: "))
    k_neighbors = int(input("Enter the k neighbors needed for prediction: "))
    measure = str(input("Enter similarity measure: "))
    list_similar_user = most_similar_user(UserID, k_neighbors, measure)
    unwatched_movies = []
    list_movie_user = ratings.loc[ratings.userId == UserID, 'movieId'].tolist()
    list_movie = movies.loc[:,'movieId'].unique().tolist() # The list of movies is too large, so we can select a specific number of movies (for example: 1000).
    for movie in list_movie:
        if movie not in list_movie_user:
            unwatched_movies.append(movie)

    predicted_ratings = update_rating(UserID, unwatched_movies, list_similar_user)
    predicted_ratings_sort = sorted(predicted_ratings.items(), key=lambda x:x[1], reverse=True)
    converted_dict = dict(predicted_ratings_sort)
    df_update_rating = pd.DataFrame(converted_dict.items(), columns=['Title', 'Rating'])
    # Return the names and ratings of the top 10 movies that the user has not seen with the highest ratings.
    print(df_update_rating.head(10))
    

if __name__=="__main__": 
    main() 