import numpy as np

# Sample movie ratings data
user_movie_ratings = {
    'User1': {'Movie1': 4, 'Movie2': 5, 'Movie3': 0, 'Movie4': 0},
    'User2': {'Movie1': 0, 'Movie2': 4, 'Movie3': 3, 'Movie4': 0},
    'User3': {'Movie1': 2, 'Movie2': 0, 'Movie3': 4, 'Movie4': 5},
    'User4': {'Movie1': 0, 'Movie2': 3, 'Movie3': 0, 'Movie4': 4},
    'User5': {'Movie1': 5, 'Movie2': 0, 'Movie3': 2, 'Movie4': 0},
    'User6': {'Movie1': 3, 'Movie2': 0, 'Movie3': 2, 'Movie4': 0},
    'User7': {'Movie1': 2, 'Movie2': 1, 'Movie3': 3, 'Movie4': 0},
    'User8': {'Movie1': 1, 'Movie2': 0, 'Movie3': 4, 'Movie4': 0},
    'User9': {'Movie1': 0, 'Movie2': 3, 'Movie3': 1, 'Movie4': 2}
}

# Convert ratings data to a matrix
def create_ratings_matrix(data):
    users = list(data.keys())
    movies = list(set(movie for ratings in data.values() for movie in ratings))
    
    matrix = np.zeros((len(users), len(movies)))
    
    for m, user in enumerate(users):
        for n, movie in enumerate(movies):
            matrix[m, n] = data[user].get(movie, 0)
    return matrix

ratings_matrix = create_ratings_matrix(user_movie_ratings)

# Calculate similarity between users using cosine similarity
user_similarity_matrix = np.dot(ratings_matrix, ratings_matrix.T)
user_norms = np.linalg.norm(ratings_matrix, axis=1, keepdims=True)
user_similarity_matrix /= np.dot(user_norms, user_norms.T)

# Recommend movies for a given user
def recommend_movies(user, data, similarity_matrix, n=2):
    user_index = list(data.keys()).index(user)
    user_ratings = data[user]
    
    unrated_movies = [movie for movie, rating in user_ratings.items() if rating == 0]
    
    # Calculate predicted ratings for unrated movies
    predicted_ratings = np.dot(similarity_matrix[user_index], ratings_matrix) / np.sum(np.abs(similarity_matrix[user_index]))
    
    # Sort unrated movies by predicted ratings in descending order
    recommended_movies_indices = np.argsort(predicted_ratings)[::-1]
    
    recommended_movies = [list(user_ratings.keys())[i] for i in recommended_movies_indices if list(user_ratings.keys())[i] in unrated_movies]
    
    return recommended_movies[:n]

# Get movie recommendations for a specific user
user_to_recommend = 'User7'
movie_recommendations = recommend_movies(user_to_recommend, user_movie_ratings, user_similarity_matrix)
print(f"Recommended movies for {user_to_recommend}: {movie_recommendations}")
