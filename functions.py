import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg
import heapq
import random

def plot_all(V, name):

    # Load V
    movies_2d, _, _ = np.linalg.svd(V, full_matrices=True)
    master_list = movies_2d[:, [0,1]]

    # Load data
    Y_data = np.loadtxt('data/data.txt').astype(int)
    ratings = Y_data[:, 2]
    movies = Y_data[:, 1]
    users = Y_data[:, 0]
    
    # Get ID of top 10 movies.
    movie_counts = np.bincount(movies)
    top_10_counts = heapq.nlargest(10, movie_counts)
    top_10_movies = []

    for i in range(len(top_10_counts)):
        movie_id = int(np.where(movie_counts == top_10_counts[i])[0])
        top_10_movies.append(movie_id)

    top_10_movies = np.asarray(top_10_movies)

    top_10_x, top_10_y = [], []
    for i in top_10_movies:
        top_10_x.append(master_list[i][0])
        top_10_y.append(master_list[i][1])

    # Get the master list of movie titles.
    movies_df = pd.read_csv('data/movies.txt', sep='\t', header=None, encoding='latin1')
    movies_df.columns = ["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", 
                         "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
                         "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movie_titles = movies_df["Movie Title"].tolist()

    # Plot
    plt.figure(figsize=(10, 10))
    plt.title("10 most rated movies")
    plt.scatter(top_10_x, top_10_y)
    for i, txt in enumerate(top_10_movies):
        plt.annotate(movie_titles[txt - 1], (top_10_x[i],top_10_y[i]))
    plt.savefig(name + ' top_10_movies.png') 

    # Get ID of best 10 movies
    def get_ratings(movies, ratings, movie_id):
        r = []
        for i in range(len(movies)):
            if (movies[i] == movie_id):
                r.append(ratings[i])
        return r

    movie_dic = {}
    for i in range(1682):
        movie_dic[i+1] = get_ratings(movies, ratings, i+1)
        
    average_ratings = np.zeros(1682)
    qualified_number = 100
    for i in range(len(average_ratings)):
        movie_i_ratings = movie_dic[i + 1]
        if (len(movie_i_ratings) >= qualified_number):
            average_ratings[i] = np.average(movie_i_ratings)
            
    best_10_ratings = heapq.nlargest(10, average_ratings)
    best_10_movies = []
    for i in range(len(best_10_ratings)):
        movie_id = int(np.where(average_ratings == best_10_ratings[i])[0]) + 1
        best_10_movies.append(movie_id)
    best_10_movies = np.asarray(best_10_movies)
    
    best_10_x, best_10_y = [], []
    for i in best_10_movies:
        best_10_x.append(master_list[i][0])
        best_10_y.append(master_list[i][1])

    # Plot 
    plt.figure(figsize=(10, 10))
    plt.title("10 highest rated movies")
    plt.scatter(best_10_x, best_10_y)
    for i, txt in enumerate(best_10_movies):
        plt.annotate(movie_titles[txt - 1], (best_10_x[i],best_10_y[i]))
    plt.savefig(name + ' best_10_movies.png')

    # 10 random movies
    random_10_movies = random.sample(range(1682), 10)
    random_10_x, random_10_y = [], []
    for i in random_10_movies:
        random_10_x.append(master_list[i][0])
        random_10_y.append(master_list[i][1])

    # Plot
    plt.figure(figsize=(10, 10))
    plt.title("10 random movies")
    plt.scatter(random_10_x, random_10_y)
    for i, txt in enumerate(random_10_movies):
        plt.annotate(movie_titles[txt - 1], (random_10_x[i],random_10_y[i]))
    plt.savefig(name + ' random_10_movies.png')
   
    # Genres
    genre_ids = [movies_df[movies_df['Comedy'] == 1]["Movie Id"].tolist(), 
               movies_df[movies_df['Romance'] == 1]["Movie Id"].tolist(),
           movies_df[movies_df['Action'] == 1]["Movie Id"].tolist()]

    genre = ['Comedy', 'Romance', 'Action']
    for k in range(3):
        random_10_movies = random.sample(genre_ids[k], 10)
        random_10_x, random_10_y = [], []
        for i in random_10_movies:
            random_10_x.append(master_list[i][0])
            random_10_y.append(master_list[i][1])

        plt.figure(figsize=(10, 10))
        plt.title(genre[k])
        plt.scatter(random_10_x, random_10_y)
        for i, txt in enumerate(random_10_movies):
            plt.annotate(movie_titles[txt - 1], (random_10_x[i],random_10_y[i]))
        plt.savefig(name + " 10 random " + genre[k] + " movies.png")