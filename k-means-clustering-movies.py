"""k-means clustering of movie ratings

Say you're a data analyst at Netflix and you want to explore the similarities and differences in people's
tastes in movies based on how they rate different movies. Can understanding these ratings contribute to a
movie recommendation system for users? Let's dig into the data and see.
"""

# Dataset Overview
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

# Import the movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
print(movies.head())

# Import the ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')

print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')

# Let's start by taking a subset of users, and seeing what their prefered genres are.
# The function get_genre_ratings calculated each user's average rating of all romance
# movies and all scifi movies.

genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'],
                                         ['avg_romance_rating', 'avg_scifi_rating'])

# Let's bias our dataset a little by removing people who like both scifi and romance,
# just so that our clusters tend to define them as liking one genre more than the other.
biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

print("So we can see we have {} users, "
      "and for each user we have their average\n"
      "rating of the romance and sci movies they have watched.".format(len(biased_dataset)))


# %matplotlib inline
# helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],
#                         'Avg scifi rating', biased_dataset['avg_romance_rating'],
#                         'Avg romance rating')

# Lets apply K-Means on above set
X = biased_dataset[['avg_scifi_rating', 'avg_romance_rating']].values

# Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=5, random_state=0)

# use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X)

# Plot
# helper.draw_clusters(biased_dataset, predictions)

# To decide exact value of k we will use "elbow method"
# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)

# Calculate error values for all k values we're interested in
errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

# Optional: Look at the values of K vs the silhouette score of running K-means with that value of k
print(list(zip(possible_k_values, errors_per_k)))

# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlabel('K - number of clusters')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')
plt.show(block=True)

# for value k = 7 is the best clustering.
# Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7, random_state=2)

# use fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# plot
helper.draw_clusters(biased_dataset, predictions_4, cmap='Accent')