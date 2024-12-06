import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import itertools
import time
import math

# System I: Recommendation Based on Popularity

movie_ratings = pd.read_csv("movie_ratings.csv" )

movie_ratings.head()

ratings = pd.read_csv('ratings.dat', sep='::', engine = 'python', header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
movies = pd.read_csv('movies.dat', sep='::', engine = 'python',
                     encoding="ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']
movies['MovieID'] = movies['MovieID'].astype(int)

movies.head()

# consider only movies that at least 10% of users have watched, to eliminate more obscure movies
non_nan_counts = movie_ratings.notna().sum()
filtered_columns = non_nan_counts[non_nan_counts > 604].index
movie_ratings = movie_ratings[filtered_columns]

movie_ratings.head()

# determine the average reviews of the remaining movies
column_sums = movie_ratings.sum()
non_nan_counts = movie_ratings.count()
average_ratings = column_sums / non_nan_counts

# now sort the movies in descending order of average rating
sorted_avg_ratings = average_ratings.sort_values(ascending=False)

# get the top 10 movies (movies with the highest average rating)
top_10_avg_rating = sorted_avg_ratings[:10]
top_10_ids = list(top_10_avg_rating.index)
top_10_ids = [int(movie_id[1:]) for movie_id in top_10_ids]

# Save the ranking of all movies for reuse
popularity_ranking = average_ratings.sort_values(ascending=False)
popularity_ranking.to_csv("popularity_ranking.csv", header=True)


# get the titles of these top 10 movies
filtered_df = movies[movies["MovieID"].isin(top_10_ids)]

print("Top 10 movies based on popularity:")
filtered_df

# for i in range(len(top_10_ids)):
#     print(f'#{i + 1}')
#     movie_id = top_10_ids[i]
#     img = mpimg.imread(f'MovieImages/{movie_id}.jpg')
#     movie_title = list(filtered_df[filtered_df['MovieID'] == movie_id]['Title'])[0]
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(movie_title)
#     plt.show()


# Adjust the top10 movie images in 2 lines horizontally
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap

# Assuming top_10_ids and filtered_df are already defined
rows = 2
cols = 5  # 10 movies total, 5 per row

fig, axes = plt.subplots(rows, cols, figsize=(20, 10))  # Create a 2x5 grid for the movies

for i, ax in enumerate(axes.flatten()):  # Flatten the 2D axes array for easy iteration
    movie_id = top_10_ids[i]
    img = mpimg.imread(f'MovieImages/{movie_id}.jpg')  # Load movie image
    movie_title = list(filtered_df[filtered_df['MovieID'] == movie_id]['Title'])[0]
    
    wrapped_title = f"#{i + 1} " + "\n".join(textwrap.wrap(movie_title, width=20))  # Adjust width as needed for line breaks
    
    ax.imshow(img, aspect='auto')  # Display image with aspect set to 'auto'
    ax.axis('off')  # Turn off axis
    ax.set_title(wrapped_title, fontsize=14, loc='center')  # Center-align the title with smaller font size
    
    # Adjust title position
    ax.title.set_position([0.5, -0.1])  # Title below the image; [0.5, -0.1] moves it to the center, slightly below
    
    # Set fixed aspect ratio and anchor to the left
    ax.set_aspect(1)  # Keep the axes square
    ax.set_anchor('W')  # Align images to the left

plt.tight_layout()  # Adjust layout for better spacing
# plt.show()
fig.savefig("top_movies.png")
plt.close(fig)


# System II: Recommendation Based on IBCF

movie_ratings = pd.read_csv("movie_ratings.csv")
movie_ratings.head()

# normalize each row
row_means = movie_ratings.mean(axis=1, skipna=True)
movie_ratings_normalized = movie_ratings.sub(row_means, axis=0)
movie_ratings_normalized = movie_ratings_normalized.to_numpy()

# %%time

# compute sums of pairwise multiplication of columns
num_columns = movie_ratings_normalized.shape[1]
pairwise_sums = np.zeros((num_columns, num_columns))

for i in range(num_columns):
    for j in range(i, num_columns):
        pairwise = movie_ratings_normalized[:, i] * movie_ratings_normalized[:, j]
        if np.sum(~np.isnan(pairwise)) < 3:
            pairwise_sums[i, j] = np.nan
            pairwise_sums[j, i] = np.nan
        else:
            pairwise_sums[i, j] = np.nansum(pairwise)
            pairwise_sums[j, i] = np.nansum(pairwise)

# %%time
# compute sums of squares of columns
num_columns = movie_ratings_normalized.shape[1]
i_squareds = np.zeros((num_columns, num_columns))
j_squareds = np.zeros((num_columns, num_columns))

for i in range(num_columns):
    for j in range(i, num_columns):
        i_column = movie_ratings_normalized[:, i]
        j_column = movie_ratings_normalized[:, j]
        
        non_nan_indices = np.where(~np.isnan(i_column) & ~np.isnan(j_column))[0]
        
        i_squareds[i, j] = np.sum(i_column[non_nan_indices]**2)
        i_squareds[j, i] = np.sum(i_column[non_nan_indices]**2)
        
        j_squareds[i, j] = np.sum(j_column[non_nan_indices]**2)
        j_squareds[j, i] = np.sum(j_column[non_nan_indices]**2)

similarity = 0.5 + (0.5 * (pairwise_sums / (np.sqrt(i_squareds) * np.sqrt(j_squareds))))

# set diagonal entries to NaN
np.fill_diagonal(similarity, np.nan)

# for each row in similarity, retain the top 30 values and set the rest to NaN
similarity_sorted = np.zeros((similarity.shape[0], similarity.shape[1]))
for i in range(similarity.shape[0]):
    row_i = np.copy(similarity[i, :])
    row_i = np.nan_to_num(row_i, nan=-np.inf)
    row_i_top_30_indices = np.argpartition(row_i, -30)[-30:]
    modified_array = np.full_like(row_i, np.nan, dtype=np.float64)
    modified_array[row_i_top_30_indices] = row_i[row_i_top_30_indices]
    similarity_sorted[i, :] = modified_array
similarity_sorted[np.isneginf(similarity_sorted)] = np.nan

# no_nans_similarity_sorted = np.where(np.isnan(similarity_sorted), '', similarity_sorted)
# no_nans_similarity_sorted = no_nans_similarity_sorted.astype(str)
# no_nans_similarity_sorted_df = pd.DataFrame(no_nans_similarity_sorted)
# no_nans_similarity_sorted_df.columns = list(movie_ratings.columns)
# no_nans_similarity_sorted_df.index = list(movie_ratings.columns)
# no_nans_similarity_sorted_df.to_csv("similarity_sorted.csv")

column_names = ['m1', 'm10', 'm100', 'm1510', 'm260', 'm3212']
column_indices = [movie_ratings.columns.get_loc(name) for name in column_names]
selected_values = similarity[column_indices, :][:, column_indices]
selected_values = np.round(selected_values, 7)
print(selected_values)

# implementing the IBCF function

movie_ratings = pd.read_csv("movie_ratings.csv")

def myIBCF(newuser):
    # keep track of which indices were already rated -- don't include these movies in the recommendations
    alreadyRatedIndices = set()
    for i in range(len(newuser)):
        if not np.isnan(newuser[i]):
            alreadyRatedIndices.add(i)
            
    validIndices = [[] for _ in range(similarity_sorted.shape[1])]
    
    # keep track of which indices to consider for each movie
    for i in range(similarity_sorted.shape[1]):
        movie_i = similarity_sorted[i, :]
        for j in range(similarity_sorted.shape[0]):
            if not np.isnan(movie_i[j]) and not np.isnan(newuser[j]):
                validIndices[i].append(j)

    finalResult = np.zeros(newuser.shape[0])
    
    for i in range(newuser.shape[0]):   
        # if np.isnan(newuser[i]):
        #     finalResult[i] = np.dot(similarity_sorted[i, :][validIndices[i]], newuser[validIndices[i]]) / np.sum(similarity_sorted[i, :][validIndices[i]])
        # else:
        #     finalResult[i] = newuser[i]
        
        # #  fix RuntimeWarning: invalid value encountered in scalar divide by 0 in denominator
        denominator = np.sum(similarity_sorted[i, :][validIndices[i]])
        if denominator > 0:
            finalResult[i] = np.dot(similarity_sorted[i, :][validIndices[i]], newuser[validIndices[i]]) / denominator
        else:
            finalResult[i] = 0  # Or some other default value

    
    # in finalResult, convert to NaN anything that was already rated
    for i in range(len(finalResult)):
        if i in alreadyRatedIndices:
            finalResult[i] = np.nan
    
    # sort in descending order
    sorted_indices = np.argsort(finalResult)[::-1]
    sorted_newuser_IBCF = finalResult[sorted_indices]
    
    # remove nans
    sorted_newuser_IBCF_no_nans = sorted_newuser_IBCF[~np.isnan(sorted_newuser_IBCF)]
    sorted_indices_no_nans = sorted_indices[-sorted_newuser_IBCF_no_nans.shape[0]:]
    
    print(f'Scores from IBCF: {sorted_newuser_IBCF_no_nans[:10]}')
    # print(f"Final scores before sorting: {finalResult}")
    # print(f"Top recommended indices: {sorted_indices[:10]}")

    movie_recs = movie_ratings.columns[sorted_indices_no_nans[:10]]
    movie_recs = list(movie_recs)
    
    # if < 10 recs, use System 1 recs as a backfill
    if len(movie_recs) < 10:
        toAdd = []
        for movie in top_10_ids:
            movie = 'm' + str(movie)
            movie_index = movie_ratings.columns.get_loc(movie)
            # check if the movie is already recommended, or if the user has already rated it
            if movie in movie_recs or movie_index in alreadyRatedIndices:
                continue
            movie_recs.append(movie)
            
    return movie_recs[0:10]

# %%time

user1181 = movie_ratings.loc["u1181"].values
user1181_recs = myIBCF(user1181)

user1181_recs

# call myIBCF on A hypothetical user who rates movie “m1613” with 5 and movie “m1755” with 4.

# %%time

hypothetical_user = np.full((movie_ratings.shape[1]), np.nan)
m1613_index = movie_ratings.columns.get_loc('m1613')
m1755_index = movie_ratings.columns.get_loc('m1755')
hypothetical_user[m1613_index] = 5
hypothetical_user[m1755_index] = 4

hypothetical_user_recs = myIBCF(hypothetical_user)

hypothetical_user_recs

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    hypothetical_user = np.full((movie_ratings.shape[1]), np.nan)
    for i, rating in new_user_ratings.items():
        index = movie_ratings.columns.get_loc('m' + str(i))
        hypothetical_user[i] = rating
    movie_ids = myIBCF(hypothetical_user)
    movie_ids_cleaned = [m_id[1:] for m_id in movie_ids]
    movie_ids_cleaned = list(map(int, movie_ids_cleaned))
    movie_recs = movies[movies['MovieID'].isin(movie_ids_cleaned)]
    return movie_recs



