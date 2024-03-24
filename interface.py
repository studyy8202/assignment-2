import os
import pandas as pd
import numpy as np
import streamlit as st



new_directory = '/Users/simonnguyen/Downloads/ml'
os.chdir(new_directory)

# Print new working directory
print("New working directory:", os.getcwd())

npo_df = pd.read_csv('npo-2.csv')


# Streamlit layout
st.title("Shows")

# Display horizontal list of movie posters with titles
st.subheader("Your shows")
show_all = st.button("Show All")

# Display only 10 movies initially
num_movies_to_display = min(len(npo_df), 10)

# Set to keep track of unique series names
unique_series = npo_df["Serie"].unique()

for series in unique_series:
    series_data = npo_df[npo_df["Serie"] == series].head(1)
    cols = st.columns(2)
    for index, row in series_data.iterrows():
        with cols[0]:
            st.header(row["Serie"])
            st.image(row["Image"], width=300)
        with cols[1] :
            st.header(row["Broadcaster"])
            st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")
        with st.expander("Episodes"):
                    episodes = npo_df[npo_df["Serie"] == series]
                    for _, episode_row in episodes.iterrows():
                        st.subheader(f"Episode {episode_row['Episode']}")
                        st.text(f"Title: {episode_row['Title']}")
                        st.write(f"Description: {episode_row['Description']}", height=100)


# # Display only 10 movies initially
# num_movies_to_display = min(len(npo_df), 10)
# for movie in npo_df[:num_movies_to_display]:
#     col = st.columns(2)
#     with col[0]:
#         st.image(npo_df["Image"], use_column_width=True)
#     with col[1]:
#         st.write(npo_df["Serie"])

# # Show all movies if the button is clicked
# if show_all:
#     st.subheader("All Movies")
#     for movie in npo_df:
#         st.image(npo_df["Image"], use_column_width=True)
#         st.write(npo_df["Serie"])

# # Function to display episodes when clicking on a movie poster
# def display_episodes(movie):
#     st.subheader(f"Episodes of {npo_df['Serie']}")
#     for episode in npo_df["Episode"]:
#         st.write(f"**{npo_df['Title']}**: {npo_df['Description']}")

# # Handle click events on movie posters
# clicked_movie_index = st.expander("Click on poster to see episodes").beta_container()
# for i, movie in enumerate(npo_df):
#     with clicked_movie_index:
#         if st.image(npo_df["Image"], use_column_width=True, caption=npo_df["Serie"]):
#             display_episodes(movie)