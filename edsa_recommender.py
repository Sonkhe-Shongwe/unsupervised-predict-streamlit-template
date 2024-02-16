"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
df_Content = load_movie_titles('resources/data/df_Content.csv')
collab_data = load_movie_titles('resources/data/Collab_ratings.csv')


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","About", "Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            # User-based preferences
            st.write('### Enter Your Three Favorite Movies')
            movie_1 = st.selectbox('First Option', df_Content)
            movie_2 = st.selectbox('Second Option', df_Content)
            movie_3 = st.selectbox('Third Option', df_Content)
            fav_movies = [movie_1, movie_2, movie_3]
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=5)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except:
                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            # User-based preferences
            # Default movie values
            default_movies = ['Superman', 'Toy Story', 'Terminator']

            st.write('### Enter Your Three Favorite Movies')
            # Select boxes with default values
            movie_1 = st.selectbox('First Option', collab_data, index=collab_data.index(default_movies[0]))
            movie_2 = st.selectbox('Second Option', collab_data, index=collab_data.index(default_movies[1]))
            movie_3 = st.selectbox('Third Option', collab_data, index=collab_data.index(default_movies[2]))

            fav_movies = [movie_1, movie_2, movie_3]
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(chosen_movies=fav_movies,
                                                           top_n=5)
                    if top_recommendations:
                        st.title("We think you'll like:")
                    else:
                        st.title("Unfortunately there is not enough data on the selected movies to return recommendations")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except Exception as e:
                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")
                    print(e)

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "About":
        st.title("About")
        st.write("This is a Streamlit web application for a Movie Recommender Engine. It recommends movies based on either content-based or collaborative-based filtering algorithms. The content-based filtering algorithm recommends movies similar to the user's preferred movies, while the collaborative-based filtering algorithm recommends movies based on the preferences of users with similar tastes. This application is part of the Explore Data Science Academy Unsupervised Predict project.")
        
        
        st.header("Meet the Team")
        st.write("- **Ntsako Rivisi**")
        st.write("- **Carol Ndlovu**")
        st.write("- **Millicent Tsweleng**")
        st.write("- **Sinethemba Sibiya**")
        st.write("- **Aphiwe Maphumulo**")
        st.write("- **Sonkhe Shongwe**")
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown('<style>img {max-width: 100%; height: 400px;}</style>', unsafe_allow_html=True)
        st.image('resources/imgs/Content.webp', use_column_width=True)

        st.write("## Content-Based Filtering")
        st.write(" ##### **Our content-based filtering algorithm assists you in exploring movies based on the distinctive characteristics of films you already admire. Here's an in-depth explanation of its functioning:**")
        st.write(" #### **Step 1**: Identification of Key Features")
        st.write("In our algorithm, each movie in the database is associated with an Overview. This overview, meticulously curated by our team of data scientists, encapsulates vital information about the movie, including its *Genres*, *Cast*, *Directors*, and *Keywords*, offering a succinct depiction of the plot. This information serves as the foundation for our algorithm to determine the most suitable movie recommendations based on your top 3 choices.")
        st.write(" #### **Step 2**: Transformation of Movie Overviews into Vectors")
        st.write("We employ a sophisticated technique known as TF-IDF (*Term Frequency-Inverse Document Frequency*) to convert the movie overviews into numerical vectors. This transformation enables us to represent the content of each movie in a format comprehensible to our algorithm.")
        st.write(" #### **Step 3**: Computation of Similarity")
        st.write("Our algorithm utilizes cosine similarity to gauge the likeness between movies, primarily based on their Overviews. This calculation enables us to rank movies according to their resemblance to your top 3 choices.")
        st.write(" #### **Step 4**: User Input and Recommendation Generation")
        st.write("You provide us with your three favorite movies, and we leverage this information to identify movies with similar content. The top five recommendations are determined by the movies with the highest similarity scores to your top 3 choices.")
        st.write(" #### **Step 5**: Presentation of Recommendations")
        st.write("Following data analysis, we present you with a curated list of top 5 recommendations, derived from content similarities.")
        st.write(" #### **Note**: ")
        st.write("- The algorithm delves into various aspects of movies, including titles, directors, genres, and plot keywords, to gain insights into your preferences.")
        st.write("- In cases where there is insufficient data on your selected movies, we provide appropriate notification.")

        st.write("## Collaborative-Based Filtering")
        st.markdown('<style>img {max-width: 100%; height: 400px;}</style>', unsafe_allow_html=True)
        st.image('resources/imgs/Collaborative.webp', use_column_width=True)

        st.write("##### **Our collaborative-based filtering algorithm harnesses the collective preferences of users with analogous tastes to deliver tailored movie recommendations. Unlike content-based models, which focus on specific movie attributes, our collaborative filtering approach prioritizes user interactions to generate personalized suggestions. Here's an exhaustive overview of its functionality:**")
        st.write(" #### **Step 1**: Data Filtering")
        st.write("We meticulously filter users who have rated a substantial number of movies, ensuring the inclusion of highly engaged individuals. Additionally, we filter movies with a significant number of ratings, enhancing the reliability of our recommendations.")
        st.write(" #### **Step 2**: Identifying Similar Users")
        st.write("For your selected favorite movies, we identify users who have consistently rated them highly, signifying similar tastes and preferences. This forms the foundation for our collaborative filtering mechanism.")
        st.write(" #### **Step 3**: Recommendation Generation")
        st.write("Drawing insights from user interactions, our model employs Singular Value Decomposition (SVD) to discern latent patterns in user preferences. By predicting ratings for movies endorsed by similar users, we curate a list of top recommendations, prioritizing those with the highest predicted ratings.")
        st.write(" #### **Step 4**: Presentation of Recommendations")
        st.write("We meticulously analyze the data to furnish you with a refined selection of top 5 recommendations, meticulously tailored to your preferences.")
        st.write(" #### **Note**: ")
        st.write("- Our collaborative model is trained using the Singular Value Decomposition (SVD) technique, enabling accurate predictions for personalized movie suggestions.")
        st.write("- Recommendations are derived from the ratings of users with analogous tastes, ensuring the endorsement of movies highly rated by individuals sharing your preferences.")
        st.write("- Unlike content-based approaches, which focus on movie attributes, our collaborative filtering algorithm prioritizes user interactions.")
        st.write("- In instances where data on your selected movies is insufficient, we provide appropriate notification.")

        
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
