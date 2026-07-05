import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Recommendation System")
st.write("Get Netflix movie and TV show recommendations using Content-Based Filtering.")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    features = [
        'director',
        'cast',
        'country',
        'rating',
        'listed_in',
        'description'
    ]

    for feature in features:
        df[feature] = df[feature].fillna('Unknown')

    df.drop_duplicates(subset='title', inplace=True)

    return df

df = load_data()

# -----------------------------
# Build Recommendation Model
# -----------------------------
@st.cache_resource
def build_model(data):

    data['combined_features'] = (
        data['listed_in'] + " " +
        data['director'] + " " +
        data['cast'] + " " +
        data['description']
    )

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(
        data['combined_features']
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    indices = pd.Series(
        data.index,
        index=data['title']
    ).drop_duplicates()

    return cosine_sim, indices

cosine_sim, indices = build_model(df)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(title, n=10):

    idx = indices[title]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n+1]

    movie_indices = [
        i[0]
        for i in sim_scores
    ]

    return df[
        ['title', 'type', 'listed_in', 'rating']
    ].iloc[movie_indices]

# -----------------------------
# User Interface
# -----------------------------
movie_list = sorted(df['title'].unique())

selected_movie = st.selectbox(
    "Select a Netflix Title",
    movie_list
)

num_recommendations = st.slider(
    "Number of Recommendations",
    1,
    20,
    10
)

if st.button("Recommend"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.subheader(
        f"Recommendations for '{selected_movie}'"
    )

    st.dataframe(
        recommendations,
        use_container_width=True
    )