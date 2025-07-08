import streamlit as st
import requests
import pandas as pd
import joblib
import io

# --- App Config ---
st.set_page_config(page_title="LENS Movie Recommender", layout="wide")

# --- Header ---
st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72, #000000); padding: 25px; border-radius: 15px; text-align: center;'>
        <h1 style='color: white; font-size: 48px;'>🎬 Movie Recommendation System</h1>
        <h3 style='color: #e0f7fa;'>Get top movie recommendations using content-based filtering</h3>
        <p style='color: #ffffff; font-size: 18px;'>Created by <b style='color:#FFD700;'>Nicky Kumari</b> | 📧 07nk05@gmail.com | 📍 Noida | 📞 76545678757</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    # st.image("riteshsamridhipics.jpg", width=200)
    st.markdown("""
        <h3 style='color:#FF6347;'>🎥 Navigation</h3>
    """, unsafe_allow_html=True)
    selected_section = st.radio("Choose Section", ["🎬 Movie Recommendation"])

# --- Load Data ---
pkg = joblib.load("movie_recommendation.pkl")
model = pkg["model"]
vector = pkg["vector"]
df5 = pkg["df5"]

# --- Recommendation Section ---
if selected_section == "🎬 Movie Recommendation":
    st.markdown("""
        <h2 style='color:#2a5298; font-size: 28px; margin-top: 10px;'>🎬 Movie Recommendation</h2>
        <p style='color:gray; font-size:16px;'>Select a movie below to receive similar recommendations based on genre, cast, and director. Visual cards and poster previews help you explore top options visually.</p>
    """, unsafe_allow_html=True)

    df5['normalized_name'] = df5['name'].str.strip().str.lower()
    #movie selectbox
    selected_movie = st.selectbox(
    "🎬 Select a movie:", 
    df5['name'].sort_values().unique(), 
    index=None, 
    placeholder="Choose a movie",
    key="movie_selector"
)

# --- Show number of recommendations only after movie selected ---
if selected_movie:
    index = df5[df5['name'] == selected_movie].index[0]

    num_recommend = st.selectbox(
    "🔢 Number of recommendations:",
    [1,2,3,4,5,6,7,8,9,10],
    index=None,
    placeholder="Choose number of recommendations",
    key="recommendation_selector"
)
    

    # ✅ Now do recommendations ONLY after both are selected
    if num_recommend:
        st.subheader(f"📽️ Top ({num_recommend}) Recommended Movies ")

        vectors = vectorizer.transform(df5['tag'])
        distances, indexes = model.kneighbors(vectors[index], n_neighbors=num_recommend + 1)

        col1, col2, col3 = st.columns([1.1, 1, 1.1])
        col_index = 0
        rec_titles, rec_ids = [], []

        for i in indexes[0][1:]:
            title = df5.loc[i]['name']
            movie_id = df5.loc[i]['movie_id']
            genre = df5.loc[i].get('genre', 'N/A')
            director = df5.loc[i].get('director', 'N/A')
            cast = df5.loc[i].get('cast', 'N/A')

            url = f"http://www.omdbapi.com/?i={movie_id}&apikey=1d2456cd"
            try:
                data = requests.get(url).json()
                poster = data.get("Poster", "")
                rating = data.get("imdbRating", "N/A")
                year = data.get("Year", "N/A")
                director = data.get("Director", director)
                cast = data.get("Actors", cast)
                genre = data.get("Genre", genre)
            except:
                poster, rating, year = "", "N/A", "N/A"

            column = [col1, col2, col3][col_index % 3]
            card = f"""
                <div style='background:#f9f9f9; padding:15px; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.12); text-align:center; margin-bottom:22px; max-width:240px; margin-left:auto; margin-right:auto;'>
                    <img src='{poster}' width='210px' height='310px' style='border-radius:8px;' /><br><br>
                    <b style='font-size:16px;'>{title}</b><br>
                    ⭐ {rating} | 📅 {year}<br>
                    🎬 {director}<br>
                    🎭 {cast}<br>
                    📚 {genre}
                </div>
            """
            column.markdown(card, unsafe_allow_html=True)
            col_index += 1
            rec_titles.append(title)
            rec_ids.append(movie_id)

        if rec_titles:
            result_df = pd.DataFrame({"Recommended Movie": rec_titles, "IMDB ID": rec_ids})
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=csv_buffer.getvalue(),
                file_name="recommended_movies.csv",
                mime="text/csv"
            )
