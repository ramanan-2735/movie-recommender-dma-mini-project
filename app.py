import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load models and error handling
def load_data():
    try:
        movies_dict = pickle.load(open('movies.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    if movies_df is None or similarity_matrix is None:
        return []
    
    movie = movie.strip()
    
    # Search for movie case-insensitively
    match = movies_df[movies_df['title'].str.lower() == movie.lower()]
    if match.empty:
        return []
        
    movie_index = match.index[0]
    
    distances = similarity_matrix[movie_index]
    # Sort distances and get top 5 recommendations (skip the 1st standard self-match)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        row = movies_df.iloc[i[0]]
        desc = str(row.get('display_overview', ''))
        if len(desc) > 100:
            desc = desc[:100] + "..."
            
        recommended_movies.append({
            'title': row.title,
            'cast': row.get('display_cast', ''),
            'overview': desc
        })
    
    # Fallback to fewer titles if dataset is very small
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        if not movie_name or not str(movie_name).strip():
            return render_template('index.html', error="Please enter a valid movie name.")
            
        movies, similarity = load_data()
        if movies is None:
            return render_template('index.html', error="Model data not found. Please run 'python model.py' first.", movie_name=movie_name)
            
        recommendations = recommend(movie_name, movies, similarity)
        
        if not recommendations:
            return render_template('index.html', error=f"Movie '{movie_name}' not found in our database. Try 'Avatar' or 'Batman'.", movie_name=movie_name)
            
        return render_template('index.html', movie_name=movie_name, recommendations=recommendations)
        
    return render_template('index.html')

@app.route('/movies', methods=['GET'])
def get_movies():
    movies, _ = load_data()
    if movies is not None:
        movie_list = movies['title'].tolist()
        return jsonify(movie_list)
    return jsonify([])

if __name__ == '__main__':
    # Ensure templates folder exists for development
    app.run(debug=True)
