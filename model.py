import os
import pandas as pd
import ast
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Function to simulate a dummy TMDB dataset if missing
def create_dummy_dataset():
    movies_data = {
        'id': [19995, 278, 155, 680, 24428],
        'title': ['Avatar', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction', 'The Avengers'],
        'overview': [
            "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission...",
            "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne...",
            "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent...",
            "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer...",
            "When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency..."
        ],
        'genres': [
            json.dumps([{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]), 
            json.dumps([{"id": 18, "name": "Drama"}, {"id": 80, "name": "Crime"}]),
            json.dumps([{"id": 18, "name": "Drama"}, {"id": 28, "name": "Action"}, {"id": 80, "name": "Crime"}, {"id": 53, "name": "Thriller"}]),
            json.dumps([{"id": 53, "name": "Thriller"}, {"id": 80, "name": "Crime"}]),
            json.dumps([{"id": 878, "name": "Science Fiction"}, {"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}])
        ],
        'keywords': [
            json.dumps([{"id": 1463, "name": "culture clash"}, {"id": 2964, "name": "future"}, {"id": 3386, "name": "space war"}]),
            json.dumps([{"id": 378, "name": "prison"}, {"id": 209, "name": "corruption"}, {"id": 417, "name": "police brutality"}]),
            json.dumps([{"id": 849, "name": "dc comics"}, {"id": 853, "name": "crime fighter"}, {"id": 549, "name": "secret identity"}, {"id": 10950, "name": "batman"}]),
            json.dumps([{"id": 396, "name": "transporter"}, {"id": 1415, "name": "brothel"}, {"id": 3684, "name": "magic"}]),
            json.dumps([{"id": 242, "name": "new york"}, {"id": 5539, "name": "shield"}, {"id": 9715, "name": "superhero"}])
        ]
    }
    credits_data = {
        'movie_id': [19995, 278, 155, 680, 24428],
        'title': ['Avatar', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction', 'The Avengers'],
        'cast': [
            json.dumps([{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0}, {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, "name": "Zoe Saldana", "order": 1}, {"cast_id": 25, "character": "Dr. Grace Augustine", "credit_id": "52fe48009251416c750aca39", "gender": 1, "id": 10205, "name": "Sigourney Weaver", "order": 2}]),
            json.dumps([{"cast_id": 3, "character": "Andy Dufresne", "credit_id": "52fe4231c3a36847f800b131", "gender": 2, "id": 504, "name": "Tim Robbins", "order": 0}, {"cast_id": 4, "character": "Ellis Boyd 'Red' Redding", "credit_id": "52fe4231c3a36847f800b135", "gender": 2, "id": 192, "name": "Morgan Freeman", "order": 1}, {"cast_id": 5, "character": "Warden Samuel Norton", "credit_id": "52fe4231c3a36847f800b139", "gender": 2, "id": 4029, "name": "Bob Gunton", "order": 2}]),
            json.dumps([{"cast_id": 35, "character": "Bruce Wayne / Batman", "credit_id": "52fe4220c3a36847f8005b07", "gender": 2, "id": 3894, "name": "Christian Bale", "order": 0}, {"cast_id": 3, "character": "Joker", "credit_id": "52fe4220c3a36847f8005a77", "gender": 2, "id": 1810, "name": "Heath Ledger", "order": 1}, {"cast_id": 16, "character": "Harvey Dent / Two-Face", "credit_id": "52fe4220c3a36847f8005a9d", "gender": 2, "id": 3903, "name": "Aaron Eckhart", "order": 2}]),
            json.dumps([{"cast_id": 2, "character": "Vincent Vega", "credit_id": "52fe4269c3a36847f801e269", "gender": 2, "id": 8891, "name": "John Travolta", "order": 0}, {"cast_id": 3, "character": "Jules Winnfield", "credit_id": "52fe4269c3a36847f801e26d", "gender": 2, "id": 2231, "name": "Samuel L. Jackson", "order": 1}, {"cast_id": 4, "character": "Mia Wallace", "credit_id": "52fe4269c3a36847f801e271", "gender": 1, "id": 139, "name": "Uma Thurman", "order": 2}]),
            json.dumps([{"cast_id": 46, "character": "Tony Stark / Iron Man", "credit_id": "52fe4495c3a368484e02b1ff", "gender": 2, "id": 3223, "name": "Robert Downey Jr.", "order": 0}, {"cast_id": 8, "character": "Steve Rogers / Captain America", "credit_id": "52fe4495c3a368484e02b1b3", "gender": 2, "id": 16828, "name": "Chris Evans", "order": 1}, {"cast_id": 36, "character": "Bruce Banner / The Hulk", "credit_id": "52fe4495c3a368484e02b1eb", "gender": 2, "id": 10313, "name": "Mark Ruffalo", "order": 2}])
        ],
        'crew': [
            json.dumps([{"credit_id": "52fe48009251416c750ac9c3", "department": "Directing", "gender": 2, "id": 2710, "job": "Director", "name": "James Cameron"}]),
            json.dumps([{"credit_id": "52fe4231c3a36847f800b153", "department": "Directing", "gender": 2, "id": 4027, "job": "Director", "name": "Frank Darabont"}]),
            json.dumps([{"credit_id": "52fe4220c3a36847f8005b63", "department": "Directing", "gender": 2, "id": 525, "job": "Director", "name": "Christopher Nolan"}]),
            json.dumps([{"credit_id": "52fe4269c3a36847f801e271", "department": "Directing", "gender": 2, "id": 138, "job": "Director", "name": "Quentin Tarantino"}]),
            json.dumps([{"credit_id": "52fe4495c3a368484e02b1ff", "department": "Directing", "gender": 2, "id": 12891, "job": "Director", "name": "Joss Whedon"}])
        ]
    }
    pd.DataFrame(movies_data).to_csv('tmdb_5000_movies.csv', index=False)
    pd.DataFrame(credits_data).to_csv('tmdb_5000_credits.csv', index=False)
    print("Created dummy tmdb_5000_movies.csv and tmdb_5000_credits.csv.")

def convert_to_list(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []
        
def convert_cast(obj):
    try:
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    except:
        return []

def fetch_director(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    except:
        return []

if __name__ == '__main__':
    # Generate dummy datasets if not present
    if not os.path.exists('tmdb_5000_movies.csv') or not os.path.exists('tmdb_5000_credits.csv'):
        create_dummy_dataset()
        
    print("Loading datasets...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    print("Merging datasets...")
    movies = movies.merge(credits, on='title')
    
    # Select columns
    if 'id' in movies.columns:
        movies.rename(columns={'id': 'movie_id'}, inplace=True)
    if 'movie_id_x' in movies.columns:
        movies.rename(columns={'movie_id_x': 'movie_id'}, inplace=True)
        
    cols_to_keep = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
    # Filter only overlapping columns
    cols_to_keep = [col for col in cols_to_keep if col in movies.columns]
    
    movies = movies[cols_to_keep]
    movies.dropna(inplace=True)
    
    print("Processing columns...")
    movies['genres'] = movies['genres'].apply(convert_to_list)
    movies['keywords'] = movies['keywords'].apply(convert_to_list)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Keep versions with spaces for displaying in the UI
    movies['display_cast'] = movies['cast'].apply(lambda x: ", ".join(x))
    movies['display_overview'] = movies['overview'].fillna('')
    
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if pd.notnull(x) else [])
    
    # Remove spaces from elements for tag making
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Create unified tags column
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    new_df = movies[['movie_id', 'title', 'tags', 'display_cast', 'display_overview']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    print("Training model...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    similarity = cosine_similarity(vectors)
    
    print("Saving pickles...")
    # Save a dictionary instead of dataframe for better cross-version compatibility
    pickle.dump(new_df.to_dict(), open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    
    print("Done! Model trained and saved as 'movies.pkl' and 'similarity.pkl'.")
