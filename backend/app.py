from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import os
import sqlite3
import threading
import time
import warnings

from config import Config
from apps.utils.cleaning import clean_dataframe_bytes, normalize_cell, parse_genres
from apps.utils.sampledb import init_database, import_sample_data

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# === KONFIGURASI (dari `config.py`) ===
DB_PATH = Config.DB_PATH
MOVIES_CSV = Config.MOVIES_CSV
RATINGS_CSV = Config.RATINGS_CSV
UPDATE_INTERVAL = Config.UPDATE_INTERVAL

# NOTE: helper cleaning + DB import functions moved to `apps.utils.cleaning` and `apps.utils.db`

# ==========================================================
# === MODEL TRAINING & RECOMMENDATION (TFRS) ===
# ==========================================================
def generate_recommendations():
    print("\n🔁 [INFO] Memulai training model & update rekomendasi...\n")
    try:
        conn = sqlite3.connect(DB_PATH)
        ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
        movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
        conn.close()

        ratings_df = clean_dataframe_bytes(ratings_df)
        movies_df = clean_dataframe_bytes(movies_df)

        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
        ratings_df['movie_id'] = ratings_df['movie_id'].astype(str)
        ratings_df['user_rating'] = pd.to_numeric(ratings_df['user_rating'], errors='coerce').fillna(0.0).astype(float)

        movies_df['movie_id'] = movies_df['movie_id'].astype(str)
        movies_df['movie_title'] = movies_df['movie_title'].astype(str)

        def parse_genres(g):
            s = str(g).strip().strip("[]")
            if s == "":
                return ['0']
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
            else:
                parts = [p.strip() for p in s.split() if p.strip()]
            return parts if parts else ['0']

        movies_df['genres_list'] = movies_df['movie_genres'].apply(parse_genres)
        movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))

        data = ratings_df.merge(movies_df[['movie_id', 'genres_list']], on='movie_id', how='inner')
        if data.empty:
            print("⚠️ Tidak ada data ditemukan!")
            return

        tf.random.set_seed(42)
        user_ids = data['user_id'].values
        movie_ids = data['movie_id'].values
        ratings_vals = data['user_rating'].values
        genres_lists = data['genres_list'].values

        dataset = tf.data.Dataset.from_tensor_slices({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings_vals,
            'genres_list': tf.ragged.constant(genres_lists)
        })

        dataset = dataset.shuffle(buffer_size=len(data), seed=42, reshuffle_each_iteration=False)
        train_size = int(len(data) * 0.8)
        train = dataset.take(train_size)
        test = dataset.skip(train_size)

        embedding_dim = 32
        genre_embedding_dim = 16

        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=data['user_id'].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(data['user_id'].unique()) + 1, embedding_dim)
        ])

        movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=data['movie_id'].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(data['movie_id'].unique()) + 1, embedding_dim)
        ])

        all_genres = sorted({g for sublist in movies_df['genres_list'] for g in sublist})
        genre_lookup = tf.keras.layers.StringLookup(vocabulary=all_genres, mask_token=None)

        class NCFGenreEmbeddingModel(tfrs.Model):
            def __init__(self, user_model, movie_model, genre_lookup, genre_embedding_dim):
                super().__init__()
                self.user_model = user_model
                self.movie_model = movie_model
                self.genre_lookup = genre_lookup
                self.genre_embedding_layer = tf.keras.layers.Embedding(
                    len(genre_lookup.get_vocabulary()) + 1, genre_embedding_dim
                )
                self.mlp = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                self.rating_task = tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()]
                )

            def compute_loss(self, features, training=False):
                user_emb = self.user_model(features['user_id'])
                movie_emb = self.movie_model(features['movie_id'])
                genres = features['genres_list']
                genre_ids = self.genre_lookup(genres)
                genre_emb = tf.reduce_mean(self.genre_embedding_layer(genre_ids), axis=1)
                x = tf.concat([user_emb, movie_emb, genre_emb], axis=1)
                predictions = tf.squeeze(self.mlp(x), axis=1)
                return self.rating_task(labels=features['rating'], predictions=predictions)

            def predict_rating(self, features):
                user_emb = self.user_model(features['user_id'])
                movie_emb = self.movie_model(features['movie_id'])
                genres = features['genres_list']
                genre_ids = self.genre_lookup(genres)
                genre_emb = tf.reduce_mean(self.genre_embedding_layer(genre_ids), axis=1)
                x = tf.concat([user_emb, movie_emb, genre_emb], axis=1)
                return tf.squeeze(self.mlp(x), axis=1)

        model = NCFGenreEmbeddingModel(user_model, movie_model, genre_lookup, genre_embedding_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
        model.fit(train.batch(512), validation_data=test.batch(512), epochs=5, verbose=0)

        def recommend_movies(user_id, k=5):
            all_movies = movies_df.copy()
            all_movies['user_id'] = user_id
            ds_dict = {
                'user_id': tf.constant(all_movies['user_id'].values),
                'movie_id': tf.constant(all_movies['movie_id'].values),
                'genres_list': tf.ragged.constant(all_movies['genres_list'].values)
            }
            ds = tf.data.Dataset.from_tensor_slices(ds_dict).batch(1024)

            preds, movie_ids_local = [], []
            for batch in ds:
                pred = model.predict_rating(batch)
                preds.append(pred.numpy())
                movie_ids_local.extend(batch['movie_id'].numpy())

            preds = np.concatenate(preds)
            movie_ids_local = [m.decode() if isinstance(m, bytes) else str(m) for m in movie_ids_local]
            top_k_idx = preds.argsort()[-k:][::-1]
            return [(movie_id_to_title.get(movie_ids_local[i], movie_ids_local[i]), float(preds[i])) for i in top_k_idx]

        print("🎯 Menghasilkan rekomendasi untuk semua user...")
        all_recommendations = []
        unique_users = list(data['user_id'].unique())

        for i, uid in enumerate(unique_users, start=1):
            top_movies = recommend_movies(uid, k=5)
            for title, score in top_movies:
                all_recommendations.append({
                    'user_id': uid,
                    'movie_title': title,
                    'predicted_rating': score
                })
            # tampilkan progress tiap 50 user
            if i % 50 == 0 or i == len(unique_users):
                print(f"✅ {i}/{len(unique_users)} user selesai")

        pd.DataFrame(all_recommendations).to_sql('recommendations', sqlite3.connect(DB_PATH), if_exists='replace', index=False)
        print("✅ [SUCCESS] Tabel 'recommendations' diperbarui!\n")

    except Exception as e:
        print(f"❌ [ERROR] {e}")

# ==========================================================
# === HYBRID RECOMMENDATION SYSTEM ===
# ==========================================================
def content_based_recommendation(user_profile, k=5):
    conn = sqlite3.connect(DB_PATH)
    ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
    movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
    conn.close()

    ratings_df = clean_dataframe_bytes(ratings_df)
    movies_df = clean_dataframe_bytes(movies_df)

    if ratings_df.empty or movies_df.empty:
        return []

    gender = user_profile.get("user_gender")
    occupation = user_profile.get("user_occupation")
    age = user_profile.get("user_age")

    similar_users = ratings_df.copy()
    if gender:
        similar_users = similar_users[similar_users["user_gender"] == gender]
    if occupation:
        similar_users = similar_users[similar_users["user_occupation"] == occupation]
    if age:
        similar_users = similar_users[
            (similar_users["user_age"].astype(float) - float(age)).abs() <= 5
        ]

    if similar_users.empty:
        return []

    movie_scores = (
        similar_users.groupby("movie_id")["user_rating"]
        .mean()
        .reset_index()
        .sort_values("user_rating", ascending=False)
        .head(k)
    )

    merged = movie_scores.merge(movies_df, on="movie_id", how="left")
    return (
        merged[["movie_title", "user_rating"]]
        .rename(columns={"user_rating": "predicted_rating"})
        .to_dict(orient="records")
    )


def popular_recommendation(k=5):
    conn = sqlite3.connect(DB_PATH)
    ratings_df = pd.read_sql_query(
        "SELECT movie_id, AVG(user_rating) AS avg_rating FROM ratings GROUP BY movie_id ORDER BY avg_rating DESC LIMIT ?",
        conn, params=(k,))
    movies_df = pd.read_sql_query("SELECT movie_id, movie_title FROM movies", conn)
    conn.close()

    merged = ratings_df.merge(movies_df, on="movie_id", how="left")
    return (
        merged[["movie_title", "avg_rating"]]
        .rename(columns={"avg_rating": "predicted_rating"})
        .to_dict(orient="records")
    )

# ==========================================================
# === BACKGROUND UPDATE LOOP ===
# ==========================================================
def auto_update_loop():
    while True:
        generate_recommendations()
        print("🕒 Menunggu 1 jam untuk update berikutnya...\n")
        time.sleep(UPDATE_INTERVAL)

# ==========================================================
# === FLASK ENDPOINTS ===
# ==========================================================
def load_recommendations():
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM recommendations", conn)
        conn.close()
        if not df.empty:
            df = clean_dataframe_bytes(df)
            df['user_id'] = df['user_id'].astype(str)
            df['movie_title'] = df['movie_title'].astype(str)
            df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce').fillna(0.0)
        return df
    return pd.DataFrame(columns=['user_id', 'movie_title', 'predicted_rating'])


@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "🎬 Hybrid Movie Recommender API is running!"})


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = str(data.get('user_id'))
    k = int(data.get('k', 5))

    user_profile = {
        "user_age": data.get("user_age"),
        "user_gender": data.get("user_gender"),
        "user_occupation": data.get("user_occupation")
    }

    recommend_df = load_recommendations()

    # 1️⃣ Collaborative Filtering
    if not recommend_df.empty and user_id in recommend_df['user_id'].astype(str).values:
        user_recs = (
            recommend_df[recommend_df['user_id'].astype(str) == user_id]
            .nlargest(k, 'predicted_rating')
            [['movie_title', 'predicted_rating']]
            .to_dict(orient='records')
        )
        return jsonify({
            "user_id": user_id,
            "strategy": "collaborative_filtering",
            "recommendations": user_recs
        })

    # 2️⃣ Content-Based
    cbf_recs = content_based_recommendation(user_profile, k)
    if cbf_recs:
        return jsonify({
            "user_id": user_id,
            "strategy": "content_based",
            "recommendations": cbf_recs
        })

    # 3️⃣ Popular Fallback
    popular_recs = popular_recommendation(k)
    return jsonify({
        "user_id": user_id,
        "strategy": "popular",
        "recommendations": popular_recs
    })


@app.route('/recommend/<user_id>', methods=['GET'])
def recommend_by_id(user_id):
    """Endpoint lama dikembalikan: GET /recommend/<user_id>"""
    k = int(request.args.get('k', 5))
    recommend_df = load_recommendations()

    if not recommend_df.empty and user_id in recommend_df['user_id'].astype(str).values:
        user_recs = (
            recommend_df[recommend_df['user_id'].astype(str) == user_id]
            .nlargest(k, 'predicted_rating')
            [['movie_title', 'predicted_rating']]
            .to_dict(orient='records')
        )
        return jsonify({
            "user_id": user_id,
            "strategy": "collaborative_filtering",
            "recommendations": user_recs
        })
    else:
        # Jika user belum punya data → fallback popular
        return jsonify({
            "user_id": user_id,
            "strategy": "popular",
            "recommendations": popular_recommendation(k)
        })

# ==========================================================
# === MAIN ===
# ==========================================================
if __name__ == '__main__':
    init_database(DB_PATH)
    import_sample_data(DB_PATH, MOVIES_CSV, RATINGS_CSV)

    update_thread = threading.Thread(target=auto_update_loop, daemon=True)
    update_thread.start()

    app.run(
            debug=True, 
            use_reloader=False, 
            port=5000
            )