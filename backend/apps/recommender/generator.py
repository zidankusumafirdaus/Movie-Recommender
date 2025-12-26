import pandas as pd
import numpy as np
import logging
import json
import time
import os
import random
from apps.models.movie import Movies
from apps.models.rating import Ratings
from apps.models.recommendation import Recommendations
from apps.extensions import database
from apps.utils.cleaning import clean_dataframe_bytes, parse_genres
from apps.recommender.normalize import _normalize_series_0_1, _normalize_id_col
from apps.recommender.profile import _build_user_genre_profile_from_df, _movie_genre_vector

logger = logging.getLogger(__name__)


def generate_recommendations(alpha=None):
    logger.info("generate_recommendations: start")
    try:
        import tensorflow as tf
        import tensorflow_recommenders as tfrs
    except Exception:
        logger.exception("TensorFlow import failed")
        raise RuntimeError("TensorFlow and TFRS are required to run generate_recommendations")

    try:
        database.connect(reuse_if_open=True)
        ratings_rows = list(Ratings.select().dicts())
        movies_rows = list(Movies.select().dicts())
    finally:
        if not database.is_closed():
            database.close()

    ratings_df = pd.DataFrame(ratings_rows)
    movies_df = pd.DataFrame(movies_rows)

    ratings_df = clean_dataframe_bytes(ratings_df)
    movies_df = clean_dataframe_bytes(movies_df)
    logger.info("generate_recommendations: loaded data movies=%s ratings=%s", movies_df.shape, ratings_df.shape)

    if ratings_df.empty or movies_df.empty:
        return

    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['movie_id'] = ratings_df['movie_id'].astype(str)
    ratings_df['user_rating'] = pd.to_numeric(ratings_df['user_rating'], errors='coerce').fillna(0.0).astype(float)

    movies_df['movie_id'] = movies_df['movie_id'].astype(str)
    movies_df['movie_title'] = movies_df['movie_title'].astype(str)

    movies_df['genres_list'] = movies_df['movie_genres'].apply(parse_genres)
    movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))

    # compute normalized average rating per movie (used for reranking)
    try:
        movie_avg = ratings_df.groupby('movie_id')['user_rating'].mean()
        movie_avg_norm = _normalize_series_0_1(movie_avg)
        movie_avg_norm = movie_avg_norm.to_dict()
    except Exception:
        movie_avg_norm = {}

    data = ratings_df.merge(movies_df[['movie_id', 'genres_list']], on='movie_id', how='inner')
    if data.empty:
        logger.info("generate_recommendations: merged training data is empty, aborting")
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
    logger.info("generate_recommendations: training start (epochs=20)")
    model.fit(train.batch(512), validation_data=test.batch(512), epochs=20, verbose=0)
    logger.info("generate_recommendations: training finished")

    def recommend_movies(user_id, k=5, alpha_local=None):
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
        # if no rerank requested, return top-k by model prediction
        if alpha_local is None:
            top_k_idx = preds.argsort()[-k:][::-1]
            return [
                (movie_ids_local[i], movie_id_to_title.get(movie_ids_local[i], movie_ids_local[i]), float(preds[i]))
                for i in top_k_idx
            ]

        # compute user genre profile vector
        all_genres = sorted({g for sublist in movies_df['genres_list'] for g in sublist})
        user_profile = _build_user_genre_profile_from_df(user_id, ratings_df, movies_df)
        user_vec = np.array([user_profile.get(g, 0.0) for g in all_genres], dtype=float)
        # score candidates by blending normalized avg rating and genre-similarity
        final_scores = []
        for idx, mid in enumerate(movie_ids_local):
            try:
                avg = float(movie_avg_norm.get(str(mid), 0.0))
            except Exception:
                avg = 0.0
            mv = _movie_genre_vector(movies_df.loc[movies_df['movie_id'] == str(mid)].iloc[0].get('movie_genres'), all_genres)
            # cosine-like similarity between user_vec and mv
            denom = (np.linalg.norm(user_vec) * (np.linalg.norm(mv) if np.linalg.norm(mv) > 0 else 1.0))
            sim = float(np.dot(user_vec, mv) / denom) if denom > 0 else 0.0
            final = float(alpha_local) * avg + (1.0 - float(alpha_local)) * sim
            final_scores.append((mid, final, float(preds[idx])))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        top = final_scores[:k]
        return [(m, movie_id_to_title.get(m, m), s) for m, s, _ in top]

    all_recommendations = []
    unique_users = list(data['user_id'].unique())
    # helper: compose top-K according to quotas (personal/trending/explore)
    def compose_recommendations(uid, k=5, weights=(0.6, 0.2, 0.2), alpha_local=None):
        # weights -> (personal, trending, explore)
        p_w, t_w, e_w = weights
        # allocate counts (ensure sum k)
        n_personal = int(round(k * float(p_w)))
        n_trending = int(round(k * float(t_w)))
        n_explore = k - n_personal - n_trending

        seen = set(ratings_df[ratings_df['user_id'].astype(str) == str(uid)]['movie_id'].astype(str).unique())

        # personal candidates from model (use a bit more to allow filtering)
        personal_raw = recommend_movies(uid, k=max(5, n_personal * 3), alpha_local=None)
        personal = [str(mid) for mid, _, _ in personal_raw if str(mid) not in seen]

        # trending candidates from normalized avg rating (movie_avg_norm dict)
        try:
            # movie_avg_norm is a dict of movie_id -> normed score
            trending_sorted = sorted(movie_avg_norm.items(), key=lambda x: x[1], reverse=True)
            trending_ids = [str(m) for m, _ in trending_sorted]
        except Exception:
            trending_ids = list(movies_df['movie_id'].astype(str).values)
        trending = [mid for mid in trending_ids if mid not in seen]

        # exploration: sample from low-popularity tail
        try:
            pop_series = pd.Series(movie_avg_norm)
            if not pop_series.empty:
                thr = pop_series.quantile(0.5)
                tail_ids = pop_series[pop_series <= thr].index.astype(str).tolist()
            else:
                tail_ids = movies_df['movie_id'].astype(str).tolist()
        except Exception:
            tail_ids = movies_df['movie_id'].astype(str).tolist()
        # exclude seen and trending
        tail_candidates = [m for m in tail_ids if m not in seen and m not in trending]
        # random sample a few
        if tail_candidates:
            explore_sample = list(np.random.choice(tail_candidates, size=min(len(tail_candidates), max(5, n_explore * 5)), replace=False))
        else:
            explore_sample = []

        final = []
        # fill from personal, then trending, then explore, dedup and exclude seen
        pools = [(personal, 'personal'), (trending, 'trending'), (explore_sample, 'explore')]
        for pool, _ in pools:
            for mid in pool:
                if len(final) >= k:
                    break
                if mid in final:
                    continue
                if mid in seen:
                    continue
                final.append(mid)
            if len(final) >= k:
                break

        # fallback: if not enough, fill with model preds ignoring seen
        if len(final) < k:
            try:
                extra_raw = recommend_movies(uid, k=max(10, k * 3), alpha_local=None)
                for mid, title, score in extra_raw:
                    mid = str(mid)
                    if len(final) >= k:
                        break
                    if mid in final:
                        continue
                    final.append(mid)
            except Exception:
                pass

        # convert ids to tuples (id, title, score) using movie_id_to_title and placeholder score 0.0
        out = []
        for mid in final[:k]:
            title = movie_id_to_title.get(str(mid), str(mid))
            out.append((str(mid), title, float(movie_avg_norm.get(str(mid), 0.0))))
        return out
    # sample metadata for a few users when rerank is enabled
    rerank_metadata = {'alpha': alpha, 'timestamp': time.time(), 'samples': []}
    for uid in unique_users:
        top_movies = compose_recommendations(uid, k=5, weights=(0.6, 0.2, 0.2), alpha_local=alpha)
        for mid, title, score in top_movies:
            # lookup movie_genres from movies_df
            mg = None
            try:
                mg_row = movies_df.loc[movies_df['movie_id'] == mid]
                if not mg_row.empty and 'movie_genres' in mg_row.columns:
                    mg = str(mg_row.iloc[0]['movie_genres'])
            except Exception:
                mg = None
            all_recommendations.append({
                'user_id': uid,
                'movie_id': mid,
                'movie_title': title,
                'predicted_rating': score,
                'movie_genres': mg,
            })
        # record a small sample for metadata
        try:
            if alpha is not None and len(rerank_metadata['samples']) < 10:
                rerank_metadata['samples'].append({'user_id': uid, 'top_k': top_movies})
        except Exception:
            pass

    try:
        database.connect(reuse_if_open=True)
        Recommendations.delete().execute()
        if all_recommendations:
            for i in range(0, len(all_recommendations), 500):
                Recommendations.insert_many(all_recommendations[i:i+500]).execute()
    finally:
        if not database.is_closed():
            database.close()
    # write rerank metadata file when rerank was used
    try:
        if alpha is not None:
            out = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'rerank_metadata.json')
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(rerank_metadata, f, indent=2, default=str)
            logger.info('generate_recommendations: wrote rerank metadata to %s', out)
    except Exception:
        logger.exception('generate_recommendations: failed to write rerank metadata')
    logger.info("generate_recommendations: wrote %d recommendations", len(all_recommendations))


def update_all_recommendations():
    try:
        generate_recommendations()
    except Exception:
        logger.exception("update_all_recommendations failed")
        raise
