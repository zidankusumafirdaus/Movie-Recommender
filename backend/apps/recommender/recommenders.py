import logging
import pandas as pd
from peewee import fn
from apps.models.movie import Movies
from apps.models.rating import Ratings
from apps.models.recommendation import Recommendations
from apps.extensions import database
from apps.utils.cleaning import clean_dataframe_bytes
from apps.recommender.normalize import _normalize_id_col

logger = logging.getLogger(__name__)


def content_based_recommendation(user_profile, k=5):
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
    ratings_df = _normalize_id_col(ratings_df, 'user_id')
    ratings_df = _normalize_id_col(ratings_df, 'movie_id')
    movies_df = _normalize_id_col(movies_df, 'movie_id')
    # normalize id columns for safe joins
    ratings_df = _normalize_id_col(ratings_df, 'movie_id')
    ratings_df = _normalize_id_col(ratings_df, 'user_id')
    movies_df = _normalize_id_col(movies_df, 'movie_id')

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
    try:
        database.connect(reuse_if_open=True)
        q = (Ratings
             .select(Ratings.movie_id, fn.AVG(Ratings.user_rating).alias('avg_rating'))
             .group_by(Ratings.movie_id)
             .order_by(fn.AVG(Ratings.user_rating).desc())
             .limit(k))
        ratings_rows = list(q.dicts())
        movies_rows = list(Movies.select(Movies.movie_id, Movies.movie_title, Movies.movie_genres).dicts())
    finally:
        if not database.is_closed():
            database.close()
    ratings_df = pd.DataFrame(ratings_rows)
    movies_df = pd.DataFrame(movies_rows)
    ratings_df = clean_dataframe_bytes(ratings_df)
    movies_df = clean_dataframe_bytes(movies_df)
    ratings_df = _normalize_id_col(ratings_df, 'movie_id')
    movies_df = _normalize_id_col(movies_df, 'movie_id')

    merged = ratings_df.merge(movies_df, on="movie_id", how="left")
    # include movie_genres in popular response
    cols = ["movie_id", "movie_title", "avg_rating"]
    if 'movie_genres' in merged.columns:
        cols.append('movie_genres')
    out = merged[cols].rename(columns={"avg_rating": "predicted_rating"})
    return out.to_dict(orient="records")


def populate_popular_recommendations(k=5):
    logger.info("populate_popular_recommendations: start (k=%d)", k)
    try:
        database.connect(reuse_if_open=True)
        ratings_rows = list(Ratings.select(Ratings.user_id, Ratings.movie_id, Ratings.user_rating).dicts())
        movies_rows = list(Movies.select(Movies.movie_id, Movies.movie_title, Movies.movie_genres).dicts())
    finally:
        if not database.is_closed():
            database.close()
    ratings_df = clean_dataframe_bytes(pd.DataFrame(ratings_rows))
    movies_df = clean_dataframe_bytes(pd.DataFrame(movies_rows))
    ratings_df = _normalize_id_col(ratings_df, 'user_id')
    ratings_df = _normalize_id_col(ratings_df, 'movie_id')
    movies_df = _normalize_id_col(movies_df, 'movie_id')

    if ratings_df.empty or movies_df.empty:
        logger.info("populate_popular_recommendations: ratings or movies empty, aborting")
        return 0

    avg_scores = (
        ratings_df.groupby('movie_id')['user_rating']
        .apply(lambda s: pd.to_numeric(s, errors='coerce').astype(float).mean())
        .reset_index(name='avg_rating')
        .sort_values('avg_rating', ascending=False)
        .head(k)
    )

    merged = avg_scores.merge(movies_df, on='movie_id', how='left')
    top_movies = merged[['movie_id', 'movie_title', 'avg_rating', 'movie_genres']].dropna()
    if top_movies.empty:
        logger.info("populate_popular_recommendations: no top movies found")
        return 0

    # prepare top movies for insertion (rename avg_rating -> predicted_rating)
    top_movies_df = top_movies.copy()
    top_movies_df = top_movies_df.rename(columns={'avg_rating': 'predicted_rating'})

    # determine which users already have recommendations so we don't overwrite them
    try:
        database.connect(reuse_if_open=True)
        existing_user_rows = list(Recommendations.select(Recommendations.user_id).distinct().dicts())
    finally:
        if not database.is_closed():
            database.close()

    existing_users = set(str(r['user_id']) for r in existing_user_rows) if existing_user_rows else set()

    all_users = [str(u) for u in ratings_df['user_id'].unique()]
    users_to_fill = [u for u in all_users if u not in existing_users]
    if not users_to_fill:
        logger.info("populate_popular_recommendations: all users already have recommendations, skipping")
        return 0

    users_df = pd.DataFrame({'user_id': users_to_fill})
    users_df['key'] = 1
    top_movies_df['key'] = 1
    cart = users_df.merge(top_movies_df, on='key').drop(columns=['key'])

    # include movie_genres in recs if present
    if 'movie_genres' in cart.columns:
        recs = cart[['user_id', 'movie_id', 'movie_title', 'predicted_rating', 'movie_genres']]
    else:
        recs = cart[['user_id', 'movie_id', 'movie_title', 'predicted_rating']]

    recs_records = recs.to_dict(orient='records')
    try:
        database.connect(reuse_if_open=True)
        if recs_records:
            for i in range(0, len(recs_records), 500):
                Recommendations.insert_many(recs_records[i:i+500]).execute()
    finally:
        if not database.is_closed():
            database.close()

    logger.info("populate_popular_recommendations: wrote %d recommendations (users_filled=%d, k=%d)", len(recs), len(users_to_fill), k)
    return len(recs)
