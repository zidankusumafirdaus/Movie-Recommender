import pandas as pd
from apps.models.recommendation import Recommendations
from apps.extensions import database
from apps.utils.cleaning import clean_dataframe_bytes
from apps.recommender.normalize import _normalize_id_col


def load_recommendations():
    try:
        database.connect(reuse_if_open=True)
        rows = list(Recommendations.select(Recommendations.user_id, Recommendations.movie_id, Recommendations.movie_title, Recommendations.predicted_rating, Recommendations.movie_genres).dicts())
    except Exception:
        return pd.DataFrame(columns=['user_id', 'movie_id', 'movie_title', 'predicted_rating', 'movie_genres'])
    finally:
        if not database.is_closed():
            database.close()

    if not rows:
        return pd.DataFrame(columns=['user_id', 'movie_id', 'movie_title', 'predicted_rating', 'movie_genres'])

    df = pd.DataFrame(rows)
    df = clean_dataframe_bytes(df)
    # normalize id columns
    df = _normalize_id_col(df, 'user_id')
    if 'movie_id' in df.columns:
        df = _normalize_id_col(df, 'movie_id')
    df['user_id'] = df['user_id'].astype(str)
    df['movie_title'] = df['movie_title'].astype(str)
    # include movie_genres if present
    if 'movie_genres' in df.columns:
        df['movie_genres'] = df['movie_genres'].astype(str)
    df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce').fillna(0.0)
    return df
