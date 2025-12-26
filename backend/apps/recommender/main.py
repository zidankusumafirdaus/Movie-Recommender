import logging

logging.basicConfig(level=logging.INFO)

from apps.recommender.loaders import load_recommendations
from apps.recommender.recommenders import (
    content_based_recommendation,
    popular_recommendation,
    populate_popular_recommendations,
)
from apps.recommender.generator import generate_recommendations, update_all_recommendations
from apps.recommender.normalize import _normalize_id_col, _normalize_series_0_1
from apps.recommender.profile import _build_user_genre_profile_from_df, _movie_genre_vector

__all__ = [
    'load_recommendations',
    'content_based_recommendation',
    'popular_recommendation',
    'populate_popular_recommendations',
    'generate_recommendations',
    'update_all_recommendations',
    '_normalize_id_col',
    '_normalize_series_0_1',
    '_build_user_genre_profile_from_df',
    '_movie_genre_vector',
]