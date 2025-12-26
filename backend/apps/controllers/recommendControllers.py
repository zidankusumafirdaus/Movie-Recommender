from flask import request, jsonify
from marshmallow import ValidationError
from apps.schemas.rating_schema import CreateRatingSchema, RecommendationResponseSchema
from apps.recommender.main import (
    load_recommendations,
    content_based_recommendation,
    popular_recommendation,
)
from apps.models import User, Movies, Ratings
import time
from apps.extensions import database

def index():
    return jsonify({"message": "🎬 Hybrid Movie Recommender API is running!"})

def recommend_by_id(user_id):
    k = int(request.args.get('k', 5))
    recommend_df = load_recommendations()

    if not recommend_df.empty and user_id in recommend_df['user_id'].astype(str).values:
        user_recs = (
            recommend_df[recommend_df['user_id'].astype(str) == user_id]
            .nlargest(k, 'predicted_rating')
            [['movie_id', 'movie_title', 'predicted_rating', 'movie_genres']]
            .to_dict(orient='records')
        )
        resp = {
            "user_id": user_id,
            "strategy": "collaborative_filtering",
            "recommendations": user_recs,
        }
        payload = RecommendationResponseSchema().dump(resp)
        return jsonify(payload)
    try:
        user = User.get(User.user_id == user_id)
        user_profile = {
            "user_age": user.user_age,
            "user_gender": user.user_gender,
            "user_occupation": user.user_occupation,
        }
        cbf_recs = content_based_recommendation(user_profile, k)
        if cbf_recs:
            resp = {
                "user_id": user_id,
                "strategy": "content_based",
                "recommendations": cbf_recs,
            }
            payload = RecommendationResponseSchema().dump(resp)
            return jsonify(payload)
    except Exception:
        pass

    resp = {
        "user_id": user_id,
        "strategy": "popular",
        "recommendations": popular_recommendation(k),
    }
    payload = RecommendationResponseSchema().dump(resp)
    return jsonify(payload)


def create_rating():
    data = request.get_json() or {}
    try:
        payload = CreateRatingSchema().load(data)
    except ValidationError as err:
        return jsonify({"errors": err.messages}), 400

    user_id = str(payload.get('user_id'))
    movie_id = str(payload.get('movie_id'))
    user_rating = payload.get('user_rating')

    try:
        user = User.get(User.user_id == user_id)
    except User.DoesNotExist:
        return jsonify({"error": "user not found"}), 404

    try:
        movie = Movies.get(Movies.movie_id == movie_id)
    except Movies.DoesNotExist:
        return jsonify({"error": "movie not found"}), 404

    # bucketize age (simple 10-year buckets)
    bucketized = None
    try:
        if user.user_age is not None:
            age = int(user.user_age)
            lower = (age // 10) * 10
            bucketized = f"{lower}-{lower+9}"
    except Exception:
        bucketized = None

    rating_data = {
        "user_id": user.user_id,
        "movie_id": movie.movie_id,
        "movie_title": movie.movie_title,
        "movie_genres": movie.movie_genres,
        "bucketized_user_age": bucketized,
        "user_age": str(user.user_age) if user.user_age is not None else None,
        "user_gender": user.user_gender,
        "user_occupation": user.user_occupation,
        "user_occupation_label": getattr(user, 'user_occupation_label', None),
        "user_zip_code": user.user_zip_code,
        "user_rating": float(user_rating) if user_rating is not None else None,
        "timestamp": int(time.time()),
    }

    try:
        Ratings.create(**rating_data)
    except Exception as e:
        return jsonify({"error": "failed to create rating", "details": str(e)}), 500

    return jsonify({"message": "rating recorded", "rating": rating_data}), 201

def get_all_movies():
    try:
        database.connect(reuse_if_open=True)
        rows = list(Movies.select(Movies.movie_id, Movies.movie_title, Movies.movie_genres).dicts())
    except Exception as e:
        return jsonify({"movies": []}), 200
    finally:
        if not database.is_closed():
            database.close()

    return jsonify({"movies": rows}), 200

def get_user_rated_movies(user_id):
    try:
        database.connect(reuse_if_open=True)
        q = (Ratings
             .select(Ratings.movie_id, Ratings.movie_title, Ratings.movie_genres, Ratings.user_rating, Ratings.timestamp)
             .where(Ratings.user_id == user_id)
             .order_by(Ratings.timestamp.desc()))
        rows = list(q.dicts())
    except Exception as e:
        return jsonify({"user_id": user_id, "rated_movies": []}), 200
    finally:
        if not database.is_closed():
            database.close()

    return jsonify({"user_id": user_id, "rated_movies": rows}), 200