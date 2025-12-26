import os
import pandas as pd
from peewee import fn
from apps.utils.cleaning import clean_dataframe_bytes
from migrate import run_migration
from apps.extensions import database
from apps.models.movie import Movies
from apps.models.rating import Ratings
from apps.models.user import User


def _ensure_tables(db_path: str):
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    try:
        run_migration()
    except Exception as e:
        print("run_migration failed:", e)


def import_sample_data(db_path: str, movies_csv: str, ratings_csv: str, users_csv: str = None):
    movies_df = pd.read_csv(movies_csv, dtype=str, keep_default_na=False)
    movies_df = clean_dataframe_bytes(movies_df)
    movies_df['movie_genres'] = movies_df['movie_genres'].astype(str)

    ratings_df = pd.read_csv(ratings_csv, dtype=str, keep_default_na=False)
    ratings_df = clean_dataframe_bytes(ratings_df)

    if 'user_occupation_text' in ratings_df.columns:
        ratings_df.rename(columns={'user_occupation_text': 'user_occupation'}, inplace=True)
    if 'raw_user_age' in ratings_df.columns:
        ratings_df.rename(columns={'raw_user_age': 'user_age'}, inplace=True)

    ratings_df['user_rating'] = pd.to_numeric(ratings_df.get('user_rating', 0), errors='coerce').fillna(0.0)
    ratings_df['timestamp'] = pd.to_numeric(ratings_df.get('timestamp', 0), errors='coerce').fillna(0).astype(int)

    users_df = None
    if users_csv:
        users_df = pd.read_csv(users_csv, dtype=str, keep_default_na=False)
        users_df = clean_dataframe_bytes(users_df)
        # normalize column names from sample file
        if 'user_occupation_text' in users_df.columns:
            users_df.rename(columns={'user_occupation_text': 'user_occupation'}, inplace=True)
        if 'raw_user_age' in users_df.columns:
            users_df.rename(columns={'raw_user_age': 'user_age'}, inplace=True)

    try:
        database.connect(reuse_if_open=True)
        Movies.delete().execute()
        Ratings.delete().execute()
        if users_df is not None:
            User.delete().execute()

        # Chunked bulk inserts to avoid SQLite 'too many SQL variables'
        if not movies_df.empty:
            movies_records = movies_df.to_dict(orient='records')
            for i in range(0, len(movies_records), 500):
                Movies.insert_many(movies_records[i:i+500]).execute()
        if not ratings_df.empty:
            ratings_records = ratings_df.to_dict(orient='records')
            for i in range(0, len(ratings_records), 500):
                Ratings.insert_many(ratings_records[i:i+500]).execute()
        # Insert unique users (by user_id) if users_df provided
        if users_df is not None and not users_df.empty:
            # select relevant columns and drop duplicates
            udf = users_df
            # prefer columns: user_id, user_age, user_gender, user_occupation, user_occupation_label, user_zip_code
            cols = []
            for c in ['user_id', 'user_age', 'user_gender', 'user_occupation', 'user_occupation_label', 'user_zip_code']:
                if c in udf.columns:
                    cols.append(c)
            if 'user_id' not in cols and 'user_id' in udf.columns:
                cols.insert(0, 'user_id')
            users_records = udf[cols].drop_duplicates(subset=['user_id']).to_dict(orient='records')
            for i in range(0, len(users_records), 500):
                User.insert_many(users_records[i:i+500]).execute()
    finally:
        if not database.is_closed():
            database.close()

    print("✅ Data sample berhasil dimasukkan ke movies.db")


def ensure_sample_data(db_path: str, movies_csv: str, ratings_csv: str, users_csv: str = None):
    _ensure_tables(db_path)

    try:
        database.connect(reuse_if_open=True)
        movies_count = Movies.select(fn.COUNT(Movies.movie_id)).scalar() or 0
        ratings_count = Ratings.select(fn.COUNT(Ratings.movie_id)).scalar() or 0
        users_count = 0
        try:
            users_count = User.select(fn.COUNT(User.user_id)).scalar() or 0
        except Exception:
            users_count = 0
    except Exception:
        movies_count, ratings_count, users_count = 0, 0, 0
    finally:
        if not database.is_closed():
            database.close()
    if movies_count > 0 and ratings_count > 0 and (users_csv is None or users_count > 0):
        print(f"🎯 Sample data already present (movies={movies_count}, ratings={ratings_count}, users={users_count}). Skipping import.")
        return

    print(f"ℹ️  Sample data missing (movies={movies_count}, ratings={ratings_count}, users={users_count}). Importing CSVs...")
    import_sample_data(db_path, movies_csv, ratings_csv, users_csv)
