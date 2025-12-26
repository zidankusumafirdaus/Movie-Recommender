import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'super-secret-key')
    JWT_SECRET_KEY = SECRET_KEY

    SAVE_DIR = os.getenv('SAVE_DIR', './database')
    DB_PATH = os.getenv('DB_PATH', os.path.join(SAVE_DIR, 'movies.db'))

    MOVIES_CSV = os.getenv('MOVIES_CSV', './data/movielens_100k_movies.csv')
    RATINGS_CSV = os.getenv('RATINGS_CSV', './data/movielens_100k_ratings.csv')
    USERS_CSV = os.getenv('USERS_CSV', './data/movielens_100k_users.csv')

    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '300'))