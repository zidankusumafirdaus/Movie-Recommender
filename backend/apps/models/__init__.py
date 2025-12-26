from .movie import Movies
from .rating import Ratings
from .recommendation import Recommendations
from .user import User
from apps.extensions import database

def create_tables():
    with database:
        database.create_tables([Movies, Ratings, Recommendations, User])
