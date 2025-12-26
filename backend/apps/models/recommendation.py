from peewee import TextField, FloatField, ForeignKeyField
from apps.extensions import BaseModel
from apps.models.user import User
from apps.models.movie import Movies

class Recommendations(BaseModel):
    # Use ForeignKeyField to establish relationships at the ORM level.
    # Keep column names `user_id` and `movie_id` so existing DB rows remain valid.
    user_id = ForeignKeyField(User, backref='recommendations', column_name='user_id', field='user_id', null=True)
    movie_id = ForeignKeyField(Movies, backref='recommendations', column_name='movie_id', field='movie_id', null=True)
    movie_title = TextField(null=True)
    predicted_rating = FloatField(null=True)
    movie_genres = TextField(null=True)

    class Meta:
        primary_key = False
