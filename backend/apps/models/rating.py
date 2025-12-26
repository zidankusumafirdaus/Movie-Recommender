from peewee import TextField, FloatField, IntegerField, ForeignKeyField
from apps.extensions import BaseModel
from apps.models.user import User
from apps.models.movie import Movies

class Ratings(BaseModel):
    user_id = ForeignKeyField(User, backref='ratings', column_name='user_id', field='user_id', null=True)
    movie_id = ForeignKeyField(Movies, backref='ratings', column_name='movie_id', field='movie_id', null=True)
    movie_title = TextField(null=True)
    movie_genres = TextField(null=True)
    bucketized_user_age = TextField(null=True)
    user_age = TextField(null=True)
    user_gender = TextField(null=True)
    user_occupation = TextField(null=True)
    user_occupation_label = TextField(null=True)
    user_zip_code = TextField(null=True)
    user_rating = FloatField(null=True)
    timestamp = IntegerField(null=True)

    class Meta:
        primary_key = False
