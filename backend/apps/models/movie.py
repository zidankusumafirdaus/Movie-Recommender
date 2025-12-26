from peewee import TextField
from apps.extensions import BaseModel

class Movies(BaseModel):
    movie_id = TextField(primary_key=True)
    movie_title = TextField(null=True)
    movie_genres = TextField(null=True)
