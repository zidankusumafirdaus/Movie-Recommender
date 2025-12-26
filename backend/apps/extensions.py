import os
from peewee import SqliteDatabase, Model
from config import Config

db_dir = os.path.dirname(Config.DB_PATH) or '.'
os.makedirs(db_dir, exist_ok=True)

database = SqliteDatabase(Config.DB_PATH)

class BaseModel(Model):
    class Meta:
        database = database