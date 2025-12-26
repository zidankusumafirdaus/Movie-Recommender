from peewee import TextField, IntegerField
from apps.extensions import BaseModel

class User(BaseModel):
    user_id = TextField(primary_key=True)
    user_age = IntegerField(null=True)
    user_gender = TextField(null=True)
    user_occupation = TextField(null=True)
    user_occupation_label = TextField(null=True)
    user_zip_code = TextField(null=True)

    class Meta:
        table_name = 'users'
