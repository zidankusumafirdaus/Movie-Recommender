from marshmallow import Schema, fields, validate, INCLUDE

class CreateUserSchema(Schema):
    user_id = fields.Str(required=True, validate=validate.Length(min=1))
    user_age = fields.Int(required=False, allow_none=True)
    user_gender = fields.Str(required=False, allow_none=True)
    user_occupation = fields.Str(required=False, allow_none=True)
    user_occupation_label = fields.Str(required=False, allow_none=True)
    user_zip_code = fields.Str(required=False, allow_none=True)

    class Meta:
        unknown = INCLUDE