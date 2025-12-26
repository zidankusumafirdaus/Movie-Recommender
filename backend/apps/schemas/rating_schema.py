from marshmallow import Schema, fields, validate, INCLUDE

class CreateRatingSchema(Schema):
    user_id = fields.Str(required=True, validate=validate.Length(min=1))
    movie_id = fields.Str(required=True, validate=validate.Length(min=1))
    user_rating = fields.Float(required=False, allow_none=True)

class RecommendationItemSchema(Schema):
    movie_id = fields.Str(required=False, allow_none=True)
    movie_title = fields.Str(required=True)
    movie_genres = fields.Str(required=False, allow_none=True)
    predicted_rating = fields.Float(required=False, allow_none=True)

    class Meta:
        unknown = INCLUDE

class RecommendationResponseSchema(Schema):
    user_id = fields.Str(required=True, allow_none=True)
    strategy = fields.Str(required=True)
    recommendations = fields.List(fields.Nested(RecommendationItemSchema), required=True)

    class Meta:
        unknown = INCLUDE