from flask import request, jsonify
from marshmallow import ValidationError
from apps.models import User
from apps.extensions import database
from apps.schemas.user_schema import CreateUserSchema

def create_user():
    data = request.get_json() or {}
    try:
        payload = CreateUserSchema().load(data)
    except ValidationError as err:
        return jsonify({'errors': err.messages}), 400

    user_id = str(payload.get('user_id'))

    try:
        database.connect(reuse_if_open=True)
        try:
            existing = User.get_or_none(User.user_id == user_id)
            if existing is not None:
                return jsonify({'error': 'user already exists', 'user_id': user_id}), 409

            kwargs = {
                'user_id': user_id,
                'user_age': payload.get('user_age'),
                'user_gender': payload.get('user_gender'),
                'user_occupation': payload.get('user_occupation'),
                'user_occupation_label': payload.get('user_occupation_label'),
                'user_zip_code': payload.get('user_zip_code'),
            }

            User.create(**kwargs)
        finally:
            if not database.is_closed():
                database.close()
    except Exception as e:
        return jsonify({'error': 'failed to create user', 'details': str(e)}), 500

    return jsonify({'message': 'user created', 'user_id': user_id}), 201