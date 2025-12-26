from flask import Blueprint
from apps.controllers import UserController

user_bp = Blueprint('user', __name__)

@user_bp.route('/users', methods=['POST'])
def create_user():
    return UserController.create_user()