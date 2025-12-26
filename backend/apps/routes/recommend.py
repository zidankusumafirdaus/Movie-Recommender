from flask import Blueprint
from apps.controllers import RecommendController

recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/', methods=['GET'])
def index():
    return RecommendController.index()

@recommend_bp.route('/recommend/<user_id>', methods=['GET'])
def recommend_by_id(user_id):
    return RecommendController.recommend_by_id(user_id)

@recommend_bp.route('/rating', methods=['POST'])
def create_rating():
    return RecommendController.create_rating()

@recommend_bp.route('/movies', methods=['GET'])
def get_all_movies():
    return RecommendController.get_all_movies()

@recommend_bp.route('/movies/rated/<user_id>', methods=['GET'])
def get_user_rated_movies(user_id):
    return RecommendController.get_user_rated_movies(user_id)
