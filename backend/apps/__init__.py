from flask import Flask
from config import Config


def create_app():
    from apps.routes.recommend import recommend_bp
    from apps.routes.user import user_bp

    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(recommend_bp, url_prefix='/api')
    app.register_blueprint(user_bp, url_prefix='/api')

    # Startup tasks: ensure data present and start background updater
    try:
        from apps.startup import ensure_bootstrap_data, start_background_tasks
        ensure_bootstrap_data()
        start_background_tasks(app)
    except Exception as e:
        print("Startup tasks error:", e)
    return app