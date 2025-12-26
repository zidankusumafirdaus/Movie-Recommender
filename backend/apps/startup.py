import threading
import time
from config import Config
from apps.utils.sampledb import ensure_sample_data
from apps.recommender.main import update_all_recommendations, populate_popular_recommendations
from apps.extensions import database
from apps.models.recommendation import Recommendations


def ensure_bootstrap_data():
    ensure_sample_data(Config.DB_PATH, Config.MOVIES_CSV, Config.RATINGS_CSV, Config.USERS_CSV)

    exists = False
    count = 0
    try:
        database.connect(reuse_if_open=True)
        exists = database.table_exists('recommendations')
        if exists:
            count = Recommendations.select().count()
    except Exception as e:
        print("Recommendations table check failed:", e)
    finally:
        if not database.is_closed():
            database.close()

    if not exists or count == 0:
        try:
            populate_popular_recommendations(k=5)
        except Exception as e:
            print("populate_popular_recommendations failed:", e)


def start_background_tasks(app):
    def _loop():
        while True:
            try:
                with app.app_context():
                    update_all_recommendations()
            except Exception as e:
                print("Background generator error:", e)
            time.sleep(Config.UPDATE_INTERVAL)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
