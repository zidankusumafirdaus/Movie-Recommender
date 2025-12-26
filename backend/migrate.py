from dotenv import load_dotenv
from apps.extensions import database
from config import Config

load_dotenv()


def run_migration():
    from apps.models.movie import Movies
    from apps.models.rating import Ratings
    from apps.models.recommendation import Recommendations
    from apps.models.user import User

    try:
        database.connect(reuse_if_open=True)
        database.create_tables([Movies, Ratings, Recommendations, User])
        print("Tables ensured (Peewee)")
    except Exception as exc:
        print("⚠️ Peewee migration failed:", exc)
    finally:
        if not database.is_closed():
            database.close()

if __name__ == '__main__':
    run_migration()