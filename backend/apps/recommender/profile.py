import numpy as np
from apps.utils.cleaning import parse_genres


def _build_user_genre_profile_from_df(user_id, ratings_df, movies_df):
    try:
        user_rows = ratings_df[ratings_df['user_id'].astype(str) == str(user_id)]
        if user_rows.empty:
            return {}
        merged = user_rows.merge(movies_df[['movie_id', 'genres_list']], on='movie_id', how='left')
        genre_scores = {}
        for _, r in merged.iterrows():
            grs = r.get('genres_list')
            if not grs:
                continue
            try:
                for g in grs:
                    genre_scores[g] = genre_scores.get(g, 0.0) + float(r.get('user_rating', 0.0))
            except Exception:
                # if stored as string, try to parse
                try:
                    for g in parse_genres(grs):
                        genre_scores[g] = genre_scores.get(g, 0.0) + float(r.get('user_rating', 0.0))
                except Exception:
                    continue
        # normalize to unit-vector style (L2)
        if not genre_scores:
            return {}
        vals = np.array(list(genre_scores.values()), dtype=float)
        norm = np.linalg.norm(vals)
        if norm == 0:
            return {k: 0.0 for k in genre_scores}
        return {k: v / norm for k, v in genre_scores.items()}
    except Exception:
        return {}


def _movie_genre_vector(movie_genres, all_genres):
    vec = np.zeros(len(all_genres), dtype=float)
    if not movie_genres:
        return vec
    try:
        gs = movie_genres
        if isinstance(gs, str):
            gs = parse_genres(gs)
        for g in gs:
            if g in all_genres:
                vec[all_genres.index(g)] = 1.0
    except Exception:
        try:
            for g in parse_genres(movie_genres):
                if g in all_genres:
                    vec[all_genres.index(g)] = 1.0
        except Exception:
            pass
    return vec
