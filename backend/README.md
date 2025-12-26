**Project**
- **Description**: Backend API untuk sistem rekomendasi film (hybrid: collaborative + content-based + popular fallback).

**Requirements**
- **Python**: 3.8+ recommended
- **Dependencies**: lihat `requirements.txt` (Flask, TensorFlow, tensorflow-recommenders, pandas, peewee, dll.)

**Quick Start**
- Clone / buka folder `backend`.
- Buat virtual environment dan instal dependensi, jalankan migrasi, lalu server:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python migrate.py
python server.py
```

**Environment**
- Konfigurasi utama ada di file `.env` dan `config.py`:
	- `DB_PATH` (default `./database/movies.db`)
	- `MOVIES_CSV`, `RATINGS_CSV`, `USERS_CSV` (default `./data/...`)
	- `UPDATE_INTERVAL` (detik) — interval update rekomendasi otomatis
	- `SECRET_KEY` (opsional)

**API (penting)**
- `GET /` : health check
- `POST /recommend` : dapatkan rekomendasi. Body JSON contoh: `{ "user_id": "1", "k": 5, "user_age": 25, "user_gender": "M", "user_occupation": "doctor" }`
- `GET /recommend/<user_id>?k=5` : alternatif (existing/legacy)

**Data & Database**
- Sample data CSV ada di `data/` (`movielens_100k_*`).
- Database SQLite secara default disimpan di `database/movies.db`.
- Jalankan `python migrate.py` untuk membuat tabel (Peewee). Aplikasi juga mengimpor sample data saat start jika DB belum ada.

**Struktur singkat (file penting)**
- `server.py` : entrypoint untuk menjalankan Flask app (host 0.0.0.0:5000)
- `app.py` : implementasi endpoint rekomendasi + training loop (tfrs) — untuk eksperimen dan development
- `migrate.py` : buat/cek tabel Peewee
- `apps/recommender/` : generator, loaders, normalizer, profile, dan logic recommender
- `apps/controllers/` dan `apps/routes/` : endpoint dan kontroler API
- `apps/models/` : model Peewee (Movies, Ratings, Recommendations, User)

**Operational notes**
- Model training menggunakan TensorFlow dan dijalankan di background thread pada startup; membutuhkan resource (CPU/GPU) dan waktu.
- Untuk production: jalankan dengan WSGI server (gunicorn/uWSGI), set `debug=False`, dan atur `UPDATE_INTERVAL` sesuai kebutuhan.
- Jika ingin non-blocking/terjadwal, pertimbangkan memindahkan training ke job terpisah (Celery / cron / cloud job).

Jika perlu, saya bisa memperkaya README ini dengan contoh request/response, penjelasan skema DB, atau instruksi deployment.

