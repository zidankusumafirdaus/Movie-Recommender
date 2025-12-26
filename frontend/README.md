# Frontend — React + Vite

Singkat: frontend React yang dibangun dengan Vite. Aplikasi ini berinteraksi dengan backend rekomendasi (lihat `backend/`). README ini berfokus pada langkah cepat untuk development dan build.

**Prerequisites**
- Node.js 16+ (atau versi LTS terbaru)
- npm atau yarn

**Install & Run (dev)**
```powershell
cd frontend
npm install
npm run dev
```

Dev server default Vite akan tersedia pada `http://localhost:5173` (port bisa berbeda jika sudah dipakai).

**Build & Preview (production)**
```powershell
npm run build
npm run preview
```

**API / Backend**
- Frontend memanggil backend melalui `src/api/api.js`.
- Saat ini `baseURL` di `src/api/api.js` adalah `/api` (proxy). Untuk development terhubung ke backend lokal (mis. `http://localhost:5000`) sesuaikan `baseURL` atau atur proxy di `vite.config.js`.

Contoh sederhana (ubah jika perlu):
```javascript
// src/api/api.js
const API = axios.create({ baseURL: "http://localhost:5000", headers: { "Content-Type": "application/json" } });
```

**Scripts (package.json)**
- `dev` : jalankan Vite dev server
- `build` : buat build produksi
- `preview` : preview hasil build

**Struktur penting**
- `src/pages/` : halaman utama (`App.jsx`, `Movies.jsx`, `Users.jsx`)
- `src/components/` : komponen UI (MovieCard, RatingForm, RecommendationList, dll.)
- `src/api/api.js` : wrapper axios untuk semua request ke backend
- `public/` : aset statis

**Quick tips**
- Jika Anda menggunakan backend lokal di port 5000, atur `baseURL` di `src/api/api.js` ke `http://localhost:5000` atau tambahkan proxy di `vite.config.js`.
- Periksa CORS di backend jika fetch gagal (untuk production gunakan reverse proxy atau sama-origin hosting).

Butuh contoh request/response, konfigurasi proxy, atau Dockerfile untuk frontend? Saya bisa tambahkan.

