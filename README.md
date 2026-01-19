# Deepfake HoaxBuster (UAS Machine Learning)

Deepfake HoaxBuster adalah aplikasi web untuk **mendeteksi wajah Real vs Fake (deepfake)** berbasis Deep Learning menggunakan **MobileNetV2 Transfer Learning**.  
Proyek ini dikembangkan untuk memenuhi kebutuhan **UAS Mata Kuliah Machine Learning**, mencakup: paper ilmiah, laporan teknis, source code, dan demo aplikasi web.

---

## Fitur Utama
1. **Klasifikasi Real vs Fake (Deepfake Detection)** dari gambar wajah menggunakan model `.keras`.
2. **Konfigurasi threshold prediksi** (melalui `models/eval_config.json`) untuk mengatur sensitivitas deteksi Fake.  
   - Probabilitas yang dihitung adalah untuk kelas **Fake**.
   - Prediksi **Fake** jika `prob_fake >= threshold`. (Default threshold saat ini: `0.05`)

> Catatan threshold: untuk demo UAS, kamu bisa menyesuaikan threshold agar lebih seimbang (mengurangi false positive pada gambar real).

---

## Tech Stack
**Backend**
- Python + Flask
- TensorFlow/Keras (load model `.keras`)
- Template HTML (Jinja) + Static CSS

**Frontend**
- React + Vite (folder `frontend/`)

---

## Struktur Repository

## Struktur Repository

```text
Deepfake-HoaxBuster/
├── app.py                              # Backend Flask (API + server)
├── requirements.txt                    # Dependency backend
├── models/
│   ├── best_mobilenetv2_real_fake.keras  # Model terlatih MobileNetV2
│   └── eval_config.json                # Konfigurasi threshold & label map
├── templates/                          # HTML Templates (Flask)
│   ├── base.html
│   ├── landing.html
│   ├── index.html
│   └── auth/                           # Folder auth & user (opsional)
├── static/
│   ├── css/
│   │   └── style.css
│   └── profile_pics/                   # Runtime folder foto profil (Ignored)
│       └── .gitkeep
├── instance/                           # Runtime config/DB lokal (Ignored)
│   └── .gitkeep
├── uploads/                            # Runtime upload gambar (Ignored)
│   └── .gitkeep
└── frontend/                           # Frontend React + Vite
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── api.js
        ├── App.jsx
        └── main.jsxyaml
```

## Cara Menjalankan (Local)

### A. Menjalankan Backend (Flask)
> Pastikan kamu berada di folder root project (selevel `app.py`)

#### 1) Buat virtual environment (opsional tapi disarankan)
**Windows (PowerShell)**
``` bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux:**
``` bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependency
``` bash
pip install -r requirements.txt
```
4) Jalankan backend
``` bash
python app.py
```
Biasanya backend berjalan di:
```
http://127.0.0.1:5000 (tergantung konfigurasi di app.py)
```
B. Menjalankan Frontend (React + Vite)
Buka terminal baru dan masuk ke folder frontend/

``` bash
cd frontend
npm install
npm run dev
```
Biasanya frontend berjalan di:

http://localhost:5173

Konfigurasi Model & Threshold
Konfigurasi inference ada di:

models/eval_config.json

Contoh isi:

threshold: batas keputusan Fake

label_map: pemetaan kelas

note: informasi bahwa probability adalah untuk kelas Fake

Jika kamu ingin mengubah sensitivitas deteksi, ubah nilai threshold pada file tersebut.

Cara Demo (untuk Video UAS)
Rekomendasi alur demo 10–20 menit:

Perkenalan singkat: problem deepfake dan tujuan aplikasi.

Tunjukkan struktur repo + file penting (app.py, models/, frontend/).

Jalankan backend (python app.py).

Jalankan frontend (npm run dev).

Demo upload 2–3 gambar:

contoh Real

contoh Fake

Tampilkan output:

label prediksi (Real/Fake)

nilai probabilitas (prob_fake)

threshold yang digunakan

Tutup dengan kesimpulan + rencana improvement (misal threshold tuning / explainability).

Troubleshooting
1) Error model tidak ketemu
Pastikan file ada:

models/best_mobilenetv2_real_fake.keras

2) Folder runtime tidak ada
Pastikan folder ini ada (sudah disiapkan via .gitkeep):

instance/

uploads/

static/profile_pics/

3) Frontend tidak bisa akses backend (CORS / base URL)
Pastikan backend berjalan lebih dulu.

Cek file frontend/src/api.js untuk base URL backend (misal http://127.0.0.1:5000).

Jika perlu, aktifkan CORS pada Flask.

Author
Chairul Fitra Ramadhan (Kelompok 16 – IF B Sore)

UAS Machine Learning

```yaml
```

## Langkah cepat yang harus kamu lakukan sekarang
1) Buat file `README.md` di root project.
2) Paste isi README di atas.
3) Commit + push:

```powershell
git add README.md
git commit -m "docs: add UAS README with run instructions"
git push
```