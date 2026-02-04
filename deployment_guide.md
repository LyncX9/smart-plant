# Panduan Deployment Cloud (Gratis & Mudah) - SmartPlant Vision 🚀

Panduan ini akan membantu Anda mengonlinekan **Backend AI (Python)** agar bisa diakses oleh Aplikasi Mobile dari mana saja (bukan hanya local emulator).

Solusi terbaik dan gratis saat ini adalah **Render.com**.

## Prasyarat
1.  Akun GitHub (Anda sudah punya).
2.  Akun [Render.com](https://render.com) (Daftar GRATIS menggunakan akun GitHub).

---

## Langkah 1: Persiapan Repository

Saya sudah menyiapkan folder khusus `backend_api` yang berisi semua file siap deploy:
*   `main.py` (FastAPI Server)
*   `model_engine.py` (Logika AI)
*   `Dockerfile` (Konfigurasi Server)
*   `requirements.txt` (Library Python)
*   `best_model_fixed.pth` (Model AI Anda)

**Tugas Anda:**
Push (upload) folder `backend_api` ini ke GitHub Repository Anda.

```bash
git add backend_api
git commit -m "Add production ready backend api"
git push origin main
```

---

## Langkah 2: Deploy ke Render.com

1.  Login ke [Render.com Dashboard](https://dashboard.render.com/).
2.  Klik tombol **New +** -> pilih **Web Service**.
3.  Pilih **Build and deploy from a Git repository**.
4.  Cari dan pilih repository `SmartPlant` Anda, klik **Connect**.
5.  Isi form konfigurasi:
    *   **Name**: `smartplant-api` (atau nama unik lain)
    *   **Region**: `Singapore` (Paling cepat ke Indonesia)
    *   **Branch**: `main`
    *   **Root Directory**: `backend_api` (⚠️ SANGAT PENTING: Harus diisi ini karena file ada di dalam folder)
    *   **Runtime**: `Docker`
    *   **Instance Type**: `Free`
6.  Klik tombol **Create Web Service**.

Render akan mulai membangun server. Proses ini butuh waktu **5-10 menit** (install library Python dan build Docker).
Tunggu sampai statusnya **Live** (Hijau).

Salin URL yang diberikan Render (contoh: `https://smartplant-api-xyz.onrender.com`). Ini adalah **BASE URL** Anda.

---

## Langkah 3: Update Aplikasi Flutter

Setelah backend online, Anda harus menghubungkan aplikasi Android ke server baru ini (bukan localhost lagi).

1.  Buka file `lib/core/constants/api_constants.dart` atau file konfigurasi URL Anda. (Cari di `lib/core/config/` jika ada).
2.  Ganti URL:
    *   Lama: `http://10.0.2.2:5000`
    *   Baru: `https://smartplant-api-xyz.onrender.com` (Pakai URL dari Render)
3.  **Hapus path `/analyze`** dari Base URL jika kodingan menggabungkannya otomatis. Pastikan strukturnya benar.

---

## Langkah 4: Build Ulang APK

Karena URL berubah, APK harus dibuat ulang.

```bash
flutter clean
flutter build apk --release
```

APK baru di `build/app/outputs/flutter-apk/app-release.apk` sekarang sudah "Production Ready" dan bisa dipakai siapa saja tanpa perlu laptop Anda menyala!

---

## Troubleshooting

*   **Error Memory**: Render Free Tier hanya punya RAM 512MB. Jika server crash saat loading model, kita mungkin perlu pakai model yang lebih kecil (Quantized) atau pindah ke Google Cloud Run.
*   **Slow Cold Start**: Server gratisan "tidur" jika tidak dipakai 15 menit. Request pertama mungkin lemot (50 detik). Ini wajar.
