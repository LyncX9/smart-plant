# Dokumentasi Teknis SmartPlant Vision

**Versi Dokumen:** 1.0  
**Tanggal:** 4 Februari 2026

---

## 1. Pendahuluan
**SmartPlant Vision** adalah sistem cerdas berbasis Artificial Intelligence (AI) dan Computer Vision untuk mendeteksi kesehatan tanaman padi melalui citra daun. Sistem ini menggabungkan Deep Learning untuk klasifikasi penyakit dan Image Processing untuk analisis morfometri fisik (urat daun dan bercak lesi).

### Fitur Utama
*   **Deteksi Penyakit**: BrownSpot, Hispa, LeafBlast, dan Healthy.
*   **Analisis Fisik**: Menghitung persentase area luka (lesi) dan kepadatan urat daun.
*   **Health Score**: Skor kesehatan 0-100 berdasarkan kombinasi AI dan kerusakan fisik.
*   **Mobile App**: Aplikasi Android user-friendly untuk petani.

---

## 2. Machine Learning & Deep Learning

### Arsitektur Model
Sistem menggunakan **MobileNetV2** sebagai backbone utama. Model ini dipilih karena ringan dan cepat, sangat cocok untuk deployment mobile atau cloud server dengan resource terbatas.

*   **Base Model**: MobileNetV2 (Pre-trained on ImageNet).
*   **Transfer Learning**: Layer classifier terakhir diganti (Fine-tuning).
*   **Classifier Head**:
    *   Dropout (0.5) untuk mencegah overfitting.
    *   Linear Layer (Output 4 kelas: BrownSpot, Healthy, Hispa, LeafBlast).
*   **Input Size**: 224x224 piksel.

### Training Strategy
*   **Dataset**: Dataset citra daun padi (Rice Leaf Disease Dataset).
*   **Augmentasi**: Rotasi, Flip, Brightness adjust untuk memperbanyak variasi data latih.
*   **Loss Function**: CrossEntropyLoss.
*   **Optimizer**: Adam / SGD dengan Learning Rate scheduler. (Menggunakan `best_model_fixed.pth` yang memiliki akurasi terbaik).

### Test Time Augmentation (TTA)
Untuk meningkatkan akurasi saat prediksi (Inference), sistem menerapkan **Aggressive TTA**. Setiap gambar yang diupload akan diduplikasi menjadi **8 variasi** (Original, Flip Horizontal, Flip Vertical, Rotasi 90, dll). Hasil prediksi dari ke-8 gambar dirata-rata untuk mendapatkan confidence score yang lebih stabil.

### Validasi Objek (Out-of-Distribution)
Sistem memiliki mekanisme keamanan:
*   Jika **Confidence Score < 35%**, sistem menganggap objek tersebut **Bukan Daun Padi** ("Unknown Object").

---

## 3. Algoritma Image Processing (Computer Vision)

Selain AI, sistem menggunakan algoritma pengolahan citra klasik untuk ekstraksi fitur fisik.

### A. Segmentasi Daun (`segment_leaf`)
Memisahkan daun dari latar belakang.
*   **Metode**: Color Thresholding pada ruang warna **HSV**.
*   **Range Warna**: Hijau Daun (H: 20-100, S: 30-255, V: 30-255).
*   **Pembersihan**: Operasi Morfologi (Closing & Opening) untuk membuang noise.

### B. Deteksi Lesi/Penyakit (`detect_lesions`)
Mengidentifikasi bercak luka pada daun.
*   **Metode**: Multi-range HSV Thresholding.
*   **Warna Target**:
    *   Coklat Tua (Brown): Penyakit BrownSpot/Blast.
    *   Kuning/Kecoklatan (Tan): Bercak kering.
*   **Output**: Persentase area lesi terhadap total area daun.

### C. Ekstraksi Urat Daun (`extract_veins`)
Mendeteksi struktur urat daun padi yang sejajar (parallel venation).
*   **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk memperjelas tekstur.
*   **Metode Utama**: **Morphological Top-Hat Transform**.
    *   Kernel Vertikal (1x15) & Horizontal (15x1) untuk menangkap garis tipis terang di latar gelap.
*   **Output**: Kepadatan urat (Vein Density) dan Kontinuitas.

### D. Perhitungan Health Score
Skor akhir kesehatan (0-100) dihitung dengan rumus pembobotan:

```python
Health Score = (CNN_Score * 0.6) + (Physical_Score * 0.4)
```
*   **CNN_Score**: Confidence dari model Deep Learning.
*   **Physical_Score**: 100 - (Persentase Area Lesi).

---

## 4. Arsitektur Sistem

### Backend API (Python FastAPI)
Logic pemrosesan berat berada di server (Cloud).
*   **Framework**: FastAPI (Python).
*   **Container**: Docker (Python 3.10 Slim + OpenCV System Deps).
*   **Endpoint Utama**: `POST /analyze` (Menerima gambar, mengembalikan JSON hasil analisis + Base64 Overlay).

### Mobile App (Flutter)
Antarmuka pengguna berbasis Android.
*   **State Management**: BLoC (Business Logic Component) untuk manajemen state yang bersih.
*   **Dependency Injection**: `get_it` & `injectable`.
*   **Navigasi**: Bottom Navigation Bar (Home, History, Guide).
*   **Fitur**:
    *   Capture Foto / Ambil dari Galeri.
    *   Crop & Rotate Image.
    *   Menampilkan Hasil Analisis Visual (Overlay Lesi & Urat).

---

## 5. Pembuatan APK (Build Process)

Langkah-langkah mengubah source code Flutter menjadi file APK siap instal.

### Persiapan Environment
1.  Install Flutter SDK & Android Studio.
2.  Konfigurasi `key.properties` (Untuk signing APK release).

### Konfigurasi Aplikasi (`pubspec.yaml`)
*   **Nama App**: Diatur di `AndroidManifest.xml` ("SmartPlant Vision").
*   **Icon**: Diconfig menggunakan `flutter_launcher_icons`.

### Perintah Build
Jalankan perintah berikut di terminal folder `smartplant-app`:

```bash
# 1. Bersihkan build lama (Wajib untuk menghindari cache error)
flutter clean

# 2. Update dependensi
flutter pub get

# 3. Build APK mode Release (Teroptimasi & Kecil)
flutter build apk --release
```

**Output**: File APK akan muncul di:
`build/app/outputs/flutter-apk/app-release.apk`

---

## 6. Deployment (Backend)

Backend dideploy menggunakan **Render.com** (Docker Runtime).
1.  Repo GitHub berisi: `Dockerfile`, `requirements.txt`, `main.py`, `model_engine.py`, `best_model_fixed.pth`.
2.  Render membaca `Dockerfile`, menginstall library, dan menjalankan server `uvicorn`.
3.  URL Cloud (misal: `https://smartplant-api.onrender.com`) dipasang ke aplikasi Flutter.

---
**Hak Cipta © 2026 SmartPlant Team**
