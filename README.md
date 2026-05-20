# 🛍️ Customer Segmentation menggunakan Fuzzy C-Means (FCM)

Proyek ini adalah implementasi algoritma *Machine Learning* untuk melakukan segmentasi pelanggan mall. Tujuan utama dari proyek ini adalah membantu pihak manajemen mall memahami karakteristik finansial pengunjung agar strategi pemasaran (seperti pemberian diskon atau penawaran keanggotaan VIP) dapat dilakukan secara tepat sasaran.



---

## 🧠 Metodologi & Algoritma
Proyek ini menggunakan algoritma **Fuzzy C-Means (FCM)**. 
Berbeda dengan K-Means konvensional yang membagi data secara kaku (*hard clustering*), FCM menggunakan pendekatan *soft clustering* (derajat keanggotaan). Hal ini sangat cocok untuk data perilaku manusia di dunia nyata, di mana seorang pelanggan mungkin memiliki kemiripan sifat dengan lebih dari satu kelompok sekaligus, sebelum akhirnya dimasukkan ke kelompok dengan probabilitas tertinggi.

**Atribut Data yang Dianalisis:**
1. Usia (Age)
2. Pendapatan Tahunan (Annual Income)
3. Skor Pengeluaran (Spending Score)

---

## 📂 Struktur Repositori (Data Pipeline)
Proyek ini dipecah menjadi tiga tahap (*modular*) sesuai standar industri perangkat lunak:

- `data/Pengunjung_Mall.csv` : Dataset mentah.
- `preprocess.py` : Skrip untuk membuang kolom yang tidak relevan (ID, Gender) dan melakukan normalisasi rentang skala data (0-1) menggunakan `MinMaxScaler`.
- `train-fcm.py` : Inti program tempat mesin belajar memetakan 200 data ke dalam 5 titik pusat (centroid).
- `analytics.py` : Skrip untuk merender visualisasi hasil (Scatter Plot) dan tabel profil kelompok.
- `requirements.txt` : Daftar dependensi *environment*.

---

## 🚀 Panduan Eksekusi (How to Run)

### 1. Persiapan Environment
Pastikan Python sudah terinstal di sistem. Buka terminal di direktori proyek ini dan jalankan perintah berikut untuk menginstal semua *library* pendukung:
```bash
pip install -r requirements.txt
