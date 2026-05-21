# Customer Segmentation using Fuzzy C-Means

Proyek ini berisi implementasi segmentasi pelanggan mall menggunakan algoritma Fuzzy C-Means yang ditulis manual dari nol, tanpa library fuzzy siap pakai. Tujuannya adalah mengelompokkan pelanggan berdasarkan karakteristik usia, pendapatan tahunan, dan skor pengeluaran agar hasil analisis bisa dipakai untuk strategi pemasaran yang lebih tepat sasaran.

## Gambaran Proyek

Pipeline proyek dibagi menjadi tiga tahap:

1. `preprocess.py` membersihkan data, menghapus kolom yang tidak diperlukan, membuang outlier dengan metode IQR, lalu melakukan normalisasi dengan `MinMaxScaler`.
2. `train-fcm.py` menjalankan Fuzzy C-Means manual dari nol menggunakan `numpy`, lalu memberi label cluster pada data bersih.
3. `visualization.py` menampilkan scatter plot hasil clustering dan profil rata-rata tiap cluster.

## Struktur File

- `data/Pengunjung_Mall.csv` : dataset mentah.
- `preprocess.py` : preprocessing data, outlier removal, dan normalisasi.
- `train-fcm.py` : implementasi FCM manual tanpa `skfuzzy`.
- `visualization.py` : visualisasi hasil cluster dan tabel profil kelompok.
- `requirements.txt` : daftar dependency.
- `.gitignore` : mengabaikan file hasil generate seperti `data_siap_fcm.csv` dan `hasil_akhir.csv`.

## Alur Kerja

### 1. Preprocessing

File `preprocess.py` akan:

- membaca `data/Pengunjung_Mall.csv`
- menghapus kolom `ID_Pelanggan` dan `Gender`
- menghapus outlier dengan metode IQR
- melakukan normalisasi nilai ke rentang 0-1
- menyimpan hasil ke `data_siap_fcm.csv`

### 2. Training FCM Manual

File `train-fcm.py` akan:

- membaca `data_siap_fcm.csv`
- menjalankan Fuzzy C-Means
- menghitung membership, centroid, jarak, dan konvergensi secara iteratif
- menyimpan hasil cluster ke `hasil_akhir.csv`

### 3. Visualisasi

File `visualization.py` akan:

- membaca `hasil_akhir.csv`
- menampilkan scatter plot berdasarkan `Pendapatan_Tahunan_Ribuan_USD` dan `Pengeluaran_USD`
- menampilkan tabel profil rata-rata tiap cluster
- membulatkan rata-rata usia, sedangkan pendapatan dan pengeluaran tetap sebagai nilai rata-rata asli

## Cara Menjalankan

### 1. Install dependency

Pastikan Python sudah terinstal, lalu jalankan:

```bash
pip install -r requirements.txt
```

### 2. Jalankan preprocessing

```bash
python preprocess.py
```

### 3. Jalankan training FCM manual

```bash
python train-fcm.py
```

### 4. Tampilkan hasil visualisasi

```bash
python visualization.py
```

## Catatan Penting

- File `data_siap_fcm.csv` dan `hasil_akhir.csv` adalah file hasil generate, jadi tidak perlu dipush ke GitHub.
- Implementasi clustering sudah tidak memakai library fuzzy eksternal seperti `skfuzzy`.
- Jika jumlah baris berubah karena outlier dihapus, itu normal karena clustering dijalankan pada data bersih.

## Hasil Output

Setelah pipeline selesai, Anda akan mendapatkan:

- file data bersih dan ternormalisasi: `data_siap_fcm.csv`
- file hasil clustering: `hasil_akhir.csv`
- scatter plot hasil segmentasi
- profil karakteristik tiap cluster
