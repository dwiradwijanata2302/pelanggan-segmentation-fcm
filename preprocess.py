import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# 1. Mendeteksi lokasi folder script ini berada secara otomatis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'Pengunjung_Mall.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data_siap_fcm.csv')

def preprocess_data():
    print(f"Mencari file di: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print("ERROR: File Pengunjung_Mall.csv tidak ditemukan di dalam folder 'data'!")
        return

    # Membaca data
    df = pd.read_csv(INPUT_PATH)
    
    # Pembersihan
    df_bersih = df.drop(['ID_Pelanggan', 'Gender'], axis=1)

    # Pembersihan anomali data (outlier) dengan metode IQR
    Q1 = df_bersih.quantile(0.25)
    Q3 = df_bersih.quantile(0.75)
    IQR = Q3 - Q1

    df_tanpa_outlier = df_bersih[
        ~((df_bersih < (Q1 - 1.5 * IQR)) |
          (df_bersih > (Q3 + 1.5 * IQR))).any(axis=1)
    ]

    print(f"Jumlah data sebelum pembersihan : {len(df_bersih)}")
    print(f"Jumlah data setelah pembersihan : {len(df_tanpa_outlier)}")
    
    # Normalisasi
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_tanpa_outlier), columns=df_tanpa_outlier.columns)
    
    # Simpan file
    df_norm.to_csv(OUTPUT_PATH, index=False)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"SUKSES! data siap tersimpan di : {OUTPUT_PATH}")
    else:
        print("ERROR: File gagal dibuat. Cek izin akses folder Anda.")

if __name__ == "__main__":
    preprocess_data()