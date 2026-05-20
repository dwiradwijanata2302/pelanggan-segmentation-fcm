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
    
    # Normalisasi
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_bersih), columns=df_bersih.columns)
    
    # Simpan file
    df_norm.to_csv(OUTPUT_PATH, index=False)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"SUKSES! File baru tercipta di: {OUTPUT_PATH}")
    else:
        print("ERROR: File gagal dibuat. Cek izin akses folder Anda.")

if __name__ == "__main__":
    preprocess_data()