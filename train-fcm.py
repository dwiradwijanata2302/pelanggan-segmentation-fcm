import pandas as pd
import numpy as np
import skfuzzy as fuzz

def run_fcm():
    # Load data yang sudah bersih
    df_norm = pd.read_csv('data_siap_fcm.csv')
    data = df_norm.values.T

    # Jalankan FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=5, m=2, error=0.005, maxiter=1000
    )

    # Ambil hasil cluster
    cluster_membership = np.argmax(u, axis=0)

    # Masukkan hasil ke file asli (Pengunjung_Mall.csv) agar mudah dibaca manusia
    df = pd.read_csv('data/Pengunjung_Mall.csv')
    df['Cluster'] = cluster_membership
    df.to_csv('hasil_akhir.csv', index=False)
    print("Proses FCM selesai! Hasil disimpan ke 'hasil_akhir.csv'")

if __name__ == "__main__":
    run_fcm()