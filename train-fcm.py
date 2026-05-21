import pandas as pd
import numpy as np
import skfuzzy as fuzz
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'Pengunjung_Mall.csv')
CLEAN_PATH = os.path.join(BASE_DIR, 'data_siap_fcm.csv')


def get_clean_data(df):
    df_bersih = df.drop(['ID_Pelanggan', 'Gender'], axis=1)

    Q1 = df_bersih.quantile(0.25)
    Q3 = df_bersih.quantile(0.75)
    IQR = Q3 - Q1

    return df.loc[
        ~((df_bersih < (Q1 - 1.5 * IQR)) |
          (df_bersih > (Q3 + 1.5 * IQR))).any(axis=1)
    ].copy()

def run_fcm():
    # Load data yang sudah bersih
    df_norm = pd.read_csv(CLEAN_PATH)
    data = df_norm.values.T

    # Jalankan FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=5, m=2, error=0.005, maxiter=1000
    )

    # Ambil hasil cluster
    cluster_membership = np.argmax(u, axis=0)

    # Masukkan hasil ke baris yang sama dengan data yang sudah lolos pembersihan
    df = pd.read_csv(INPUT_PATH)
    df = get_clean_data(df)

    if len(df) != len(cluster_membership):
        raise ValueError(
            f"Jumlah baris data bersih ({len(df)}) tidak sama dengan jumlah cluster ({len(cluster_membership)})"
        )

    df['Cluster'] = cluster_membership
    output_path = os.path.join(BASE_DIR, 'hasil_akhir.csv')
    df.to_csv(output_path, index=False)
    print(f"Proses FCM selesai! Hasil disimpan ke '{output_path}'")

if __name__ == "__main__":
    run_fcm()