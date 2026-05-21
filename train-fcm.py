import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, 'data_siap_fcm.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'hasil_akhir.csv')


def initialize_membership(n_samples, n_clusters, seed=42):
    rng = np.random.default_rng(seed)
    membership = rng.random((n_samples, n_clusters))
    membership = membership / membership.sum(axis=1, keepdims=True)
    return membership


def update_centroids(data, membership, fuzziness):
    membership_power = membership ** fuzziness
    numerator = membership_power.T @ data
    denominator = membership_power.sum(axis=0)[:, None]
    return numerator / denominator


def update_membership(data, centroids, fuzziness):
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    distances = np.zeros((n_samples, n_clusters))

    for cluster_index in range(n_clusters):
        distances[:, cluster_index] = np.linalg.norm(data - centroids[cluster_index], axis=1)

    zero_distance_rows = np.any(distances == 0, axis=1)
    if np.any(zero_distance_rows):
        membership = np.zeros_like(distances)
        zero_rows = np.where(zero_distance_rows)[0]
        for row_index in zero_rows:
            cluster_index = np.where(distances[row_index] == 0)[0][0]
            membership[row_index, cluster_index] = 1.0

        non_zero_rows = ~zero_distance_rows
        if np.any(non_zero_rows):
            non_zero_distances = np.fmax(distances[non_zero_rows], 1e-10)
            exponent = -2.0 / (fuzziness - 1.0)
            temp = non_zero_distances ** exponent
            membership[non_zero_rows] = temp / temp.sum(axis=1, keepdims=True)
        return membership

    distances = np.fmax(distances, 1e-10)
    exponent = -2.0 / (fuzziness - 1.0)
    temp = distances ** exponent
    return temp / temp.sum(axis=1, keepdims=True)


def manual_fcm(data, n_clusters=5, fuzziness=2.0, error=0.005, maxiter=1000, seed=42):
    data = np.asarray(data, dtype=float)
    n_samples = data.shape[0]
    membership = initialize_membership(n_samples, n_clusters, seed=seed)

    for iteration in range(maxiter):
        old_membership = membership.copy()
        centroids = update_centroids(data, membership, fuzziness)
        membership = update_membership(data, centroids, fuzziness)

        if np.linalg.norm(membership - old_membership) < error:
            print(f"Konvergensi tercapai pada iterasi ke-{iteration + 1}")
            break

    return membership, centroids


def run_fcm():
    print('Membaca data bersih...')
    df_clean = pd.read_csv(CLEAN_PATH).astype(float)
    data_array = df_clean.values

    jumlah_cluster = 5
    fuzziness = 2.0
    batas_error = 0.005
    maks_iterasi = 1000

    print('Menjalankan FCM manual dari nol...')
    membership, centroids = manual_fcm(
        data_array,
        n_clusters=jumlah_cluster,
        fuzziness=fuzziness,
        error=batas_error,
        maxiter=maks_iterasi,
    )

    cluster_membership = np.argmax(membership, axis=1)

    df_hasil = df_clean.copy()
    df_hasil['Cluster'] = cluster_membership
    df_hasil.to_csv(OUTPUT_PATH, index=False)

    print(f'SUKSES! Hasil disimpan ke: {OUTPUT_PATH}')
    print('Centroid akhir:')
    print(centroids)


if __name__ == '__main__':
    run_fcm()
