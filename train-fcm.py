import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, 'data_siap_fcm.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'hasil_akhir.csv')
DETAIL_PATH = os.path.join(BASE_DIR, 'uji_25_sampel.csv')
ITERATION_PATH = os.path.join(BASE_DIR, 'riwayat_iterasi.csv')

JUMLAH_CLUSTER = 5
JUMLAH_SAMPLE_UJI = 25
FUZZINESS = 2.0
BATAS_ERROR = 0.005
MAKS_ITERASI = 1000


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


def calculate_objective(data, membership, centroids, fuzziness):
    objective = 0.0
    for cluster_index, centroid in enumerate(centroids):
        distances = np.linalg.norm(data - centroid, axis=1)
        objective += np.sum((membership[:, cluster_index] ** fuzziness) * (distances ** 2))
    return objective


def manual_fcm(data, n_clusters=5, fuzziness=2.0, error=0.005, maxiter=1000, seed=42):
    data = np.asarray(data, dtype=float)
    n_samples = data.shape[0]
    membership = initialize_membership(n_samples, n_clusters, seed=seed)
    history = []
    stop_reason = 'maksimum iterasi tercapai'

    for iteration in range(maxiter):
        old_membership = membership.copy()
        centroids = update_centroids(data, membership, fuzziness)
        membership = update_membership(data, centroids, fuzziness)
        membership_change = np.linalg.norm(membership - old_membership)
        objective = calculate_objective(data, membership, centroids, fuzziness)

        history.append({
            'Iterasi': iteration + 1,
            'Perubahan_Membership': membership_change,
            'Objective_FCM': objective,
        })

        if membership_change < error:
            stop_reason = f'konvergensi tercapai karena perubahan membership {membership_change:.6f} < batas error {error}'
            print(f"Konvergensi tercapai pada iterasi ke-{iteration + 1}")
            break

    return membership, centroids, pd.DataFrame(history), stop_reason


def build_ringkas_report(df_data, membership):
    ringkas = df_data.copy().reset_index(drop=True)
    ringkas['Cluster'] = np.argmax(membership, axis=1)
    ringkas['Nilai_Membership_Tertinggi'] = membership.max(axis=1)
    return ringkas


def build_detail_report(df_data, membership):
    detail = df_data.copy().reset_index(drop=True)
    detail['Cluster'] = np.argmax(membership, axis=1)
    detail['Nilai_Membership_Tertinggi'] = membership.max(axis=1)

    for cluster_index in range(membership.shape[1]):
        detail[f'Membership_K{cluster_index + 1}'] = membership[:, cluster_index]

    return detail


def run_fcm():
    print('Membaca data bersih...')
    df_clean = pd.read_csv(CLEAN_PATH).astype(float)
    df_uji = df_clean.head(JUMLAH_SAMPLE_UJI).copy()
    data_array = df_uji.values

    print(f'Pakai subset {len(df_uji)} baris untuk uji manual per satu.')
    print(f'Jumlah cluster yang dipakai: {JUMLAH_CLUSTER}')

    print('Menjalankan FCM manual dari nol...')
    membership, centroids, history, stop_reason = manual_fcm(
        data_array,
        n_clusters=JUMLAH_CLUSTER,
        fuzziness=FUZZINESS,
        error=BATAS_ERROR,
        maxiter=MAKS_ITERASI,
    )

    df_ringkas = build_ringkas_report(df_uji, membership)
    df_detail = build_detail_report(df_uji, membership)

    df_ringkas.to_csv(OUTPUT_PATH, index=False)
    df_detail.to_csv(DETAIL_PATH, index=False)
    history.to_csv(ITERATION_PATH, index=False)

    cluster_membership = df_ringkas['Cluster'].to_numpy()

    print(f'SUKSES! Hasil ringkas disimpan ke: {OUTPUT_PATH}')
    print(f'Detail uji 25 sampel disimpan ke: {DETAIL_PATH}')
    print(f'Riwayat iterasi disimpan ke: {ITERATION_PATH}')
    print(f'Alasan berhenti: {stop_reason}')
    print('Centroid akhir:')
    print(centroids)
    print('Distribusi cluster pada subset uji:')
    print(pd.Series(cluster_membership).value_counts().sort_index())


if __name__ == '__main__':
    run_fcm()
