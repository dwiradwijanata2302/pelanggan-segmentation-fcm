import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, 'hasil_akhir.csv')
ITERATION_PATH = os.path.join(BASE_DIR, 'riwayat_iterasi.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')
BATAS_ERROR = 0.005


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def save_figure(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Grafik disimpan ke: {path}')


def plot_cluster_scatter(df):
    plt.figure(figsize=(11, 7))
    ax = sns.scatterplot(
        data=df,
        x='Pendapatan_Tahunan_Ribuan_USD',
        y='Pengeluaran_USD',
        hue='Cluster',
        palette='Set2',
        s=120,
        edgecolor='white',
        linewidth=0.6,
    )
    ax.set_title('Sebaran Pelanggan per Cluster', fontsize=15, weight='bold')
    ax.set_xlabel('Pendapatan Tahunan (normalized)', fontsize=11)
    ax.set_ylabel('Pengeluaran (normalized)', fontsize=11)
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    save_figure('01_scatter_cluster.png')
    plt.show()


def plot_cluster_profile_heatmap(df):
    profil_cluster = df.groupby('Cluster')[['Usia', 'Pendapatan_Tahunan_Ribuan_USD', 'Pengeluaran_USD']].mean()
    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(
        profil_cluster,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        linewidths=0.5,
        cbar_kws={'label': 'Nilai rata-rata'},
    )
    ax.set_title('Profil Rata-rata Tiap Cluster', fontsize=15, weight='bold')
    ax.set_xlabel('Fitur', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    save_figure('02_heatmap_profil_cluster.png')
    plt.show()


def plot_cluster_size(df):
    cluster_counts = df['Cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=cluster_counts.index.astype(str), y=cluster_counts.values, palette='Set3')
    ax.set_title('Jumlah Data pada Setiap Cluster', fontsize=15, weight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Jumlah Sampel', fontsize=11)
    for index, value in enumerate(cluster_counts.values):
        ax.text(index, value + 0.2, str(value), ha='center', va='bottom', fontsize=10)
    save_figure('03_jumlah_cluster.png')
    plt.show()


def plot_convergence(riwayat):
    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.plot(riwayat['Iterasi'], riwayat['Perubahan_Membership'], marker='o', linewidth=2, color='#2a6fdb', label='Perubahan membership')
    ax.axhline(y=BATAS_ERROR, color='#c1121f', linestyle='--', linewidth=2, label='Batas error')
    ax.set_title('Konvergensi FCM per Iterasi', fontsize=15, weight='bold')
    ax.set_xlabel('Iterasi', fontsize=11)
    ax.set_ylabel('Perubahan membership', fontsize=11)
    ax.legend()
    save_figure('04_konvergensi_iterasi.png')
    plt.show()

def show_results():
    ensure_figure_dir()
    df = pd.read_csv(RESULT_PATH)
    sns.set_theme(style='whitegrid', context='talk')

    print("\nProfil Karakteristik Tiap Kelompok:")
    profil_cluster = df.groupby('Cluster')[['Usia', 'Pendapatan_Tahunan_Ribuan_USD', 'Pengeluaran_USD']].mean()
    profil_cluster['Usia'] = profil_cluster['Usia'].round(0)
    print(profil_cluster)

    plot_cluster_scatter(df)
    plot_cluster_profile_heatmap(df)
    plot_cluster_size(df)

    if os.path.exists(ITERATION_PATH):
        riwayat = pd.read_csv(ITERATION_PATH)
        plot_convergence(riwayat)

        print("\nPenjelasan berhenti iterasi:")
        if riwayat['Perubahan_Membership'].iloc[-1] < BATAS_ERROR:
            print(f"Iterasi berhenti karena perubahan membership terakhir {riwayat['Perubahan_Membership'].iloc[-1]:.6f} sudah lebih kecil dari batas error {BATAS_ERROR:.3f}.")
        else:
            print("Iterasi berhenti karena mencapai batas iterasi maksimum.")

if __name__ == "__main__":
    show_results()