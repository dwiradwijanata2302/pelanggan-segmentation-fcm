import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_results():
    df = pd.read_csv('hasil_akhir.csv')

    # Grafik Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Pendapatan_Tahunan_Ribuan_USD', y='Pengeluaran_USD', 
                    hue='Cluster', palette='Set1', s=100)
    plt.title('Hasil Segmentasi Pelanggan (FCM)')
    plt.show()

    # Tabel Rata-rata
    print("\nProfil Karakteristik Tiap Kelompok:")
    profil_cluster = df.groupby('Cluster')[['Usia', 'Pendapatan_Tahunan_Ribuan_USD', 'Pengeluaran_USD']].mean()
    profil_cluster['Usia'] = profil_cluster['Usia'].round(0)
    print(profil_cluster)

if __name__ == "__main__":
    show_results()