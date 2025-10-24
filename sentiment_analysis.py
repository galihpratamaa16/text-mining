import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Kamus slang
slang_dict = {
    'smgt': 'semangat', 'bgt': 'banget', 'gpp': 'tidak apa apa', 'tdk': 'tidak',
    'udh': 'sudah', 'jg': 'juga', 'yg': 'yang', 'd': 'di', 'kl': 'kalau',
    'pk': 'pak', 'kluivert out': 'kluivert keluar', 'min': 'admin',
    'aj': 'saja', 'jgn': 'jangan', 'ni': 'ini', 'tp': 'tapi', 'ku': 'aku',
    'pki': 'partai komunis indonesia', 'sy': 'saya', 'mksih': 'terima kasih',
    'cmn': 'cuma', 'dr': 'dari', 'bnyk': 'banyak', 'gak': 'tidak', 'ga': 'tidak',
    'udh': 'sudah', 'tpi': 'tapi', 'blm': 'belum', 'bkn': 'bukan', 'krn': 'karena'
}

# Kamus Sentimen Sederhana Bahasa Indonesia (Lexicon)
# Skor, Positif = +1, Negatif = -1
lexicon = defaultdict(int)

# Kata-kata Positif
pos_words = ['dukung', 'semangat', 'keren', 'bagus', 'setuju', 'terbaik', 'senang', 
             'mantap', 'top', 'bangga', 'berhasil', 'luar biasa', 'cocok', 'juara', 
             'respect', 'sukses', 'terimakasih', 'amin', 'betul', 'wajib', 'terbaik']
for word in pos_words:
    lexicon[word] = 1

# Kata-kata Negatif
neg_words = ['pecat', 'keluar', 'out', 'gagal', 'parah', 'buruk', 'kecewa', 'rugi', 
             'malu', 'bodoh', 'salah', 'tidak setuju', 'bohong', 'rusak', 'menghina',
             'tidak', 'kurang', 'sial', 'gila', 'miris', 'jahat', 'hancur', 'tolol',
             'taruhan', 'botak', 'kesel', 'usir', 'judol', 'mundur']
for word in neg_words:
    lexicon[word] = -1

# PREPROCESSING
def clean_and_normalize(text):
    """Membersihkan teks dan menormalisasi kata slang."""
    
    # 1. Case Folding: Ubah ke huruf kecil
    text = text.lower()
    
    # 2. Pembersihan Karakter: Hapus URL, Username (@), Hashtag (#), Angka, dan Karakter Khusus
    text = re.sub(r'http\S+|www.\S+', '', text) # Menghapus URL
    text = re.sub(r'@\w+', '', text)            # Menghapus Username TikTok
    text = re.sub(r'#\w+', '', text)            # Menghapus Hashtag
    text = re.sub(r'[^\w\s]', '', text)         # Menghapus Tanda Baca/Karakter Khusus
    text = re.sub(r'\d+', '', text)             # Menghapus Angka
    
    # 3. Normalisasi Slang
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    
    # 4. Menghilangkan Spasi Berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment(text, lexicon):
    """Menghitung skor sentimen berdasarkan kamus (lexicon)."""
    words = text.split()
    score = 0
    
    # Menghitung skor
    for word in words:
        score += lexicon[word]
        
    # Mengklasifikasikan sentimen
    if score > 0:
        return 'Positif'
    elif score < 0:
        return 'Negatif'
    else:
        return 'Netral'

def main():
    print("--- PROGRAM ANALISIS SENTIMEN KOMENTAR TIKTOK ---")
    
    # Pemuatan Data
    try:
        # Mencoba memuat file dengan delimiter ';' dan encoding utf-8
        df = pd.read_csv("dataset.csv", delimiter=';', encoding='utf-8')
    except Exception:
        # Jika gagal, coba encoding lain (seringkali latin1)
        df = pd.read_csv("dataset.csv", delimiter=';', encoding='latin1')
    
    # Menghapus baris dengan nilai 'Content' kosong (NaN)
    df_clean = df.dropna(subset=['Content']).copy()
    data_komentar = df_clean['Content']
    
    print(f"Total Komentar yang diproses: {len(data_komentar)}")
    print("\n[STEP 1/3] Melakukan Pra-Pemrosesan (Pembersihan & Normalisasi Slang)...")
    
    # --- Pra-Pemrosesan ---
    df_clean['Content_Clean'] = data_komentar.apply(clean_and_normalize)
    
    # --- Analisis Sentimen ---
    print("[STEP 2/3] Melakukan Klasifikasi Sentimen Berbasis Kamus...")
    df_clean['Sentiment'] = df_clean['Content_Clean'].apply(lambda x: get_sentiment(x, lexicon))
    
    print("\n[STEP 3/3] Menampilkan Hasil dan Visualisasi...")
    
    # --- Hasil Numerik ---
    sentiment_counts = df_clean['Sentiment'].value_counts()
    sentiment_percent = df_clean['Sentiment'].value_counts(normalize=True) * 100

    print("\n===============================================")
    print("HASIL ANALISIS SENTIMEN KESELURUHAN KOMENTAR")
    print("===============================================")
    
    results = pd.DataFrame({
        'Jumlah': sentiment_counts,
        'Persentase': sentiment_percent.round(2)
    })
    results = results.sort_values(by='Jumlah', ascending=False)
    print(results.to_string())

    # --- Sampel Hasil ---
    print("\n--- 3 Sampel Komentar Positif ---")
    print(df_clean[df_clean['Sentiment'] == 'Positif'][['Content', 'Sentiment']].head(3).to_string())

    print("\n--- 3 Sampel Komentar Negatif ---")
    print(df_clean[df_clean['Sentiment'] == 'Negatif'][['Content', 'Sentiment']].head(3).to_string())

    # --- Visualisasi Hasil ---
    plt.figure(figsize=(8, 6))
    sns.barplot(x=results.index, y=results['Persentase'], 
                palette={'Positif': 'green', 'Negatif': 'red', 'Netral': 'gray'})
    
    plt.title('Distribusi Sentimen Komentar TikTok', fontsize=14)
    plt.ylabel('Persentase Komentar (%)', fontsize=12)
    plt.xlabel('Sentimen', fontsize=12)
    
    # Menambahkan label persentase di atas bar
    for index, row in results.iterrows():
        plt.text(row.name, row['Persentase'] + 0.5, f"{row['Persentase']:.2f}%", 
                 color='black', ha="center", fontsize=10)
        
    plt.ylim(0, results['Persentase'].max() * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()