import pandas as pd
import re
from test import WordCloud
import matplotlib.pyplot as plt
from io import StringIO

try:
    df = pd.read_csv("dataset.csv", delimiter=';')
except FileNotFoundError:
    print("File 'dataset.csv' tidak ditemukan. Menggunakan representasi data yang diunggah.")
    pass

df['Content'] = df['Content'].astype(str).fillna('')

text = ' '.join(df['Content'])

text = text.lower()

# Menghilangkan tanda baca dan angka
text = re.sub(r'[^a-z\s]', '', text)

# Daftar stopwords umum Bahasa Indonesia
stopwords_indonesia = set([
    "yang", "dan", "di", "ke", "dari", "untuk", "itu", "ini", "dengan", "adalah",
    "tidak", "bisa", "akan", "tapi", "juga", "lagi", "seperti", "ada", "saya", "kita",
    "mereka", "pun", "nya", "saja", "udah", "udah", "pak", "ayo", "jgn", "smpe", "dia",
    "d", "pe" 
])

# Menghilangkan kata yng sngat pendek (tunggal)
words = text.split()
filtered_words = [word for word in words if word not in stopwords_indonesia and len(word) > 2]
clean_text = ' '.join(filtered_words)

# Membuat objek WordCloud
test = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords_indonesia,
    min_font_size=10,
    colormap='viridis'
).generate(clean_text)

# Menampilkan hasil Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(test, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud dari Komentar Timnas Indonesia")
plt.show()

print("\nProses Text Mining (Pembersihan Teks dan Word Cloud) Selesai.")
print("Kata-kata yang paling sering muncul (dengan font lebih besar) telah divisualisasikan.")