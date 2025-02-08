import numpy as np
import matplotlib.pyplot as plt

# Загружаем матрицу встречаемости слов.
# allow_pickle=True сообщает NumPy, что файл сохранён в специальном формате pickle.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_term_occurence.npy"
filename = np.DataSource().open(url).name
X = np.load(filename, allow_pickle=True)

# Загружаем слова, соответствующие строкам матрицы.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_words.npy"
filename = np.DataSource().open(url).name
words = np.load(filename, allow_pickle=True)

print("(количество уникальных слов, количество документов):", X.shape)

print("Первые 10 слов:", words[:10])

# Визуализация матрицы
plt.figure(figsize=(6, 12))
plt.spy(X)
plt.xlabel("Документы", fontsize=20, labelpad=15)
plt.ylabel("Слова", rotation=0, fontsize=20, labelpad=25)
plt.xticks([0, 125, 250], rotation=90)

