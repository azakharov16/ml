import numpy as np

# Загружаем матрицу встречаемости слов.
# allow_pickle=True сообщает NumPy, что файл сохранён в специальном формате pickle.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_term_occurence.npy"
filename = np.DataSource().open(url).name
X = np.load(filename, allow_pickle=True)

# Загружаем слова, соответствующие строкам матрицы.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_words.npy"
filename = np.DataSource().open(url).name
words = np.load(filename, allow_pickle=True)

# Загружаем оригинальные описания фильмов.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_summaries.npy"
filename = np.DataSource().open(url).name
summaries = np.load(filename, allow_pickle=True)


U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Количество используемых сингулярных векторов\топиков.
k = 10

# Считаем эмбеддинги слов с помощью матрицы U и s.
word_vectors = (U @ np.diag(s))[:, :k]

# Считаем эмбеддинги документов с помощью матрицы V и s.
doc_vectors = (np.diag(s) @ Vt)[:k].T # Транспонируем, чтобы объединить с пространством слов.

# Новый документ по словам. Поддерживаются только слова из `words`.
new_doc = ["бильбо", "кольцо", "хоббит"]

# Строим вектор нового документа как сумму векторов его слов.
# Сразу создаём вектор-строку, чтобы потом не транспонировать.
search_vector = np.zeros((1, k))
i = 0
while i < len(new_doc):
	# Находим вектор слова.
    search_index = np.where(words == new_doc[i])[0]
	# Прибавляем вектор слова к вектору нового документа.
    search_vector += word_vectors[search_index]
    i += 1

# Усредняем векторы слов в документе.
search_vector /= len(new_doc)

# Ищем ближайшие векторы документов и возвращаем соответствующие тексты.
def find_nearest_doc(doc_vector, n=3):
	# L2 расстояние между векторами.
    distances = np.mean((doc_vector - doc_vectors) ** 2, axis=1)
	# Индексы векторов с наименьшим расстоянием.
    return summaries[np.argsort(distances)[1:n]]

print(find_nearest_doc(search_vector, n=10))
