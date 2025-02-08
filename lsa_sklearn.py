from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

preprocessed_texts = np.array(['игра крутой большой онлайн',
                                 'огурец отстой не вкусный',
                                 'огурчик супер',
                                 'обычный огурчик понравиться соленый',
                                 'игра ужасный геймплей донат',
                                 'лопата шикарный взять на дача',
                                 'вкусный как на дача',
                                 'дрель мощь просверливать весь стена',
                                 'огурец понравиться свежий',
                                 'вкусный немного горьковатый'])

# Строим матрицу встречаемости слов.
count_model = CountVectorizer()
X = count_model.fit_transform(preprocessed_texts).toarray().astype(float).T
words = count_model.get_feature_names()

# Функция возвращает индекс ближайшего к `search_vector` вектора из строк `vectors`.
def find_nearest_vector_id(search_vector, vectors):
		# L2 расстояние между векторами.
    distances = np.mean((search_vector - vectors) ** 2, axis=1)
		# Индексы векторов с наименьшим расстоянием.
    return np.argmin(distances)

# По заданной таблице встречаемости постройте матрицу эмбеддингов слов и матрицу эмбеддингов
# документов. Выведите документ, который ближе всего к слову вкусный, которое соответствует
# 4-й строке матрицы встречаемости.
