import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# Подготовка данных и модели
X, y = load_digits(return_X_y=True)
X = (X - X.mean()) / X.std()
model = LogisticRegression(max_iter=1000)
model.fit(X[:-1], y[:-1])
b = model.intercept_

# Матрица весов модели, её нужно сжимать
W = model.coef_

# Номер картинки в датасете для построения графика. Должен быть меньше 1797.
plot_image_num = 52


# Ваш код для сжатия W


# Код для отрисовки графиков и подсчёта качества работы модели.

def softmax(x):
    # Вспомогательная функция для расчёта вероятностей.
    s = np.exp(x - x.min())
    return s / s.sum()


def make_prediction(W, b, x):
    # Расчёт предсказанных по картинке вероятностей.
    scores = W @ x + b
    return softmax(scores)


def print_compression_rate(W, U, s, VT):
    # Выводит степень сжатия матрицы.
    before = np.prod(W.shape)
    after = np.prod(U.shape) + np.prod(s.shape) + np.prod(VT.shape)
    print(f"Степень сжатия: {(before - after) / before * 100:.3f}%, с {before} чисел до {after}")
    print()


def plot_predicted_score(W, b, x, y):
    # Визуально показывает предсказанные вероятности.
    plt.figure(figsize=(12, 3))
    p = make_prediction(W, b, x)
    plt.subplot(1, 2, 1)
    plt.title(f"Исходная цифра: {y}", pad=20)
    plt.imshow(x.reshape(8, 8), cmap='Greys')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(10), p)
    plt.xticks(range(10))
    plt.title('Предсказанная вероятность')
    plt.ylabel('P', rotation=0, labelpad=10, fontsize=13)
    plt.xlabel('Цифра')
    plt.ylim(0, 1)
    plt.tight_layout()


def print_accuracy(W, b, X, y):
    # Выводит долю верных предсказаний на всём датасете.
    preds = [np.argmax(make_prediction(W, b, x)) for x in X]
    print(f"Модель предсказывает верно в {accuracy_score(preds, y) * 100:.3f}% случаев.")


print("До сжатия")
plot_predicted_score(W, b, X[plot_image_num], y[plot_image_num])
print_accuracy(W, b, X, y)
plt.show()

# print_compression_rate(W, U, s, VT)

# print()
# print("После сжатия")
# plot_predicted_score(W_approx, b, X[plot_image_num], y[plot_image_num])
# print_accuracy(W_approx, b, X, y)
# plt.show()