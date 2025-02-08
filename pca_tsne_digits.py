from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
eps = 1e-6
points = PCA(2).fit_transform(X / (eps + X.std(0)))
plt.scatter(points[:, 0], points[:, 1], cmap='rainbow', c=y)

points = TSNE(2).fit_transform(X)
plt.scatter(points[:, 0], points[:, 1], cmap='rainbow', c=y)

def scatter_texts(points, texts, labels=None):
    i = 0
    points = np.copy(points) + np.random.randn(len(points), 2) * points.std(0) / 5
    plt.figure(figsize=(12, 8))
    while i < len(X):
        color = 'C0' if labels is None else plt.cm.tab10_r(labels[i] / labels.max())
        plt.text(
            points[i, 0],
            points[i, 1],
            texts[i],
            c=color
        )
        i += 1
    plt.xlim(points[:, 0].min(), points[:, 0].max())
    plt.ylim(points[:, 1].min(), points[:, 1].max())
