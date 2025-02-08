import numpy as np
from matplotlib import pyplot as plt
data = np.array([
        [1, 4, 5, 5, 12],
        [3, 6, 2, 2, 5],
        [2, 10, 12, 0, 2],
        [3, 9, 6, 2, 10],
        [2, 5, 0, 6, 25],
        [0, 8, 3, 3, 14],
        [1, 8, 9, 2, 12],
        [1, 6, 2, 3, 23],
        [2, 8, 10, 7, 26],
        [0, 5, 7, 2, 16]
])
data_centered = data - np.mean(data, axis=0)
cov_mat = np.cov(data_centered, rowvar=False)
eigenval, eigenvec = np.linalg.eigh(cov_mat)
idx = np.argsort(eigenval)[::-1]
sorted_eigenval = eigenval[idx]
sorted_eigenvec = eigenvec[:, idx]
n_components = 2
eigenvec_subset = sorted_eigenvec[:, 0:n_components]
data_pca = np.matmul(data_centered, eigenvec_subset)
plt.plot(range(1, data.shape[1] + 1), eigenval / np.sum(eigenval))
plt.show()
