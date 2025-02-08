import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from pathlib import Path

plt.style.use('seaborn')
path = Path(os.path.abspath(__file__)).parent

df = pd.read_hdf(path.joinpath('.hd5'))
Nobs = df.shape[0]
num_factors = []
sc = StandardScaler()
X_num = sc.fit_transform(df[num_factors])

# PCA screeplot
pca = PCA(n_components=len(num_factors)).fit(X_num)
plt.figure(figsize=(15, 10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='black', lw=2)
plt.xlabel("Number of components")
plt.ylabel("Total explained variance")
plt.yticks(np.arange(0.0, 1.1, 0.05))
plt.xticks(range(0, len(num_factors), 1))
plt.axhline(0.9, color='orange')
plt.axhline(0.95, color='red')
plt.title("PCA cumulative explained variance")
plt.close()

# PCA cluster diagnostics
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_num)
X_df = pd.DataFrame(data=X_num, columns=num_factors)
X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
labels = X_df['CLUSTER_LABEL'].values
nclust = X_df['CLUSTER_LABEL'].unique().tolist()
plt.figure(figsize=(24, 20))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=labels, edgecolors='none',
    alpha=0.7, s=40, cmap=plt.cm.get_cmap('nipy_spectral', nclust)
)
plt.colorbar()
plt.title("PCA projection")
plt.savefig(path.joinpath('.png'))
plt.close()
# TSNE cluster diagnostics
tsne = TSNE(n_components=2, random_state=17)
X_tsne = tsne.fit_transform(X_num)
plt.figure(figsize=(24, 20))
plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1], c=labels, edgecolors='none',
    alpha=0.7, s=40, cmap=plt.cm.get_cmap('nipy_spectral', nclust)
)
plt.colorbar()
plt.title("TSNE projection")
plt.savefig(path.joinpath('.png'))
plt.close()
# Plotting cluster centroids in space with reduced dimension
plt.figure(figsize=(24, 20))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color='blue', market='.', edgecolors='none', alpha=0.7, s=40)
centroids = np.zeros(shape=(nclust, 2))
i = 0
for lab in labels.unique():
    X_k = X_tsne[np.array(X_df['CLUSTER_LABEL'] == lab, dtype=bool), :]
    centroid_k = X_k.mean(axis=0)
    centroids[i, :] = centroid_k
    i += 1
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='white', alpha=1, s=200, edgecolors='black')
for lab, c in zip(labels.unique().tolist(), list(centroids)):
    plt.scatter(c[0], c[1], marker='$%d$' % lab, alpha=1, s=50, color='black', edgecolors='black')
plt.title("Cluster centers")
plt.savefig(path.joinpath('.png'))
plt.close()
