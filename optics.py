import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.manifold import TSNE
from matplotlib import gridspec
from matplotlib import pyplot as plt

path = Path(os.path.abspath(__file__)).parent
df = pd.read_hdf(path.joinpath('.hd5'))
factors = []
sc = StandardScaler()
X = sc.fit_transform(df[factors])
min_size = 100
min_prop = min_size / X.shape[0]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

def get_reachability_plot(data, model, eps_grid):
    space = np.arange(data.shape[0])
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]
    nclust = np.unique(labels).shape[0] - 1
    npar = len(eps_grid)
    plt.figure(figsize=(25, 15))
    G = gridspec.GridSpec(2, npar + 1)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    for i, eps in enumerate(eps_grid):
        exec(f"ax{i + 3} = plt.subplot(G[1, {i + 1}])")
        labs = cluster_optics_dbscan(
            reachability=model.reachability_,
            core_distances=model.core_distances_,
            ordering=model.ordering_, eps=eps
        )
        exec(f"labels_{i + 1} = labs.copy()")
    colormap = plt.cm.get_cmap('gist_rainbow', nclust)
    for lab in np.unique(labels[labels != -1]):
        xk = space[labels == lab]
        R_k = reachability[labels == lab]
        ax1.scatter(xk, R_k, c=np.full_like(xk, lab), cmap=colormap, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], color='black', alpha=0.3)
    for eps in eps_grid:
        ax1.plot(space, np.full_like(space, eps), 'k-', alpha=0.5)
    ax1.set_ylabel("Reachability")
    ax1.set_title("Reachability plot")
    for lab in np.unique(labels[labels != -1]):
        X_k = X_tsne[model.labels_ == lab, :]
        ax2.scatter(X_k[:, 0], X_k[:, 1], c=np.full_like(X_k[:, 0], lab), cmap=colormap, alpha=0.3)
    ax2.plot(X_tsne[model.labels_ == -1, 0], X_tsne[model.labels_ == -1, 1], 'k+')
    ax2.set_title("Automatic clustering with OPTICS")
    for i, eps in enumerate(eps_grid):
        exec(f"labs = labels_{i + 1}.copy()")
        for lab in np.unique(labs[labs != -1]):
            X_k = X_tsne[labs == lab, :]
            exec(f"ax{i + 3}.scatter(X_k[:, 0], X_k[:, 1], c=np.full_like(X_k[:, 0], lab), cmap=colormap, alpha=0.3)")
        exec(f"ax{i + 3}.plot(X_tsne[labs == -1, 0], X_tsne[labs == -1, 1], 'k+')")
        title = f"Clustering at {eps} eps cut with DBSCAN"
        exec(f"ax{i + 3}.set_title(title)")
    plt.tight_layout()
    plt.show()

model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=min_prop, metric='euclidean', cluster_method='xi')
model.fit(X)
get_reachability_plot(data=X, model=model, eps_grid=[0.5, 2.0])
