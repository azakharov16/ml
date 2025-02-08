import os
import random
import numpy as np
import pandas as pd
import tables as tb
import scipy.stats as ss
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from tqdm import tqdm
from utils import hist_huge

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (30, 20)
path = Path(os.path.abspath(__file__)).parent

df = pd.read_hdf(path.joinpath('.hd5'))
Nobs = df.shape[0]
id_ls = df['ID_COL'].values.tolist()
nlevel = 15
f_mat = tb.open_file(path.joinpath('dist_mat.h5'), 'r')
D_mat = f_mat.root.data
t = 0.3

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_mat = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    return dendrogram(linkage_mat, **kwargs)

# Method 1 - predefined number of clusters
nclust = 8
avglnk = AgglomerativeClustering(n_clusters=nclust, affinity='precomputed', compute_full_tree=True, linkage='average')
avglnk = avglnk.fit(D_mat)
fig, ax = plt.subplots()
plot_dendrogram(avglnk, truncate_mode='level', p=nlevel, color_threshold=t, ax=ax)
line = mlines.Line2D([0, 1], [t, t], color='red', ls='solid', lw=2)
tr = ax.transAxes
line.set_transform(tr)
ax.add_line(line)
plt.title(f"Dendrogram of {nclust} clusters with threshold {np.round(t, 2)}")

# Method 2 - automatically find the number of clusters
avglnk = AgglomerativeClustering(
    n_clusters=None, affinity='precomputed', compute_full_tree=True, linkage='average', distance_threshold=t
)
avglnk = avglnk.fit(D_mat)
fig, ax = plt.subplots()
plot_dendrogram(avglnk, truncate_mode='level', p=nlevel, color_threshold=t, ax=ax)
line = mlines.Line2D([0, 1], [t, t], color='red', ls='solid', lw=2)
tr = ax.transAxes
line.set_transform(tr)
ax.add_line(line)
plt.title(f"Dendrogram of {avglnk.n_clusters_} clusters with threshold {np.round(t, 2)}")
f_mat.close()

# Method 3 - use distance vector instead of matrix
f_vec = tb.open_file(path.joinpath('dist_vec.h5'), 'r')
D_vec = f_vec.root.data
Z = linkage(D_vec, method='average')
dendrogram(Z, truncate_mode='level', p=nlevel, color_threshold=t, ax=ax)
cluster_labels = fcluster(Z, t, criterion='distance')
f_vec.close()

# Bootstrapping a huge dendrogram
Nsim = 100000
threshold_grid = np.arange(0.05, 0.45, 0.05)
nclust_arr = np.zeros((Nsim, len(threshold_grid)), dtype=int)
chunk_size = 1000
clust_arr_dict = {}
for ind_t in range(len(threshold_grid)):
    clust_arr_dict[ind_t] = np.zeros((Nobs, Nsim), dtype=int)
    # TODO: use scipy.sparse matrix class here
for n in tqdm(range(Nsim)):
    random.seed(2 * n)
    # TODO: replace with routines from np.random module
    id_chunk = random.sample(id_ls, chunk_size)
    id_idx = [id_ls.index(x) for x in id_chunk]
    D_small = np.zeros((chunk_size, chunk_size))
    for i, ind_i in enumerate(id_idx):
        for j, ind_j in enumerate(id_idx):
            D_small[i, j] = D_mat[ind_i, ind_j]
    Z = linkage(D_small[np.triu_indices(chunk_size, k=1)].flatten(), 'average')
    for ind_t, t in enumerate(threshold_grid):
        if n % (Nsim / 50) == 0:
            fig, ax = plt.subplots()
            dendrogram(
                Z, truncate_mode='level', p=nlevel, color_threshold=t, ax=ax,
                leaf_rotation=90, leaf_font_size=8
            )
            line = mlines.Line2D([0, 1], [t, t], color='red', ls='solid', lw=2)
            tr = ax.transAxes
            line.set_transform(tr)
            ax.add_line(line)
            plt.title(f"Dendrogram for step {n + 1} with threshold {np.round(t, 2)}")
            plt.close(fig)
        cluster_labels = fcluster(Z, t, criterion='distance')
        nclust = len(np.unique(cluster_labels))
        nclust_arr[n, ind_t] = nclust
        clust_arr_dict[ind_t][id_idx, n] = cluster_labels

# Bootstrapped clusters histogram
nbins = 35
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.grid(False)
cmap = plt.get_cmap('tab10')
ind_x = np.arange(1, np.max(nclust_arr) + 1)
for ind_t in range(len(threshold_grid)):
    col = cmap.colors[ind_t]
    counts, bins, _ = ax1.hist(
        nclust_arr[:, ind_t], bins=nbins, density=False, alpha=0.5, color=col,
        label=f"Threshold={threshold_grid[ind_t]}"
    )
    sample_pdf = ss.gaussian_kde(nclust_arr[:, ind_t])
    ax2.plot(bins, sample_pdf(bins), color=col, lw=2)
ax1.set_xlabel("Clusters")
ax1.set_ylabel("Number of scenarios")
ax1.set_xticks(ind_x)
ax2.set_ylabel("Prob")
ax2.set_xticks(ind_x)
ax2.set_ylim(0.0, 1.0)
ax1.set_title(f"Number of clusters for different thresholds for {Nsim} simulations")
ax1.legend()
fig.show()
fig.savefig(path.joinpath('.png'))
plt.close(fig)

# Here some value of threshold t is selected
f = tb.open_file(path.joinpath('coph_dist_vec.h5'), 'r')
C_vec = f.root.data
dist_unique, dist_counts = hist_huge(C_vec)
dist_props = dist_counts / np.sum(dist_counts)
dist_fin = np.unique(np.round(dist_unique, 2))
props_fin = np.zeros(dist_fin.size)
for i in range(dist_fin.size):
    elem = dist_fin[i]
    ind = np.where(np.round(dist_unique, 2) == elem)
    props_fin[i] = np.sum(dist_props[ind])
fig, ax = plt.subplots()
ax.bar(np.arange(dist_fin.size), props_fin, align='edge')
ax.set_xticks(np.arange(dist_fin.size))
ax.set_xticklabels(dist_fin.astype(str))
ax.set_xlabel("Empirical similarity")
ax.set_ylabel("Frequency")
ax.set_title("Empirical probabilities of a pair falling into the same cluster")
fig.show()
fig.savefig(path.joinpath('hist_similarities_full.png'))
plt.close(fig)
print(np.min(dist_unique[np.where(np.cumsum(dist_props) >= 0.99)]))

# Distribution of clusters for a given threshold
t_grid = np.arange(0.1, 0.205, 0.005)
nclust_arr = []
for t in t_grid:
    cluster_labels = np.zeros(Nobs, dtype=int)
    prop_threshold = t
    pos = 0
    incr = Nobs - 1
    for i in tqdm(range(Nobs)):
        if i == 0:
            label = 1
            cluster_labels[i] = label
        else:
            label = cluster_labels[i]
        flag_arr = C_vec[pos:(pos + incr), 0].flatten() >= prop_threshold
        if label != 0:
            cluster_labels[(i + 1):][flag_arr] = label
        else:
            label_arr = cluster_labels[(i + 1):][flag_arr]
            if label_arr.nonzero()[0].size > 0:
                max_prop_pos = np.argmax(C_vec[pos:(pos + incr), 0].flatten()[np.where(label_arr != 0)])
                new_label = label_arr[np.where(label_arr != 0)].item(max_prop_pos)
            else:
                new_label = np.max(cluster_labels) + 1
            cluster_labels[i] = new_label
            cluster_labels[(i + 1):][flag_arr] = new_label
        pos += incr
        incr -= 1
    nclust = np.unique(cluster_labels).size
    nclust_arr.append(nclust)
    vals, counts = np.unique(cluster_labels, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(np.arange(counts.size), counts, align='edge')
    if nclust > 200:
        step = 10
    elif nclust > 50:
        step = 5
    else:
        step = 1
    t = np.round(t, 3)
    ax.set_xticks(np.arange(0, vals.size, step))
    ax.set_xticklabels(vals[::step].astype(str))
    ax.set_xlabel("Cluster labels")
    ax.set_ylabel("Cluster size")
    ax.set_title(f"Distribution of clusters for threshold {t}")
    fig.show()
    fig.savefig(path.joinpath(f'hist_clusters_t_{t}.png'), bbox_inches='tight')
    plt.close(fig)
f.close()
