import os
import numpy as np
import pandas as pd
import tables as tb
import seaborn as sb
from math import factorial
from itertools import combinations, product
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
from matplotlib import pyplot as plt
from matplotlib import cm

plt.style.use('seaborn')
path = Path(os.path.abspath(__file__)).parent

def calc_cd_index(df, factors):
    sc = StandardScaler()
    X = sc.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
    labels = X_df['CLUSTER_LABEL'].values
    X_arr = np.array(X_df.drop('CLUSTER_LABEL', axis=1))
    repr_dict = {}
    for lab in labels:
        X_k = X_arr[labels == lab, :]
        n_k = X_k.shape[0]
        centroid_k = X_k.mean(axis=0)
        n_repr = n_k // 10
        dist_vec = np.zeros(n_k, dtype=float)
        repr_arr = np.zeros((n_repr, X_k.shape[1]))
        for r in range(n_repr):
            range_arr = np.array(list(range(n_k)), dtype=int)
            for i in range_arr:
                dist_vec[i] = distance.euclidean(X_k[i, :], centroid_k)
            x_repr = np.argmax(dist_vec)
            centroid_k = X_k[x_repr, :]
            repr_arr[r, :] = centroid_k
            X_k = np.delete(X_k, x_repr, axis=0)
            n_k -= 1
        repr_dict[lab] = repr_arr
    return repr_dict

def calc_db_index(df, factors):
    sc = StandardScaler()
    X = sc.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
    labels = X_df['CLUSTER_LABEL'].values
    X_df.drop('CLUSTER_LABEL', axis=1, inplace=True)
    return davies_bouldin_score(X_df, labels)

def calc_silhouette_index(df, factors):
    sc = StandardScaler()
    X = sc.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
    labels = X_df['CLUSTER_LABEL'].values
    X_df.drop('CLUSTER_LABEL', axis=1, inplace=True)
    X_arr = np.array(X_df)
    return silhouette_score(X_arr, labels)

def plot_silhouettes(df, factors, avg_score=None, path=Path(os.path.abspath(__file__)).parent):
    """
    :param df: object of type pd.DataFrame
    :param factors: list of column names of numeric factors in df
    :param avg_score: the silhouette score returned by the function calc_silhouette_index
    :param path: path to save the plot to
    :return: None
    """
    if avg_score is None:
        avg_score = calc_silhouette_index(df=df, factors=factors)
    sc = StandardScaler()
    X = sc.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
    labels = X_df['CLUSTER_LABEL'].values
    clusters = X_df['CLUSTER_LABEL'].unique().tolist()
    nclust = X_df['CLUSTER_LABEL'].nunique()
    X_df.drop('CLUSTER_LABEL', axis=1, inplace=True)
    X_arr = np.array(X_df)
    silhouettes = silhouette_samples(X_arr, labels)
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([0, X_arr.shape[0] + (nclust + 1) * 10])
    y_lower = 10
    for j in range (nclust):
        label_k = clusters[j]
        values_k = silhouettes[labels == label_k]
        values_k.sort()
        n_k = values_k.shape[0]
        y_upper = y_lower + n_k
        color = cm.nipy_spectral(float(j) / nclust)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, values_k, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * n_k, str(label_k))
        y_lower = y_upper + 10
    ax.set_title("Silhouette plot")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=avg_score, color='red', linestyle='dashed')
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0.0, 0.1])
    plt.savefig(path.joinpath('silhouette_plot.png'), bbox_inches='tight')
    plt.close()
    return None

def calc_rsq_index(df, factors):
    sc = StandardScaler()
    X = sc.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df['CLUSTER_LABEL'] = df['CLUSTER_LABEL'].copy()
    labels = X_df['CLUSTER_LABEL'].values
    clusters = X_df['CLUSTER_LABEL'].unique().tolist()
    nclust = X_df['CLUSTER_LABEL'].nunique()
    X_arr = np.array(X_df.drop('CLUSTER_LABEL', axis=1))
    wss = 0.0
    for lab in labels:
        cluster_size = len(labels[labels == lab].tolist())
        cluster_nums = X_df[X_df['CLUSTER_LABEL'] == lab].index.values.tolist()
        ncomb = int(factorial(cluster_size) / (factorial(cluster_size - 2) * factorial(2)))
        combn = combinations(cluster_nums, 2)
        for _ in tqdm(range(ncomb)):
            i, j = next(combn)
            wss += pow(distance.euclidean(X_arr[i, :], X_arr[j, :]), 2)
    bss = 0.0
    ncomb_clust = int(factorial(nclust) / (factorial(nclust - 2) * factorial(2)))
    combn_clust = combinations(clusters, 2)
    for _ in tqdm(range(ncomb_clust)):
        lab1, lab2 = next(combn_clust)
        idx1 = X_df[(X_df['CLUSTER_LABEL'] == lab1)].index.values.tolist()
        idx2 = X_df[(X_df['CLUSTER_LABEL'] == lab2)].index.values.tolist()
        for i, j in product(idx1, idx2, repeat=1):
            bss += pow(distance.euclidean(X_arr[i, :], X_arr[j, :]), 2)
    rsq_ind = bss / (bss + wss)
    return wss, bss, rsq_ind

def get_cluster_pairplots(df, factors, nm_cluster='CLUSTER_LABEL', path=path):
    clusters = df[nm_cluster].unique().tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[factors])
    X_df = pd.DataFrame(data=X, columns=factors)
    X_df[nm_cluster] = df[nm_cluster].copy()
    for comb in tqdm(list(combinations(clusters, 2))):
        clust_labels = list(comb)
        df_temp = X_df[X_df[nm_cluster].isin(clust_labels)].copy()
        sb.pairplot(df_temp, hue=nm_cluster, diag_kind='hist')
        plt.savefig(
            path.joinpath(f'pairplots_{clust_labels[0]}_{clust_labels[1]}.png'),
            bbox_inches='tight'
        )
        plt.close()
        del df_temp
    return None

# Cophenetic distances
N = 1000000  # number of observations
Nsim = 100000  # number of simulations
cluster_labels = np.array([], dtype='int32')  # matrix of size (N, Nsim)
# Participation of each observation in the cluster bootstrap
participation = np.zeros(N, dtype='float32')
for i in tqdm(range(N)):
    participation[i] = np.nonzero(cluster_labels[i, :])[0].shape[0] / Nsim
f_out = tb.open_file(path.joinpath('coph_dist.h5'), 'w')
filters = tb.Filters(complevel=0)
out = f_out.create_carray(f_out.root, 'data', tb.Float32Atom(), shape=(N * (N + 1) / 2 - N, 1), filters=filters)
k = 0
for i in tqdm(range(N)):
    labels_i = cluster_labels[i, :].flatten()
    nsim_i = np.nonzero(labels_i)[0].shape[0]
    ind_i = np.where(labels_i != 0)
    coph_dists = []
    for j in range(i + 1, N):
        labels_j = cluster_labels[j, :].flatten()
        nsim_j = np.nonzero(labels_j)[0].shape[0]
        ind_j = np.where(labels_j != 0)
        ind_ij = np.intersect1d(ind_i, ind_j)
        labels_ij = labels_i[ind_ij]
        labels_ji = labels_j[ind_ij]
        n_same = np.sum(labels_ij == labels_ji)
        n_base = min(nsim_i, nsim_j)
        dist_ij = n_same / n_base
        coph_dists.append(dist_ij)
    out[k:(k + len(coph_dists)), 0] = coph_dists
    k += (len(coph_dists) - 1)
