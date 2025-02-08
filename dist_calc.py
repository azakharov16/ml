import os
import numpy as np
import pandas as pd
import tables as tb
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
path = Path(os.path.abspath(__file__)).parent

def pairwise_distance(path_a, path_b):
    seq_a, seq_b = path_a.split('->'), path_b.split('->')
    len_a, len_b = len(seq_a), len(seq_b)
    min_len = min([len_a, len_b])
    n = 0  # num of migrations from the same state
    m = 0  # num of migrations from the same state to the same state
    for i in range(min_len - 2):
        if seq_a[i] == seq_b[i]:
            n += 1
            if (seq_a[i + 1] == seq_b[i + 1]) or (seq_a[i + 2] == seq_b[i + 2]):
                m += 1
    if seq_a[min_len - 2] == seq_b[min_len - 2]:
        n += 1
        if seq_a[min_len - 1] == seq_b[min_len - 1]:
            m += 1
    try:
        similarity = m / n
    except ZeroDivisionError:
        similarity = 0.0
    return 1.0 - similarity

paths = np.array([], dtype=str)
N = len(paths)  # 1000000000
# A numpy array of size (N, N) cannot be created due to MemoryError
# Squared distance matrix
f = tb.open_file(path.joinpath('dist_mat.h5'), 'w')
filters = tb.Filters(complevel=5, complib='blosc')
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(N, N), filters=filters)
for i in tqdm(range(N)):
    out[i, :] = list(map(lambda x: pairwise_distance(x, paths[i]), paths))
f.close()
# Condensed distance matrix
f = tb.open_file(path.joinpath('dist_vec.h5'), 'w')
filters = tb.Filters(complevel=0)
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(N * (N + 1) / 2 - N, 1), filters=filters)
j = 0
for i in tqdm(range(N - 1)):
    triu_row = list(map(lambda x: pairwise_distance(x, paths[i]), paths[(i + 1):]))
    out[j:(j + len(triu_row)), 0] = triu_row
    j += len(triu_row) - 1
f.close()
# TODO: vectorize with partial and np.vectorize(), compare speed

# Simple Euclidean distance
df = pd.read_hdf(path.joinpath('.hd5'))
Nobs = df.shape[0]
num_factors = []
sc = StandardScaler()
X_num = sc.fit_transform(df[num_factors])
f = tb.open_file(path.joinpath('euclid_dist_vec.h5'), 'w')
filters = tb.Filters(complevel=0)
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(N * (N + 1) / 2 - N, 1), filters=filters)
k = 0
for i in tqdm(Nobs):
    dists = []
    for j in range(i + 1, Nobs):
        dist_ij = distance.euclidean(X_num[i, :], X_num[j, :])
        dists.append(dist_ij)
    out[k:(k + len(dists)), 0] = dists
    k += len(dists) - 1
f.close()
