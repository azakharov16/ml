import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tqdm import tqdm

path = Path(os.path.abspath(__file__)).parent
df = pd.read_hdf(path.joinpath('.hd5'))
factors = []
sc = StandardScaler()
X = sc.fit_transform(df[factors])

# Manual gridsearch for DBSCAN
eps_grid = np.arange(0.1, 1.6, 0.1)
npar = len(eps_grid)
n_start = 5  # including
n_end = 55  # excluding
step = 5  # int only
nstep = (n_end - n_start) / step
nlarge_pred = np.zeros(shape=(npar, nstep), dtype=int)
large_prop = np.zeros(shape=(npar, nstep), dtype=float)
i = 0
for eps in tqdm(eps_grid):
    j = 0
    for n in range(n_start, n_end, step):
        dbscan = DBSCAN(eps=eps, min_samples=n)
        label_arr = dbscan.fit_predict(X)
        N = label_arr.size
        labels, counts = np.unique(label_arr, return_counts=True)
        large_pred = labels[np.logical_and(counts > 100, labels != -1)]
        nlarge_pred[i, j] = large_pred.size
        large_prop[i, j] = label_arr[np.in1d(label_arr, large_pred)].size / N
        j += 1
    i += 1

df1 = pd.DataFrame(
    data=nlarge_pred,
    index=['eps_' + str(round(e, 2)) for e in eps_grid],
    columns=['min_samples_' + str(n) for n in range(n_start, n_end, step)]
)
df2 = pd.DataFrame(
    data=large_prop,
    index=['eps_' + str(round(e, 2)) for e in eps_grid],
    columns=['min_samples_' + str(n) for n in range(n_start, n_end, step)]
)

