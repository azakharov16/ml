import os
import numpy as np
import pandas as pd
from pathlib import Path

path = Path(os.path.abspath(__file__)).parent

df = pd.read_hdf(path.joinpath('.hd5'))

def genr_rand_labels(df, nm_cluster='CLUSTER_LABEL', nrep=1000):
    Nobs = df.shape[0]
    counts = df[nm_cluster].value_counts(normalize=True)
    clusters = counts.index
    props = counts.values
    rand_labels = np.zeros(shape=(Nobs, nrep), dtype=int)
    for i in range(nrep):
        np.random.seed(5 * i)
        rand_labels[:, i] = np.random.choice(a=clusters, size=Nobs, p=props, replace=True)
    return rand_labels
