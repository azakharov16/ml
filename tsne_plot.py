import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import  StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
plt.style.use('seaborn')
path = Path(os.path.abspath(__file__)).parent
df = pd.read_csv(path.joinpath('.csv'))

# Numeric factors
factors=[]
sc = StandardScaler()
X = sc.fit_transform(df[factors])
X_df = pd.DataFrame(data=X, columns=factors)
X_df.reset_index(inplace=True, drop=True)
tsne = TSNE(n_components=2, random_state=17)
X_tsne = tsne.fit_transform(X_df)
for f in factors:
    ncol = df[f].nunique()
    plt.figure(figsize=(24, 20))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df[f].values, cmap=plt.get_cmap('nipy_spectral', ncol))
    plt.title(f"tSNE projection with colored {f}")
    plt.colorbar()
    plt.savefig(path.joinpath(f'tsne_proj_{f}.png'), bbox_inches='tight')
    plt.close()

# Binary / categorical factors
colors_vec = df['f_flag'].map({0: 'blue', 1: 'orange'}).tolist()
plt.figure(figsize=(24, 20))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=colors_vec)
plt.title(f"...")
plt.savefig(path.joinpath('tsne_proj_f_flag.png'), bbox_inches='tight')
plt.close()
