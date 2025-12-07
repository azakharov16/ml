import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from pathlib import Path

path = Path(r'/home/andrey/Documents/DATASETS')
plt.style.use('seaborn-v0_8')

df = pd.read_csv(path.joinpath('Titanic-Dataset.xls'))
df_prop = pd.crosstab(
    index=df['Sex'], columns=df['Pclass'], values=df['Survived'],
    aggfunc='mean'
)
df_age = pd.crosstab(
    index=df['Sex'], columns=df['Survived'], values=df['Age'],
    aggfunc='mean', dropna=True
)

df_gr = df[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass'], as_index=False).mean()

df_sample = pd.merge(
    df, df_gr.rename(columns={'Age': 'Mean_age'}),
    on=['Sex', 'Pclass'], how='left'
)
df_sample['Age'] = np.where(
    df_sample['Age'].isna(), df_sample['Mean_age'], df_sample['Age']
)
factors = ['Age', 'Sex', 'Pclass']
target_col = 'Survived'
df_sample['Sex'] = df_sample['Sex'].map({'male': 1, 'female': 0})
X = df_sample[factors].values
y = df_sample[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
# Default parameters:
# criterion = 'gini' [NOT 'log_loss']
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print(f"Precision: {round(precision_score(y_true=y_test, y_pred=y_pred) * 100, 2)}%")
print(f"Recall: {round(recall_score(y_true=y_test, y_pred=y_pred) * 100, 2)}%")
print(f"F1 score: {round(f1_score(y_true=y_test, y_pred=y_pred) * 100, 2)}%")

probs = tree.predict_proba(X_test)[:, 1]
gini = 2.0 * roc_auc_score(y_test, probs) - 1.0
print(f"Gini: {round(gini * 100, 2)}%")

# Permutation importance
result_train = permutation_importance(
    tree, X_train, y_train, n_repeats=40, random_state=45,
    scoring='roc_auc', n_jobs=-1
)
result_test = permutation_importance(
    tree, X_test, y_test, n_repeats=40, random_state=45,
    scoring='roc_auc', n_jobs=-1
)
sorted_importances_idx = result_train.importances_mean.argsort()
train_importances = pd.DataFrame(
    result_train.importances[sorted_importances_idx].transpose(),
    columns=np.array(factors)[sorted_importances_idx],
)
test_importances = pd.DataFrame(
    result_test.importances[sorted_importances_idx].transpose(),
    columns=np.array(factors)[sorted_importances_idx],
)

fig, ax = plt.subplots()
ax = test_importances.plot.box(vert=False, whis=10)
ax.set_title(f"Permutation Importances (Test set)")
ax.set_xlabel("Decrease in key loss metric")
ax.axvline(x=0, color="k", linestyle="--")
ax.figure.tight_layout()
plt.savefig(path.joinpath('perm_imp_titanic.png'))
plt.close()

# Shapely analysis
explainer = shap.TreeExplainer(
    tree, data=X_train, model_output='raw',
    feature_perturbation='interventional', feature_names=factors
)
shap_vals = explainer.shap_values(X_test, check_additivity=True)
fig, ax = plt.subplots()
shap.summary_plot(
    shap.Explanation(values=shap_vals, feature_names=factors),
    X_test, show=False
)
plt.savefig(path.joinpath('shap_beeswarm_titanic.png'))
plt.close()



