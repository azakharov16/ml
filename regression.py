import os
from locale import normalize

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from matplotlib import pyplot as plt
from pathlib import Path

plt.style.use('seaborn')
path = Path(os.path.abspath(__file__)).parent
df = pd.read_hdf(path.joinpath('.hd5'))
factors = []
X = np.array(df[factors])
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)

def winsorize_sample(df, factors, upper=True, lower=True, q=0.95, df_source=None):
    df_w = df.copy()
    if df_source is None:
        for col in factors:
            lbound = {False: None, True: df[col].quantile(1 - q)}[lower]
            ubound = {False: None, True: df[col].quantile(q)}[upper]
            df_w[col] = df_w[col].clip(lower=lbound, upper=ubound)
    else:
        for col in factors:
            lbound = {False: None, True: df_source[col].quantile(1 - q)}[lower]
            ubound = {False: None, True: df_source[col].quantile(q)}[upper]
            df_w[col] = df_w[col].clip(lower=lbound, upper=ubound)
    return df_w

def rsq_adj(rsq, n, p):
    return 1.0 - (1.0 - rsq) * (n - 1) / (n - p - 1)

# Forward selection
scorer = make_scorer(r2_score)
linreg = LinearRegression(fit_intercept=True, normalize=False)
fs = SequentialFeatureSelector(linreg, n_features_to_select=5, direction='forward', scoring=scorer, cv=5)
fs.fit(X_train, y_train)
print(np.array(factors)[fs.get_support()])

# Backward elimination
be = SequentialFeatureSelector(linreg, n_features_to_select=5, direction='backward', scoring=scorer, cv=5)
be.fit(X_train, y_train)
print(np.array(factors)[be.get_support()])

# RFECV
model = LinearRegression(fit_intercept=True, normalize=False)
rfecv = RFECV(model, step=1, cv=5, min_features_to_select=1)
rfecv.fit(X_train, y_train)
print(np.array(factors)[rfecv.support_])
plt.figure(figsize=(12, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# LassoCV
lasso = LassoCV(
    eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
    max_iter=1000, tol=0.0001, cv=5, random_state=32, selection='cyclic'
)
lasso.fit(X_train, y_train)
print(np.array(factors)[lasso.coef_ > 0])

# Logistic regression constructs
model = LogisticRegression(fit_intercept=True, solver='newton-cg', max_iter=500, penalty='none')
rfe = RFE(model, step=1, verbose=2, n_features_to_select=4)
# rfe.fit(X_train, y_train)
# print(np.array(factors)[rfe.support_])
# gini = 2.0 * roc_auc_score(y_test, rfe.predict_proba(X_test)[:, 1]) - 1.0
m_l1 = LogisticRegression(fit_intercept=True, solver='liblinear', penalty='l1', C=1.0)
m_l2 = LogisticRegression(fit_intercept=True, solver='newton-cg', penalty='l2', C=1.0)
