import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import scorecardpy as sc
import seaborn as sb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

path = Path(os.path.abspath(__file__)).parent

def calculate_psi(expected, actual, bucket_type='bins', buckets=10, axis=0):
    """
    :param expected: numpy matrix
    :param actual: numpy matrix of the same size as expected
    :param bucket_type: 'bins' or 'quantiles'
    :param buckets: number of buckets (int)
    :param axis: 0=vertical, 1=horizontal
    :return: ndarray of PSI values for each variable
    """
    def psi(expected_array, actual_array, buckets):
        """
        :param expected_array: numpy array
        :param actual_array: numpy array of the same size as expected
        :param buckets: number of buckets (int)
        :return: PSI value for a single variable
        """
        def scale_range(raw_range, lower, upper):
            scaled_range = raw_range.copy()
            scaled_range -= np.min(scaled_range)
            scaled_range /= np.max(scaled_range) / (upper - lower)
            scaled_range += lower
            return scaled_range
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        if bucket_type == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif bucket_type == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
        else:
            raise ValueError("This bucket type is not supported.")
        exp_perc = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        act_perc = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value
        psi_value = sum([sub_psi(exp_perc[i], act_perc[i]) for i in range(0, len(exp_perc))])
        return psi_value
    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])
    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)
        else:
            raise ValueError("Axis argument not supported.")
    return psi_values

def anova_1way(y_name, x_name, data):
    df = data[[x_name, y_name]].copy()
    glob_mean = df[y_name].mean()
    ss_total = np.sum((df[y_name] - glob_mean) ** 2)
    group_means = df.groupby(x_name).mean().rename(columns={y_name: 'group_mean'})
    df = pd.merge(df, group_means, left_on=x_name, right_index=True)
    ss_resid = np.sum((df[y_name] - df['group_mean']) ** 2)
    ss_explained = np.sum((df['group_mean'] - glob_mean) ** 2)
    assert np.allclose(ss_total, ss_explained + ss_resid)
    n_groups = df[x_name].nunique()
    df1 = n_groups - 1
    df2 = df.shape[0] - n_groups
    ms_explained = ss_explained / df1
    ms_resid = ss_resid / df2
    f_stat = ms_explained / ms_resid
    pval = 1.0 - ss.f.cdf(f_stat, df1, df2)
    return f_stat, pval

def calc_gini_confint(g, n0, n1, alpha=0.05):
    theta = (1.0 + g) / 2.0
    Q1 = theta / (2.0 - theta)
    Q2 = 2.0 * (theta ** 2) / (1.0 + theta)
    sigma = np.sqrt((theta * (1.0 - theta) + (n1 - 1) * (Q1 - theta ** 2) + (n0 - 1) * (Q2 - theta ** 2)) / (n0 * n1))
    sigma_g = 2.0 *  sigma
    z_crit = ss.norm.ppf(1.0 - alpha / 2.0)
    g_lower = g - sigma_g * z_crit
    g_upper = g + sigma_g * z_crit
    return g_lower, g_upper


if __name__ == '__main__':
    df = pd.read_hdf(path.joinpath('.hd5'))
    # Example: individual factor Gini
    y = df['target'].values
    factors = []
    df_gini = pd.DataFrame(columns=['Gini_LB', 'Gini_coef', 'Gini_UB'])
    for f in factors:
        m = LogisticRegression(fit_intercept=True, penalty='none')
        x = np.array(df[f].values).reshape(-1, 1)
        m.fit(x, y)
        gini = 2 * roc_auc_score(y, m.predict_proba(x)[:, 1]) - 1
        n1 = y.sum()
        n0 = df.shape[0] - n1
        gini_lb, gini_ub = calc_gini_confint(gini, n0=n0, n1=n1)
        df_gini.at[f, 'Gini_LB'] = gini_lb
        df_gini.at[f, 'Gini_coef'] = gini
        df_gini.at[f, 'Gini_UB'] = gini_ub
    # Example: WOE binning
    factors_woe = []
    df_woe = df[factors_woe + ['target']].copy()
    bins = sc.woebin(df_woe, y='target')
    sc.woebin_plot(bins)
    breaks_adj = {'factor1': ['level1', 'level2', 'level3']}
    bins_adj = sc.woebin(df_woe, y='target', breaks_list=breaks_adj)
    factors_proc = [f + '_woe' for f in factors_woe]
    df[factors_proc] = sc.woebin_ply(df[factors_woe], bins_adj)
    # Example: histogram for PSI
    factor = ''
    X = np.array(df[factors])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    df_train = pd.DataFrame(X_train, columns=factors)
    df_test = pd.DataFrame(X_test, columns=factors)
    _ = plt.hist(
        df_train[factor], df_test[factor],
        color=['blue', 'red'],
        label=['train', 'test'],
        density=True, alpha=0.5
    )
    plt.legend()
    plt.title(f"Distribution for factor {factor}")
    # Example: correlation matrix
    corr_mat = df[factors].corr(method='pearson')
    xticks = yticks = factors
    fig, ax = plt.subplots(figsize=(30, 30))
    sb.heatmap(
        corr_mat, annot=True, vmax=1.0, vmin=-1.0, cmap=plt.cm.coolwarm,
        ax=ax, yticklabels=yticks, xticklabels=xticks
    )
    plt.yticks(rotation=0)
    plt.show()
    # Example: rank correlations with target
    df_cor = pd.DataFrame(columns=['Corr_spearman', 'Corr_kendall'])
    for f in tqdm(factors):
        rho = ss.spearmanr(df[f], df['target'])[0]
        tau = ss.kendalltau(df[f], df['target'])[0]
        df_cor.at[f, 'Corr_spearman'] = rho
        df_cor.at[f, 'Corr_kendall'] = tau
