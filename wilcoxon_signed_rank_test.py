import numpy as np
from scipy.stats import ranksums


def wilcoxon_test(scores, n_clfs, alpha):
    w_stat = np.zeros((n_clfs, n_clfs))
    p_value = np.zeros((n_clfs, n_clfs))
    better_scores = np.zeros((n_clfs, n_clfs), dtype=bool)
    sig_difference = np.zeros((n_clfs, n_clfs), dtype=bool)
    results = np.zeros((n_clfs, n_clfs), dtype=bool)

    for i in range(n_clfs):
        for j in range(n_clfs):
            w_stat[i, j], p_value[i, j] = ranksums(scores[:, i], scores[:, j])
            better_scores[i, j] = np.mean(scores[:, i]) > np.mean(scores[:, j])
            sig_difference[i, j] = better_scores[i, j] and p_value[i, j] < alpha
            results[i, j] = better_scores[i, j] and sig_difference[i, j]

    return w_stat, results
