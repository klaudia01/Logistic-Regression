import numpy as np
from scipy.stats import ttest_rel

scores = np.load('classifiers_accuracy.npy')
n_clfs = 4
alpha = 0.05

t_stat = np.zeros((n_clfs, n_clfs))
p_value = np.zeros((n_clfs, n_clfs))
better_scores = np.zeros((n_clfs, n_clfs), dtype=bool)
advantage = np.zeros((n_clfs, n_clfs), dtype=bool)

for i in range(n_clfs):
    for j in range(n_clfs):
        t_stat[i, j], p_value[i, j] = ttest_rel(scores[:, i], scores[:, j])
        better_scores[i, j] = np.mean(scores[:, i]) > np.mean(scores[:, j])
        advantage[i, j] = better_scores[i, j] and p_value[i, j] < alpha

print(t_stat)
print(p_value)
print(better_scores)
print(advantage)
