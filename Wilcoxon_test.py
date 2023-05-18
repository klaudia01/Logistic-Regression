import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums

scores = np.load('classifiers_accuracy.npy')
n_clfs = 4
alpha = 0.05

w_stat = np.zeros((n_clfs, n_clfs))
p_value = np.zeros((n_clfs, n_clfs))
better_scores = np.zeros((n_clfs, n_clfs), dtype=bool)
advantage = np.zeros((n_clfs, n_clfs), dtype=bool)

ranks = rankdata(scores, axis=1, method='min')
ranks = np.where(ranks > 3, 3, ranks)

for i in range(n_clfs):
    for j in range(n_clfs):
        w_stat[i, j], p_value[i, j] = ranksums(scores[:, i], scores[:, j])
        better_scores[i, j] = np.mean(scores[:, i]) > np.mean(scores[:, j])
        advantage[i, j] = better_scores[i, j] and p_value[i, j] < alpha

print(w_stat)
print(p_value)
print(better_scores)
print(advantage)
