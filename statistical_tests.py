import numpy as np
import pandas as pd

from t_test import t_test


def statistical_tests():
    scores = np.load('classifiers_accuracy.npy')

    n_clfs = 5
    clfs = ['LR', 'DT', 'GaussianNB', 'KNN', 'LR SKlearn']
    alpha = 0.05

    # ----Test T-studenta----
    t_stat, results = t_test(scores, n_clfs, alpha)  # wywołanie testu

    # wyświetlanie wyników
    df = pd.DataFrame(t_stat, columns=clfs, index=clfs)
    print('Wartości t-statystki:\n', df, '\n')

    df = pd.DataFrame(results, columns=clfs, index=clfs)
    print('Czy klasyfikator osiągnął statystycznie lepsze wyniki:\n', df, '\n')

