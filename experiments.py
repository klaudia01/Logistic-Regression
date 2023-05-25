import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from logistic_regression import LogisticRegression
from data_generator import data_generator
from n_iterations import n_iterations
from reference_methods import reference_methods
from pca import pca
from k_best import k_best


def experiments():
    # wczytywanie danych
    data = 'dataset.csv'
    try:
        data = np.genfromtxt(data, delimiter=',')
    except FileNotFoundError:
        data_generator()
        data = np.genfromtxt(data, delimiter=',')

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)  # inicjalizacja walidacji krzyżowej

    metrics = ['Dokładność', 'Precyzja', 'Czułość']  # inicjalizacja listy nazw metryk

    # ----Eksperyment 1: Znalezienie optymalnego hiperparametru liczby iteracji----
    iterations_list = [5, 10, 20, 30, 50, 100, 200]  # inicjalizacja listy możliwych wartości liczby iteracji

    optimal_iterations, mean_scores, std_scores = n_iterations(X, y, iterations_list, rskf)  # wywołanie eksperymentu

    # wyświetlanie wyników
    show_scores_iter = []

    for i, score, std_score in zip(iterations_list, mean_scores, std_scores):
        s = f'{score} ± {std_score}'
        show_scores_iter.append([i, s])

    df = pd.DataFrame(show_scores_iter, columns=['Liczba iteracji', 'Dokładność'])

    print(df, '\n\nOptymalna liczba iteracji: ', optimal_iterations, '\n')

    # wyświetlanie wykresu
    plt.plot(iterations_list, mean_scores)
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Dokładność')
    plt.title('Wykres zależności dokładności od liczby iteracji')
    plt.show()

    # ----Eksperyment 2: Ekstrakcja (PCA) i selekcja (KBest) cech----
    n_features = [1, 3, 6, 9, 12]  # inicjalizacja listy możliwych wartości liczby cech

    # wywołanie eksperymentu
    mean_scores, mean_scores_2, mean_scores_3, std_scores, std_scores_2, std_scores_3 = pca(X, y, n_features,
                                                                                            optimal_iterations, rskf)

    # wyświetlanie wyników PCA
    show_scores_pca = []

    for i, score, std_score, score_2, std_score_2, score_3, std_score_3 in zip(n_features, mean_scores, std_scores,
                                                                               mean_scores_2, std_scores_2,
                                                                               mean_scores_3,
                                                                               std_scores_3):
        s = f'{score} ± {std_score}'
        s_2 = f'{score_2} ± {std_score_2}'
        s_3 = f'{score_3} ± {std_score_3}'
        show_scores_pca.append([i, s, s_2, s_3])

    df = pd.DataFrame(show_scores_pca, columns=['Liczba cech'] + metrics)
    print('PCA\n', df, '\n')

    # wyświetlanie wykresu
    plt.plot(n_features, mean_scores)
    plt.xlabel('Liczba zachowanych cech')
    plt.ylabel('Dokładność')
    plt.title('Wykres zależności dokładności od liczby zachowanych cech')
    plt.show()

    # wywołanie eksperymentu
    mean_scores, mean_scores_2, mean_scores_3, std_scores, std_scores_2, std_scores_3 = k_best(X, y, n_features,
                                                                                               optimal_iterations, rskf)

    # wyświetlanie wyników KBest
    show_scores_kbest = []

    for i, score, std_score, score_2, std_score_2, score_3, std_score_3 in zip(n_features, mean_scores, std_scores,
                                                                               mean_scores_2, std_scores_2,
                                                                               mean_scores_3,
                                                                               std_scores_3):
        s = f'{score} ± {std_score}'
        s_2 = f'{score_2} ± {std_score_2}'
        s_3 = f'{score_3} ± {std_score_3}'
        show_scores_kbest.append([i, s, s_2, s_3])

    df = pd.DataFrame(show_scores_kbest, columns=['Liczba cech'] + metrics)
    print('KBest\n', df, '\n')

    # wyświetlanie wykresu
    plt.plot(n_features, mean_scores)
    plt.xlabel('Liczba wybranych cech')
    plt.ylabel('Dokładność')
    plt.title('Wykres zależności dokładności od liczby wybranych cech')
    plt.show()

    # ----Eksperyment 3: Porównanie do metod referencyjnych----

    # inicjalizacja listy klasyfikatorów
    classifiers = [LogisticRegression(learning_rate=0.01, iterations=optimal_iterations),
                   DecisionTreeClassifier(),
                   GaussianNB(),
                   KNeighborsClassifier(),
                   LogisticRegressionSklearn()
                   ]

    # wywołanie eksperymentu
    mean_scores, mean_scores_2, mean_scores_3, std_scores, std_scores_2, std_scores_3 = reference_methods(X, y,
                                                                                                          classifiers,
                                                                                                          rskf)

    # wyświetlanie wyników
    clfs = ['LR', 'DT', 'GaussianNB', 'KNN', 'LR SKlearn']
    show_scores_clfs = []

    for i, score, std_score, score_2, std_score_2, score_3, std_score_3 in zip(clfs, mean_scores, std_scores,
                                                                               mean_scores_2, std_scores_2,
                                                                               mean_scores_3,
                                                                               std_scores_3):
        s = f'{score} ± {std_score}'
        s_2 = f'{score_2} ± {std_score_2}'
        s_3 = f'{score_3} ± {std_score_3}'
        show_scores_clfs.append([i, s, s_2, s_3])

    df = pd.DataFrame(show_scores_clfs, columns=['Klasyfikator'] + metrics)
    print(df, '\n')
