import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from logistic_regression import LogisticRegression


def pca(X, y, n_components_list, optimal_iterations, rskf):

    # inicjalizacja list wyników metryk
    scores = np.zeros((10, len(n_components_list)))
    scores_2 = np.zeros((10, len(n_components_list)))
    scores_3 = np.zeros((10, len(n_components_list)))

    for j, n_components in enumerate(n_components_list):
        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            # Podział danych na zbiór treningowy i testowy
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # inicjalizacja PCA
            pca = PCA(n_components=n_components)
            # trenowanie algorytmu PCA
            pca.fit(X_train)
            # transformacja danych treningowych i testowych
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

            # inicjalizacja modelu regresji logistycznej
            lg = LogisticRegression(learning_rate=0.01, iterations=optimal_iterations)
            # trenowanie modelu i predykcja etykiet
            lg.fit(X_train, y_train)
            y_pred = lg.predict(X_test)

            # wyznaczanie wyników metryk
            scores[i, j] = accuracy_score(y_test, y_pred)
            scores_2[i, j] = precision_score(y_test, y_pred)
            scores_3[i, j] = recall_score(y_test, y_pred)

    # zapisanie wyników do pilku
    np.savez('PCA_results.npz', scores, scores_2, scores_3)

    # wyznaczanie średnich wyników metryk i odchyleń standardowych
    mean_scores = np.round(np.mean(scores, axis=0), 3)
    std_scores = np.round(np.std(scores, axis=0), 3)

    mean_scores_2 = np.round(np.mean(scores_2, axis=0), 3)
    std_scores_2 = np.round(np.std(scores_2, axis=0), 3)

    mean_scores_3 = np.round(np.mean(scores_3, axis=0), 3)
    std_scores_3 = np.round(np.std(scores_3, axis=0), 3)

    return mean_scores, mean_scores_2, mean_scores_3, std_scores, std_scores_2, std_scores_3
