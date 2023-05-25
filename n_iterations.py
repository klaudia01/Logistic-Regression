import numpy as np
from sklearn.metrics import accuracy_score

from logistic_regression import LogisticRegression


def n_iterations(X, y, iterations_list, rskf):
    # inicjalizacja macierzy wyników
    scores = np.zeros((10, len(iterations_list)))

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        # Podział danych na zbiór treningowy i testowy
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for i_iterations, iterations in enumerate(iterations_list):
            # inicjalizacja modelu regresji logistycznej
            lg = LogisticRegression(learning_rate=0.01, iterations=iterations)
            # trenowanie modelu i predykcja etykiet
            lg.fit(X_train, y_train)
            y_pred = lg.predict(X_test)
            scores[i, i_iterations] = accuracy_score(y_test, y_pred)

    # zapisanie wyników do pilku
    np.save('n_iterations_results.npy', scores)

    # wyznaczanie średniego wyniku metryki
    mean_scores = np.round(np.mean(scores, axis=0), 3)
    std_scores = np.round(np.std(scores, axis=0), 3)

    # znalezienie optymalnej wartości liczby iteracji
    optimal_iterations = iterations_list[np.argmax(mean_scores)]

    return optimal_iterations, mean_scores, std_scores
