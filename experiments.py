# Klaudia Barabasz 259046
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from logistic_regression import LogisticRegression

X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=5,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)
# zapisanie wygenerowanych danych syntetycznych do pliku CSV
dataset = np.concatenate((X, y[:, None]), axis=1)
np.savetxt('dataset.csv', dataset, delimiter=',')

# inicjalizacja walidacji krzyżowej
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

# Eksperyment 1: Znalezienie optymalnego hiperparametru liczby iteracji

# inicjalizacja listy możliwych wartości liczby iteracji
iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]

# inicjalizacja macierzy wyników
scores = np.zeros((10, 7))

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

    # wyznaczanie średniego wyniku metryki
    mean_scores = np.round(np.mean(scores, axis=0), 3)

    # znalezienie optymalnej wartości liczby iteracji
    optimal_iterations = iterations_list[np.argmax(mean_scores)]

    # wyświetlanie wyników
    df = pd.DataFrame({'Liczba iteracji': iterations_list,
                       'Dokładność': [mean_scores[0], mean_scores[1], mean_scores[2], mean_scores[3], mean_scores[4],
                                      mean_scores[5], mean_scores[6]]})

print(df, '\n\nOptymalna liczba iteracji: ', optimal_iterations, '\n')

# zapisanie wyników do pilku
np.save('n_iterations_results.npy', scores)

# Eksperyment 2: Ekstrakcja (PCA) i selekcja (KBest) cech

# inicjalizacja list wyników metryk
scores = []
scores_2 = []
scores_3 = []

# inicjalizacja PCA
pca = PCA(n_components=6)

for i, (train, test) in enumerate(rskf.split(X, y)):
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # trenowanie algorytmu KBest
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
    accuracy = accuracy_score(y_test, y_pred)
    scores = np.append(scores, accuracy)

    precision = precision_score(y_test, y_pred)
    scores_2.append(precision)

    recall = recall_score(y_test, y_pred)
    scores_3.append(recall)

# wyznaczanie średnich wyników metryk
mean_accuracy_PCA = np.round(np.mean(scores), 3)
mean_precision_PCA = np.round(np.mean(scores_2), 3)
mean_recall_PCA = np.round(np.mean(scores_3), 3)

# zapisanie wyników do pilku
np.savez('PCA_results.npz', scores, scores_2, scores_3)

# inicjalizacja KBest
KBest = SelectKBest(k=int(np.sqrt(X.shape[1])))

for i, (train, test) in enumerate(rskf.split(X, y)):
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # trenowanie algorytmu KBest
    KBest.fit(X_train, y_train)

    # inicjalizacja modelu regresji logistycznej
    lg = LogisticRegression(learning_rate=0.01, iterations=optimal_iterations)
    # trenowanie modelu i predykcja etykiet
    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)

    # wyznaczanie wyników metryk
    accuracy = accuracy_score(y_test, y_pred)
    scores = np.append(scores, accuracy)

    precision = precision_score(y_test, y_pred)
    scores_2.append(precision)

    recall = recall_score(y_test, y_pred)
    scores_3.append(recall)

# wyznaczanie średnich wyników metryk
mean_accuracy_KBest = np.round(np.mean(scores), 3)
mean_precision_KBest = np.round(np.mean(scores_2), 3)
mean_recall_KBest = np.round(np.mean(scores_3), 3)

# zapisanie wyników do pilku
np.savez('KBest_results.npz', scores, scores_2, scores_3)

# wyświetlanie wyników
df = pd.DataFrame({'Metoda preprocessingu': ['PCA', 'KBest'],
                   'Dokładność': [mean_accuracy_PCA, mean_accuracy_KBest],
                   'Precyzja': [mean_precision_PCA, mean_precision_KBest],
                   'Czułość': [mean_recall_PCA, mean_recall_KBest]})

print(df, '\n')

# Eksperyment 3: Porównanie do metod referencyjnych

# inicjalizacja macierzy wyników
scores = np.zeros((10, 5))
scores_2 = np.zeros((10, 5))
scores_3 = np.zeros((10, 5))

# inicjalizacja listy klasyfikatorów
classifiers = [LogisticRegression(learning_rate=0.01, iterations=optimal_iterations),
               DecisionTreeClassifier(),
               GaussianNB(),
               KNeighborsClassifier(),
               LogisticRegressionSklearn()
               ]

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for i_classifier, classifier in enumerate(classifiers):
        # tworzenie kopii klasyfikatora
        clf = clone(classifier)
        # trenowanie modelu i predykcja etykiet
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # wyznaczanie wyników metryk
        scores[i, i_classifier] = accuracy_score(y_test, y_pred)
        scores_2[i, i_classifier] = precision_score(y_test, y_pred)
        scores_3[i, i_classifier] = recall_score(y_test, y_pred)

# wyznaczanie średnich wyników metryk
mean_scores = np.round(np.mean(scores, axis=0), 3)
mean_scores_2 = np.round(np.mean(scores_2, axis=0), 3)
mean_scores_3 = np.round(np.mean(scores_3, axis=0), 3)

# zapisanie wyników do pilku
np.savez('reference_methods_results.npz', scores, scores_2, scores_3)

# wyświetlanie wyników
df = pd.DataFrame({'Klasyfikator': ['LR', 'DT', 'GaussianNB', 'KNN', 'RL SKlearn'],
                   'Dokładność': [mean_scores[0], mean_scores[1], mean_scores[2], mean_scores[3], mean_scores[4]],
                   'Precyzja': [mean_scores_2[0], mean_scores_2[1], mean_scores_2[2], mean_scores_2[3],
                                mean_scores_2[4]],
                   'Czułość': [mean_scores_3[0], mean_scores_3[1], mean_scores_3[2], mean_scores_3[3],
                               mean_scores_3[4]]})
print(df)
