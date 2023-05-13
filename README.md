# Regresja logistyczna
## Struktura plików
**logistic_regression.py** - implementacja algorytmu regresji logistycznej

**experiments.py** - implementacja trzech eksperymentów: 
* Eksperyment 1: Znalezienie optymalnego hiperparametru liczby iteracji
* Eksperyment 2: Ekstrakcja (PCA) i selekcja (KBest) cech
* Eksperyment 3: Porównanie do metod referencyjnych (drzewo decyzyjne, naiwny klasyfikator bayesowski, K najbliższych 
sąsiadów oraz implementacja regresji logistycznej w bibliotece Scikit-learn)

**dataset.csv** - plik CSV zawierający wygenerowane dane syntetyczne

**n_iterations_results.npy** - plik zwierający wyniki eksperymentu 1

**PCA_results.npz** - plik zwierający wyniki ekstrakcji cech (PCA) z eksperymentu 2

**PCA_results.npz** - plik zwierający wyniki selekcji cech (KBest) z eksperymentu 3

**reference_methods_results.npz** - plik zwierający wyniki eksperymentu 3
## Uruchomienie
Uruchomienie eksperymentu jest możliwe poprzez włączenie skryptu **experiments.py**.