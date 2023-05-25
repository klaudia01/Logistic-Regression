import numpy as np
from sklearn.datasets import make_classification


def data_generator():
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
