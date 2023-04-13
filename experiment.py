# Klaudia Barabasz 259046
from logistic_regression import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

x, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=1
        )

cross_validation = RepeatedKFold(n_splits=2, n_repeats=5)
lg = LogisticRegression(0.01, 1000)
scores = cross_val_score(lg, x, y, scoring='accuracy', cv=cross_validation)
print("Dokładność: ", np.mean(scores))
