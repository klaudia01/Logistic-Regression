import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid_function(x, weight):
    # wyznaczanie wartości funkcji sigmoidalnej
    z = np.dot(x, weight)
    return 1 / (1 + np.exp(-z))


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate  # inicjalizacja szybkości uczenia
        self.iterations = iterations  # inicjalizacja liczby iteracji
        self.weight = None  # inicjalizacja wag
        self.features = None  # inicjalizacja liczby cech
        self.samples = None  # inicjalizacja liczby próbek

    def fit(self, x, y):
        self.samples, self.features = x.shape
        self.weight = np.zeros(self.features)

        for i in range(self.iterations):
            sigmoid = sigmoid_function(x, self.weight)
            # wyznaczanie gradientu prostego
            gradient_descent = np.dot(x.T, sigmoid - y) / self.samples
            # aktualizacja wag metodą gradientu prostego
            self.weight -= self.learning_rate * gradient_descent
        return self

    def predict(self, x):
        sigmoid = sigmoid_function(x, self.weight)
        labels = []
        # przypisanie 1, kiedy wartość funkcji sigmoidalnej jest większa niż 0.5 oraz 0 w pozostałych przypadkach
        for i in sigmoid:
            if i > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        return labels
