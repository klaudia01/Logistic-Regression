# Klaudia Barabasz 259046
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = None
        self.features = None
        self.samples = None

    def fit(self, x, y):
        self.samples, self.features = x.shape
        self.weight = np.zeros(self.features)

        for i in range(self.iterations):
            z = np.dot(x, self.weight)
            sigmoid = 1 / (1 + np.exp(-z))
            gradient_descent = np.dot(x.T, sigmoid - y) / self.samples
            self.weight -= self.learning_rate * gradient_descent
        return self

    def predict(self, x):
        z = np.dot(x, self.weight)
        sigmoid = 1 / (1 + np.exp(-z))
        labels = []
        for i in sigmoid:
            if i > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        return labels
