import numpy as np
import random as rnd
from .metrics import Metrics

class LogisticRegression:
    def __init__(self, n_features, log_path):
        self.n_features = n_features
        self.weights = np.random.randn(self.n_features) * 0.01
        self.bias = 0
        self.log_path = log_path

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self.sigmoid(X @ self.weights + self.bias)
    
    def fit(self, X, Y_true, learning_rate=0.05, epochs=1000, beta=0.01, eps=0.4, log_p=10):
        logs = []
        risk = np.mean(Metrics.log_loss(self.predict(X), Y_true))
        for epoch in range(1, epochs + 1):
            i = rnd.randint(0, X.shape[0] - 1)
            Y_pred = self.predict(X[i])
            delta = Y_pred - Y_true[i]
            dw = (X[i] * delta)
            db = delta
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            loss = Metrics.s_log_loss(Y_pred, Y_true[i])
            risk = beta * loss + (1 - beta) * risk
            if epoch % log_p == 0:
                logs.append(f'Epoch: {epoch}, Loss: {risk}\n')
            if risk <= eps:
                break
        self._save_logs(logs)
    
    def _save_logs(self, logs):
        with open(self.log_path, 'w') as file:
            file.write(''.join(logs))
