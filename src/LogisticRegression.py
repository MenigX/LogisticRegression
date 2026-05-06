import numpy as np
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
    
    def fit(self, X, Y_true, learning_rate=0.05, epochs=1000):
        logs = []
        for epoch in range(1, epochs + 1):
            Y_pred = self.predict(X)
            delta = Y_pred - Y_true
            dw = (X.T @ delta) / Y_true.shape[0]
            db = np.sum(delta) / Y_true.shape[0]
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            if epoch % 100 == 0:
                logs.append(f'Epoch: {epoch}, Loss: {Metrics.log_loss(Y_pred, Y_true)}\n')
        self.save_logs(logs)
    
    def save_logs(self, logs):
        with open(self.log_path, 'w') as file:
            file.write(''.join(logs))
