import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iteration=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iteration
        self.weights = None
        self.bias = None
        self.loss_history = []


    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = self._predict(X)

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._mse(y, y_pred)
            self.loss_history.append(loss)

        return self
    
    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return self._r2(y,y_pred)
    
    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)** 2)
    
    def _r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)