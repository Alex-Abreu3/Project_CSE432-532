import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_= None
        self.std_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self,X):
        return(X- self.mean_) / (self.std_ + 1e-8)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def train_test_split(X, y, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        n = len(X)
        n_test = int(n*test_size)

        indices =np.random.permutation(n)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
