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

class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y):
        n_samples = len(X)
        indices = np.arange(n_samples)

        #shuffle indices if requested
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        #split indices into N_splits equal sized folds
        fold_sizes = np.full(self.n_splits, n_samples// self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        folds = []

        for fold_size in fold_sizes:
            folds.append(indices[current:current+fold_size])
            current+=fold_size
        
        #yield train and test indices for each fold
        for i in range(self.n_splits):
            test_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_indices, test_indices

class


