import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        #center the data by subtracting the mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        #compute the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        #compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        #sorted eignvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        #keep only the top n_components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_=(self.explained_variance_ / np.sum(eigenvalues))

        return self
