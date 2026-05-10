import numpy as np

class ANN:
    def __init__(self, hidden_size=128, learning_rate=0.01, n_iterations=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights1 = None
        self.weights2 = None
        self.bias1 = None
        self.bias2 = None
        self.classes_ = None
        self.loss_history =[]
    
    def _relu(self,z):
        #relu activation returns zero for negative vales z for postive
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        #derivative of relu 1 where z>0, 0 elswhere
        return (z > 0).astype(float)
    
    def _softmax(self,z):
        #convert raw scores to probabilities that sum to 1
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot(self, y, n_classes):
        #convert ingar labels to one hot encoded matrix
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)),y] =1
        return one_hot
    def fit(self, X,y):
        n_samples, n_features = X.shape

        #get all unique emotion labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        #map string labels to integers
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([self.label_to_idx[label] for label in y])
        y_one_hot = self._one_hot(y_encoded, n_classes)

        