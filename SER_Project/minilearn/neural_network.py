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

        # initalize wegihts with small random values
        np.random.seed(42)
        self.weights1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.bias1 = np.zeros((1,self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, n_classes) * 0.01
        self.bias2 = np.zeros((1,n_classes))

        #training loop
        for i in range(self.n_iterations):
            #forward pass
            z1 = np.dot(X, self.weights1) + self.bias1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.weights2) + self.bias2
            a2 = self._softmax(z2)

            #compute cross entropy loss
            loss = -np.mean(np.sum(y_one_hot * np.log(a2 + 1e-8), axis=1))
            self.loss_history.append(loss)

            #backward pass
            dz2 = a2 - y_one_hot
            dw2 = (1/n_samples) * np.dot(a1.T, dz2)
            db2 = (1/n_samples) * np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.weights2.T)
            dz1 = da1 * self._relu_derivative(z1)
            dw1 = (1/n_samples) * np.dot(X.T, dz1)
            db1 = (1/n_samples) * np.sum(dz1, axis=0, keepdims=True)

            # update weights
            self.weights2 -= self.learning_rate * dw2
            self.bias2 -= self.learning_rate * db2
            self.weights1 -= self.learning_rate * dw1
            self.bias1 -= self.learning_rate * db1

            if (i+1) %100 ==0:
                print(f"Iteration {i+1}/{self.n_iterations} | Loss: {loss:.4f}")
        return self
    
    def predict(self, X):
        #forward pass
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self._softmax(z2)

        #pcik class with hightest probability
        y_idx = np.argmax(a2, axis=1)
        return np.array([self.classes_[idx] for idx in y_idx])
    
    def score(self, X, y):
        #calculate accuracy
        return np.mean(self.predict(X) ==y)
    
