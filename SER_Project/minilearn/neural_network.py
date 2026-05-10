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
    
    