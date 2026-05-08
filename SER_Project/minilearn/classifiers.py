import numpy as np
# for calculation

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations= n_iterations
        self.weights = None
        self.bias = None
        self.loss_history=[]
    
    def _softmax(self, z):
        #subtracts the max value from nuerical stabilty before computing the exponetials
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        # normalize so each row of probabilities sums to 1 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot(self, y, n_classes):
        # creates a matrix of zerros with rows per sample and one column per class
        one_hot = np.zeros((len(y), n_classes))
        # place a 1 in the column corresponding to the correct class for each sample.
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # find all unqiue emotion labels in dataset
        self.classes_ = np.unique(y)
        n_classes=len(self.classes_)

        #map each emotion label to an index (angry = 0, calm =1)
        self.label_to_idx = {label:idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([self.label_to_idx[label] for label in y])

        #initialize weights and bias to zero 
        self.weights = np.zero((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        #convert integar labels to one hot encoded matrix for loss computation
        y_one_hot = self._one_hot(y_encoded, n_classes)

        # run gradient descent for the specified number of iterations
        for i in range(self.n_iterations):
            #compute raw scores
            z = np.dot(X, self.weights) + self.bias
            # convert raw scores to calss probabilites using softmax
            y_pred = self._softmax(z)

            #compute gradients shwoing how much to adjust weights
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1/n_samples) * np.sum(y_pred - y_one_hot, axis=0)

            #update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate* db

            #compute cross entropy loss
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))
            self.loss_history.append(loss)
        
        return self
    
    def predict(self, X):
        # compute raw scores using the learned weights
        z = np.dot(X, self.weights) + self.bias
        # convert to class probabilites
        y_pred = self._softmax(z)
        # select the class with the hightest probability for each sample
        y_idx = np.argmax(y_pred, axis=1)
        return np.array([self.calsses_[idx] for idx in y_idx])
    def score(self, X, y):
        #calculate accuracy as the fraction of correctly predicted labels
        return np.mean(self.predict(X)==y)
    
class GuassianNaiveBayers:
    def __init__(self):
        self.classes = None
        self.mean_ = {}
        self.var_ = {}
        self.prior_ = {}
    def fit(self, X,y):
        # get all unique emotion labels
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            #get all smaples belonging to this class
            X_cls = X[y==cls]
           # compute mean and varieance of each features for this class
            self.mean_[cls] = np.mean(X_cls, axis=0)
            self.var_[cls] = np.var(X_cls, axis=0)
            # compute prior probability (how common is this emotion in the dataset)
            self.prior_[cls] = len(X_cls) / len(y)
        return self
    def _gaussian_likelihood(self, cls, x):
        mean = self.mean_[cls]
        var = self.var_[cls]

        #compute the gaussian probability for each feature given the class
        numerator = np.exp(-((x - mean)**2)/ (2 * var + 1e-8))
        denominator = np.sqrt(2 * np.pi * var + 1e-8)
        return numerator/denominator
    
    def _compute_posterior(self, x):
        posteriors = {}

        for cls in self.classes_:
            #start with the log of the prior probability
            prior = np.log(self.prior_[cls])

            #add the log likelihood of each feature given this class
            likelihood = np.sum(np.log(self._gaussian_likelihood(cls,x) + 1e-8))

            #posterior = prior + likelihood in log space
            posteriors[cls] = prior + likelihood
        # retunr the class with the hgihtest posterior probability
        return max(posteriors, key=posteriors.get)
    def predict(self,X):
        #predict the emotoin for each sample
        return np.array([self._compute_posterior(x) for x in X])
    
    def score(self, X, y):
        #calculate accuracy ad fraction of correcct prediction
        return np.mean(self.predict(X)==y)
    
    

