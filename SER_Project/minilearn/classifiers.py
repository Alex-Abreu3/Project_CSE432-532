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
        self.weights = np.zeros((n_features, n_classes))
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
        return np.array([self.classes_[idx] for idx in y_idx])
    def score(self, X, y):
        #calculate accuracy as the fraction of correctly predicted labels
        return np.mean(self.predict(X)==y)
    
class GaussianNaiveBayes:
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
    
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        #just memorizes the training set no learning
        self.X_train = X
        self.y_train = y
        return self
    def _euclidean_distance(self, x1, x2):
        # calculate the striaght line distance between two points
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict_single(self,x):
        #compute distance from this sample to every training sample
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        #sort distances and get the indices of the lk closest samples
        k_indices = np.argsort(distances)[:self.k]

        #get the emotion labels of the k nearest neighbors
        k_labels = [self.y_train[i] for i in k_indices]

        return max(set(k_labels), key=k_labels.count)
    def predict(self,X):
        #predict the emotion for every sample
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X,y):
        #calculate accuracy ad fraction of correct predictions
        return np.mean(self.predict(X)==y)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.classes_ = None
    def fit(self, X, y):
        #get all unique classes
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        # store one set of weights per class
        self.weights = np.zeros((len(self.classes_), n_features))
        self.bias = np.zeros(len(self.classes_))

        # train a binary classifier for each class against all others
        for idx,cls in enumerate(self.classes_):
            #convery lables to +1 for this class and -1 for all others
            y_binary = np.where(y==cls, 1,-1).astype(float)

            w = np.zeros(n_features)
            b = 0

            #graident decent loop
            for _ in range(self.n_iterations):
                for i, x_i in enumerate(X):
                    #check if this sample is corectly classifed with margin
                    if y_binary[i] * (np.dot(x_i, w) - b) >=1:
                        w -= self.learning_rate * (2 * self.lambda_param * w)
                    else:
                        #misclassifeied - updae weights and bias
                        w -= self.learning_rate * (2* self.lambda_param * w - np.dot(x_i, y_binary[i]))
                        b -= self.learning_rate * y_binary[i]
            self.weights[idx] = w
            self.bias[idx] = b
        return self
    def predict(self, X):
        #compute scores for each class
        scores = np.dot(X, self.weights.T) + self.bias
        #pick the calss with the hightest score of each sample
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]
    
    def score(self, X,y):
        #calculate accuracy as fraction of correct predictions
        return np.mean(self.predict(X)==y)

class DecisionTree:
    def __init__(self,max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
