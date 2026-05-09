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
    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None

    def _gini(self, y):
        #compute gini impurity, measure how mixed the classes are
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)
    
    def _best_split(self,X,y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        #try every feature and every possible threshold
        for feature in range(X.shape[1]):
            col = X[:, feature]
            #sample 10 evenly spaced sheshold instead of every unqiue value
            thresholds = np.linspace(col.min(), col.max(), 10)

            for threshold in thresholds:
                #split data into left and right based on threshold
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask)==0 or sum(right_mask)==0:
                    continue
                
                #compute weighted gini impurity of the split
                n = len(y)
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (sum(left_mask)/n * left_gini + sum(right_mask)/n * right_gini)

                #keep track of the best split found so far
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        return  int(best_feature) if best_feature is not None else None, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        node = self.Node()
        # stop growing if max depth reached or not enough samples
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node.value = max(set(y.tolist()), key=y.tolist().count)
            return node
        #find the best split for this node
        feature, threshold = self._best_split(X,y)

        if feature is None:
            node.value = max(set(y.tolist()), key=y.tolist().count)
            return node
        
        #split the data and recursively build left and right subtrees
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.feature=feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth +1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth+1)

        return node
    
    def fit(self, X, y):
        #build the tree starting from root
        self.root = self._build_tree(X,y)
        return self
    
    def _predict_single(self, x, node):
        # if the leaf node returned the predicted class
        if node.value is not None:
            return node.value

        #otherwise traves left or right based of feature value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        #predict the class for every sample
        return np.array([self._predict_single(x,self.root) for x in X])
    
    def score(self,X,y):
        #calculare accuracy as fraction of correct predictions
        return np.mean(self.predict(X)==y)

class KMeans:
    def __init__(self, n_clusters=8, max_iterations=300, random_state=None):
        self.n_clusters= n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    
    def fit(self,X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        #randomly intialize centroids by picking k random samples
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx].copy()

        for iteration in range(self.max_iterations):
            #assign each sample to the nearest centroid
            self.labels_ = self._assign_clusters(X)

            #save old centroids to check for convergence
            old_centroids = self.centroids.copy()

            #update centroids to the mean of all samples in eahc cluster
            for k in range(self.n_clusters):
                cluster_samples = X[self.labels_ ==k]
                if len(cluster_samples) >0:
                    self.centroids[k] = np.mean(cluster_samples, axis=0)
                
            #stops if centroid are converged
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration +1}")
                break
        return self
    
    def _assign_clusters(self, X):
        #compute distance from each sample to every centroid
        distances = np.array([np.sqrt(np.sum((X-centroid)**2, axis=1)) for centroid in self.centroids])
        #assign each sample to the nearest centroid
        return np.argmin(distances, axis=0)
    
    def predict(self,X):
        #assign new samples to nearest Centriod
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        #fit and return cluster labels in one step
        self.fit(X)
        return self.labels_