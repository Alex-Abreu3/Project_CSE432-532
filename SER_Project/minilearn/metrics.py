import numpy as np

def accuracy_score(y_true, y_pred):
    #fraction of correctly predicted labels
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix(y_true, y_pred):
    #get all unique classes in sorted order
    classes = np.unique(np.concatenate([y_true, y_pred]))
    b_classes = len(classes)

