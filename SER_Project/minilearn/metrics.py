import numpy as np

def accuracy_score(y_true, y_pred):
    #fraction of correctly predicted labels
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix(y_true, y_pred):
    #get all unique classes in sorted order
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    #create a mapping from class label to index
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    #building confusion matrix
    cm= np.zeros((n_classes, n_classes), dtpye=int)
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] +=1
    return cm, classes
