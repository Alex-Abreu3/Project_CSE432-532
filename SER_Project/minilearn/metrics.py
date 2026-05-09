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

def precision_score(y_true, y_pred, average='macro'):
    cm, classes = confusion_matrix(y_true, y_pred)

    precisions = []
    for i in range(len(classes)):
        #true postives for this class
        tp=cm[i,i]
        #all predicted as this class
        fp = np.sum(cm[:, i]) - tp
        precision = tp/ (tp+fp) if (tp+fp) > 0 else 0
        precisions.append(precision)
    if average == 'macro':
        return np.mean(precisions)
    return np.array(precisions)

def recall_score(y_true, y_pred, average='macro'):
    cm, classes = confusion_matrix(y_true, y_pred)

    recalls = []
    for i in range(len(classes)):
        #true postives for this class
        tp = cm[i,i]
        #actual for this class
        fn = np.sum(cm[i,:]) -tp
        recall = tp/(tp+fn) if (tp + fn) > 0 else 0
        recalls.append(recall)

    if average ==' macro':
        return np.mean(recalls)
    return np.array(recalls)

def f1_score(y_true, y_pred, average='marco'):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    f1_scores = []
    for p, r in zip(precision, recall):
        f1 = 2 * (p*r) / (p+r) if (p+r) > 0 else 0
        f1_scores.append(f1)
    if average == 'macro':
        return np.mean(f1_scores)
    return np.array(f1_scores)