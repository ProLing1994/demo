import numpy as np
import sklearn.metrics


def get_fpr_tpr(y_true, y_scores):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_scores, drop_intermediate=False)
    return fpr, tpr, thresholds

def get_fpr_tpr_auc(y_true, y_scores):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_scores, drop_intermediate=False)
    auc = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, thresholds, auc

def get_auc(fpr, tpr):
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

def get_confusion_matrix(y_true, y_pred):
    if y_true == y_pred:
        y_true = np.array(y_true)
        tn = len(y_true[y_true==0])
        fp = 0
        fn = 0
        tp = len(y_true[y_true==1])
        return tn, fp, fn, tp
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def get_accuracy(tn, fp, fn, tp):
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    return accuracy

def get_tpr(tn, fp, fn, tp):
    if (tp+fn) == 0:
        return 0
    tpr = tp/(tp+fn)
    return tpr

def get_fpr(tn, fp, fn, tp):
    if (fp+tn) == 0:
        return 0
    fpr = fp/(fp+tn)
    return fpr

def get_average_precision(y_true, y_pred):
    average_precision = sklearn.metrics.average_precision_score(y_true=y_true.flatten(), y_score=y_pred.flatten(), average="macro")
    return average_precision
    