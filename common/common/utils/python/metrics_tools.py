import numpy as np
import sklearn.metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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

# 精度 = precision = PPV(positive predictive value)
def get_ppv(tn, fp, fn, tp):
    if (tp+fp) == 0:
        return 0
    ppv = tp/(tp+fp)
    return ppv

def get_average_precision(y_true, y_pred, average=None):
    average_precision = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_pred, average=average)
    return average_precision

def get_roc_auc(y_true, y_pred, average=None):
    auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average=average)
    return auc

def get_precision_recall(y_true, y_pred):
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    return precisions, recalls, thresholds

def get_eer(y_true, y_pred):
    # Snippet from https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)           
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

if __name__ == "__main__":
    y_true = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]])
    y_pred = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    print(get_average_precision(y_true, y_pred))
    print(get_average_precision(y_true, y_pred, 'macro'))

    print(get_average_precision(y_true[:, 0], y_pred[:, 0]))
    print(get_average_precision(y_true[:, 1], y_pred[:, 1]))
    print(get_average_precision(y_true[:, 2], y_pred[:, 2]))
    print(get_average_precision(y_true[:, 3], y_pred[:, 3]))