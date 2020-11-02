import sklearn.metrics


def get_fpr_tpr(y_true, y_scores):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_scores, drop_intermediate=False)
    return fpr, tpr, thresholds


def get_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def get_accuracy(tn, fp, fn, tp):
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    return accuracy


def get_tpr(tn, fp, fn, tp):
    tpr = tp/(tp+fn)
    return tpr


def get_fpr(tn, fp, fn, tp):
    fpr = fp/(fp+tn)
    return fpr