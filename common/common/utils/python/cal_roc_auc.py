import sklearn.metrics


def get_fpr_tpr(label, proba):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, proba, drop_intermediate=False)
    return fpr, tpr, thresholds