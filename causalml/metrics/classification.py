import logging
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, f1_score

from .const import EPS
from .regression import regression_metrics


logger = logging.getLogger("causalml")


def logloss(y, p):
    """Bounded log loss error.
    Args:
        y (numpy.array): target
        p (numpy.array): prediction
    Returns:
        bounded log loss error
    """

    p[p < EPS] = EPS
    p[p > 1 - EPS] = 1 - EPS
    return log_loss(y, p)


def classification_metrics(
    y, p, w=None, metrics={"AUC": roc_auc_score, "Log Loss": logloss}
):
    """Log metrics for classifiers.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction
        w (numpy.array, optional): a treatment vector (1 or True: treatment, 0 or False: control). If given, log
            metrics for the treatment and control group separately
        metrics (dict, optional): a dictionary of the metric names and functions
    """
    regression_metrics(y=y, p=p, w=w, metrics=metrics)


def custom_f1_score(y_pred, y_true_dmatrix):
    """
    the signature is func(y_predicted, DMatrix_y_true) where DMatrix_y_true is
    a DMatrix object such that you may need to call the get_label method.
    It must return a (str, value) pair where the str is a name for the evaluation
    and value is the value of the evaluation function.
    The callable function is always minimized.
    :param y_pred: np.array, probability score predicted by the xgbclassifier
    :param y_true_dmatrix: xgb DMatrix, true label, with positive instances as 1
    """
    y_true = y_true_dmatrix.get_label()
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return 'f1', f1


def custom_ks_score(y_pred, y_true_dmatrix):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    y_true = y_true_dmatrix.get_label()
    ks = ks_stats(y_true, y_pred)
    return 'ks', ks

