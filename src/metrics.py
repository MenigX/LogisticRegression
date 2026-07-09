import numpy as np

class Metrics:
    @staticmethod
    def auc_roc(y_proba, y_true):
        idx = np.argsort(y_proba)[::-1]
        sorted_labels = y_true[idx]

        tpr = 0
        auc = 0
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        for label in sorted_labels:
            if label == 1:
                tpr += 1
            else:
                auc += tpr
        if P == 0 or N == 0:
            return 0.5
        auc /= (P * N)
        return auc
    
    @staticmethod
    def accuracy(tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)
    
    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp)
    
    @staticmethod
    def recall(tp, fn):
        return tp / (tp + fn)
    
    @staticmethod
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)
    
    @staticmethod
    def log_loss(Y_pred, Y_true, eps=1e-15):
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        return -(Y_true @ np.log(Y_pred) + (1 - Y_true) @ np.log(1 - Y_pred)) / Y_true.shape[0]
    
    @staticmethod
    def s_log_loss(pred, true):
        return -(true * np.log(pred) + (1 - true) * np.log(pred))