from .metrics import *


class Test:
    def __init__(self, y_true, y_pred, log_path, eps=0.5):
        self.log_path = log_path
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.eps = eps
        self._calc()
    
    def _calc(self):
        for x, y in zip(self.y_pred, self.y_true):
            if x >= self.eps:
                if y == 1:
                    self.tp += 1
                else:
                    self.tn += 1
            else:
                if y == 1:
                    self.fp += 1
                else:
                    self.fn += 1
        
    def monitor(self):
        message = ''
        message += f'LogLoss: {Metrics.log_loss(self.y_pred, self.y_true)}\n'
        message += f'AUC_ROC: {Metrics.auc_roc(self.y_pred, self.y_true)}\n'
        message += f'Accuracy: {Metrics.accuracy(self.tp, self.tn, self.fp, self.fn)}\n'
        message += f'Precision: {Metrics.precision(self.tp, self.fp)}\n'
        message += f'Recall: {Metrics.recall(self.tp, self.fn)}\n'
        message += f'F1: {Metrics.f1(self.tp, self.fp, self.fn)}\n'
        self._save_logs(message)

    def _save_logs(self, log):
        with open(self.log_path, 'w') as file:
            file.write(''.join(log))