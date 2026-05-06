from pathlib import Path
from src.data import Data
from src.LogisticRegression import LogisticRegression
from src.metrics import Metrics


MAIN_DIR = Path(__file__).resolve().parent

def main():
    data = Data(str(MAIN_DIR) + '/datasets/German_credit.csv')
    data.to_csv(str(MAIN_DIR) + '/datasets/credit')
    X_train, Y_train, X_test, Y_test = data.to_numpy()

    log_path = str(MAIN_DIR) + '/logs/fit_logs.log'
    model = LogisticRegression(X_train.shape[1], log_path)
    model.fit(X_train, Y_train, epochs=1000)
    Y_pred = model.predict(X_test)

    print(f'LogLoss: {Metrics.log_loss(Y_pred, Y_test)}')
    print(f'AUC_ROC: {Metrics.auc_roc(Y_pred, Y_test)}')

if __name__ == '__main__':
    main()