from pathlib import Path
from src.data import Data
from src.LogisticRegression import LogisticRegression
from src.metrics import Metrics
from src.test import Test


MAIN_DIR = Path(__file__).resolve().parent

def main():
    data = Data(str(MAIN_DIR) + '/datasets/German_credit.csv')
    data.to_csv(str(MAIN_DIR) + '/datasets/credit')
    X_train, Y_train, X_test, Y_test = data.to_numpy()

    log_path = str(MAIN_DIR) + '/logs/'
    model = LogisticRegression(X_train.shape[1], log_path + 'fit_logs.log')
    model.fit(X_train, Y_train, epochs=1000, learning_rate=0.01, eps=0.2)
    Y_pred = model.predict(X_test)
    test = Test(Y_test, Y_pred, log_path + 'test_logs.log', eps=0.5)
    test.monitor()
    

if __name__ == '__main__':
    main()