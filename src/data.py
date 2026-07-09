import pandas as pd
from pandas.api.types import CategoricalDtype


class Data:
    def __init__(self, url):
        self.df = pd.read_csv(url)
        self.df_train = None
        self.df_test = None
        self.columns_d = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing']
        self.columns_o = {
            'employment': ['unemployed', '<1 year', '1-4 years', '4-7 years', '>=7 years'],
            'job': ['unskilled', 'unskilled resident', 'skilled', 'highly skilled']
        }
        self.to_int = {
            'yes': 1,
            True: 1,
            'good': 1,
            'none': 0,
            'no': 0,
            False: 0,
            'bad': 0,
        }
        #RobustScaler (median, IQR)
        self.scale = {
            "duration": None,
            "credit_amount": None,
            "employment": None,
            "installment_commitment": None,
            "residence_since": None,
            "age": None,
            "existing_credits": None,
            "job": None,
            "num_dependents": None,
        }

        self._data_preprocess()

    def _data_preprocess(self):
        self.df = pd.get_dummies(self.df, columns=self.columns_d, drop_first=True)
        
        for col, categories in self.columns_o.items():
            dtype = CategoricalDtype(categories, ordered=True)
            self.df[col] = self.df[col].astype(dtype).cat.codes

        self.df.replace(self.to_int, inplace=True)
        self.df_train = self.df.sample(frac=0.7, random_state=12)
        self.df_test = self.df.drop(self.df_train.index)

        self.calc_scale()
        self.apply_scale(self.df_train)
        self.apply_scale(self.df_test)

        self.df_train = self.df_train.reset_index(drop=True)
        self.df_test = self.df_test.reset_index(drop=True)

    def calc_scale(self):
        for key in self.scale:
            temp = self.df_train[key]
            median = temp.median()
            iqr = temp.quantile(0.75) - temp.quantile(0.25)
            if iqr == 0:
                iqr = 1
            self.scale[key] = (median, iqr)

    def apply_scale(self, df):
        for key in self.scale:
            df[key] = (df[key] - self.scale[key][0]) / self.scale[key][1]

    def to_csv(self, name):
        self.df_train.to_csv(f'{name}_train.csv', index=False)
        self.df_test.to_csv(f'{name}_test.csv', index=False)

    def to_numpy(self):
        target = 'class'
        X_train = self.df_train.drop(columns=[target]).to_numpy()
        Y_train = self.df_train[target].to_numpy()
        X_test = self.df_test.drop(columns=[target]).to_numpy()
        Y_test = self.df_test[target].to_numpy()
        return X_train, Y_train, X_test, Y_test