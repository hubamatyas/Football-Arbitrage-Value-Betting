import pandas as pd
from category_encoders import OneHotEncoder
from sklearn import preprocessing

class XTableEncoder:
    def __init__(self, X_table: pd.DataFrame):
        self.X_table: pd.DataFrame = X_table

    def encode(self) -> pd.DataFrame:
        encoder = OneHotEncoder(cols=['HT', 'AT'], use_cat_names=True)
        X_encoded = encoder.fit_transform(self.X_table)
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
        self.X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)

    def normalise(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(self.X_encoded)
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.X_encoded.columns)

    def run(self):
        self.encode()
        self.normalise()
        return self.X_scaled
    

class YSeriesEncoder:
    def __init__(self, y_series: pd.Series):
        self.y_series: pd.Series = y_series

    def run(self) -> pd.Series:
        class_mapping = {'H': 0, 'A': 1, 'D': 2}
        return self.y_series.map(class_mapping)
    

class CrossChecker:
    def __init__(self, X_train_encoded: pd.DataFrame, X_test_encoded: pd.DataFrame):
        self.X_train_encoded: pd.DataFrame = X_train_encoded
        self.X_test_encoded: pd.DataFrame = X_test_encoded

    def run(self):
        diff1 = set(self.X_test_encoded.columns) - set(self.X_train_encoded.columns)
        diff2 = set(self.X_train_encoded.columns) - set(self.X_test_encoded.columns)

        for col in diff1:
            self.X_train_encoded[col] = 0
        for col in diff2:
            self.X_test_encoded[col] = 0

        # self.X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)
        self.X_train_encoded = self.X_train_encoded.reindex(sorted(self.X_train_encoded.columns), axis=1)
        self.X_test_encoded = self.X_test_encoded.reindex(sorted(self.X_test_encoded.columns), axis=1)
        return self.X_train_encoded, self.X_test_encoded