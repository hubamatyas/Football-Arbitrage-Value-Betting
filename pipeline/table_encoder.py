import pandas as pd
from category_encoders import OneHotEncoder

class XTableEncoder:
    def __init__(self, X_table: pd.DataFrame):
        self.X_table: pd.DataFrame = X_table

    def encode(self):
        encoder = OneHotEncoder(cols=['HT', 'AT'], use_cat_names=True)
        X_encoded = encoder.fit_transform(self.X_table)
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
        return X_encoded.reindex(sorted(X_encoded.columns), axis=1)
    

class YSeriesEncoder:
    def __init__(self, y_series: pd.Series):
        self.y_series: pd.Series = y_series

    def encode(self):
        class_mapping = {'H': 0, 'A': 1, 'D': 2}
        return self.y_series.map(class_mapping)