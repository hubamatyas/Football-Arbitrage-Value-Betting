import pandas as pd
import numpy as np

class DataSet:
    X: pd.DataFrame = None
    y: pd.Series = None

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df_train = None
        self.df_test = None
        self.df_val = None

        self.test_columns = ['HomeTeam', 'AwayTeam', 'Date', 'B365H', 'B365A', 'B365D', 'PreHXG', 'PreAXG', 'RoundID', 'Season', 'GameWeek']

    def get_unique_teams(self) -> list[str]:
        home_teams = self.df['HomeTeam'].unique().tolist()
        away_teams = self.df['AwayTeam'].unique().tolist()
        return list(set(home_teams + away_teams))

    def split_data(self, validation=False, train_test_ratio=0.8, test_val_ratio=0.5) -> tuple[DataSet, DataSet, DataSet] | tuple[DataSet, DataSet]:
        if not validation:
            self.df_train, self.df_test = np.split(self.df, [int(train_test_ratio * len(self.df))])
            return self.get_train_data(), self.get_test_data()
        else:
            self.df_train, self.df_test = np.split(self.df, [int(train_test_ratio * len(self.df))])
            self.df_test, self.df_val = np.split(self.df_test, [int(test_val_ratio * len(self.df_test))])
            return self.get_train_data(), self.get_test_data(), self.get_val_data()

        
    def split_data_last_n(self, n=10) -> tuple[DataSet, DataSet]:
        self.df_train = self.df[:-n]
        self.df_test = self.df[-n:]

        return self.get_train_data(), self.get_test_data()
    
    def get_train_data(self) -> DataSet:
        train = DataSet()
        train.y = self.df_train['FTR']
        train.X = self.df_train

        return train
    
    def get_test_data(self) -> DataSet:
        test = DataSet()
        test.y = self.df_test['FTR']
        # make sure to avoid data leakage
        test.X = self.df_test[self.test_columns]
        return test
    
    def get_val_data(self) -> DataSet:
        val = DataSet()
        val.y = self.df_val['FTR']
        # make sure to avoid data leakage
        val.X = self.df_val[self.test_columns]

        return val