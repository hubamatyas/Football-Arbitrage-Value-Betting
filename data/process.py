import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_unique_teams(self) -> list[str]:
        home_teams = self.df['HomeTeam'].unique().tolist()
        away_teams = self.df['AwayTeam'].unique().tolist()
        return list(set(home_teams + away_teams))

    def split_data(self, validation=False, train_test_ratio=0.8, test_val_ratio=0.5) -> tuple:
        if not validation:
            df_train, df_test = np.split(self.df, [int(train_test_ratio * len(self.df))])
            return df_train, df_test
        else:
            df_train, df_test = np.split(self.df, [int(train_test_ratio * len(self.df))])
            df_test, df_val = np.split(df_test, [int(test_val_ratio * len(df_test))])
            return df_train, df_test, df_val
