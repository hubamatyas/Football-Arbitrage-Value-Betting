import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        home_teams = df['HomeTeam'].unique().tolist()
        away_teams = df['AwayTeam'].unique().tolist()
        self.unique_teams = list(set(home_teams + away_teams))