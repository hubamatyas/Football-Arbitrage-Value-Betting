import pandas as pd
from data.utils import Season
from datetime import datetime

class DataLoader:
    def __init__(self, path: str, season: Season):
        self.path: str = path
        self.season: datetime = season.value
        self.df = None

    def read_csv(self) -> pd.DataFrame:
        # Read and clean data. Convert unformatted date to appropriate string then datetime object
        df = pd.read_csv(self.path)
        df = df.dropna(how='all')
        self.df = df
    
    def parse_df(self) -> pd.DataFrame:
        for index, row in self.df.iterrows():
            date = row['Date'].split('/')
            if len(date[-1]) == 2:
                self.df.at[index, 'Date'] = f'{date[0]}/{date[1]}/20{date[-1]}'
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df = self.df.sort_values(by=['Date'])
    
    def select_seasons(self) -> pd.DataFrame:
        self.df = self.df[self.df['Date'] > self.season]

    def get_df(self) -> pd.DataFrame:
        return self.df
    
    def load(self) -> pd.DataFrame:
        self.read_csv()
        self.parse_df()
        self.select_seasons()
        return self.df