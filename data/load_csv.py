import pandas as pd
from data.utils import Season

class DataLoader:
    def __init__(self, path: str, season: Season):
        self.path = path
        self.season = season.value
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
    
    def select_seasons(self, season: Season) -> pd.DataFrame:
        self.df = self.df[self.df['Date'] > season.value]

    def get_df(self) -> pd.DataFrame:
        return self.df