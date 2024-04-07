import pandas as pd

class FootyStatsCleaner:
    def __init__(self, original_df):
        self.columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
            'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR',
            'AR', 'HP', 'AP', 'B365H', 'B365D', 'B365A', 'PreHXG', 'PreAXG', 'Status'
            ]
        self.cleaned_df = pd.DataFrame(columns=self.columns)
        self.original_df: pd.DataFrame = original_df

        # df = df[df['status'] == 'incomplete']
        # df = df[df['date'] > '2024-04-06']

    def run(self):
        self.cleaned_df['Date'] = self.original_df['date']
        self.cleaned_df['HomeTeam'] = self.original_df['home_name']
        self.cleaned_df['AwayTeam'] = self.original_df['away_name']
        self.cleaned_df['FTHG'] = self.original_df['homeGoalCount']
        self.cleaned_df['FTAG'] = self.original_df['awayGoalCount']
        self.cleaned_df['FTR'] = self.original_df.apply(lambda row: 'H' if row['homeGoalCount'] > row['awayGoalCount'] else 'A' if row['homeGoalCount'] < row['awayGoalCount'] else 'D', axis=1)
        self.cleaned_df['HTHG'] = self.original_df['ht_goals_team_a']
        self.cleaned_df['HTAG'] = self.original_df['ht_goals_team_b']
        self.cleaned_df['HTR'] = self.original_df.apply(lambda row: 'H' if row['ht_goals_team_a'] > row['ht_goals_team_b'] else 'A' if row['ht_goals_team_a'] < row['ht_goals_team_b'] else 'D', axis=1)
        self.cleaned_df['HS'] = self.original_df['team_a_shots']
        self.cleaned_df['AS'] = self.original_df['team_b_shots']
        self.cleaned_df['HST'] = self.original_df['team_a_shotsOnTarget']
        self.cleaned_df['AST'] = self.original_df['team_b_shotsOnTarget']
        self.cleaned_df['HC'] = self.original_df['team_a_corners']
        self.cleaned_df['AC'] = self.original_df['team_b_corners']
        self.cleaned_df['HF'] = self.original_df['team_a_fouls']
        self.cleaned_df['AF'] = self.original_df['team_b_fouls']
        self.cleaned_df['HY'] = self.original_df['team_a_yellow_cards']
        self.cleaned_df['AY'] = self.original_df['team_b_yellow_cards']
        self.cleaned_df['HR'] = self.original_df['team_a_red_cards']
        self.cleaned_df['AR'] = self.original_df['team_b_red_cards']
        self.cleaned_df['HP'] = self.original_df['team_a_possession']
        self.cleaned_df['AP'] = self.original_df['team_b_possession']
        self.cleaned_df['B365H'] = self.original_df['odds_ft_1']
        self.cleaned_df['B365D'] = self.original_df['odds_ft_x']
        self.cleaned_df['B365A'] = self.original_df['odds_ft_2']
        self.cleaned_df['PreHXG'] = self.original_df['team_a_xg_prematch']
        self.cleaned_df['PreAXG'] = self.original_df['team_b_xg_prematch']
        self.cleaned_df['Status'] = self.original_df['status']

        return self.cleaned_df
