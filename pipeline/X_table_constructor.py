import pandas as pd
import json

class XTableConstructor:
    def __init__(self,individual_stats, pairwise_stats, pi_ratings=None, pi_pairwise=None, pi_weighted=None):
        self.individual_stats: pd.DataFrame = individual_stats
        self.pairwise_stats: pd.DataFrame = pairwise_stats
        self.pi_ratings: pd.DataFrame = pi_ratings
        self.pi_pairwise: pd.DataFrame = pi_pairwise
        self.pi_weighted: pd.DataFrame = pi_weighted
        self.X_table: pd.DataFrame = pd.DataFrame()
    
    def construct_table(self) -> pd.DataFrame:
        for _, row in self.pairwise_stats.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            home_team_stats = self.individual_stats.loc[self.individual_stats['Team'] == home_team]
            away_team_stats = self.individual_stats.loc[self.individual_stats['Team'] == away_team]

            home_team_pairwise = self.pairwise_stats.loc[(self.pairwise_stats['HomeTeam'] == home_team) & (self.pairwise_stats['AwayTeam'] == away_team)]
            away_team_pairwise = self.pairwise_stats.loc[(self.pairwise_stats['HomeTeam'] == away_team) & (self.pairwise_stats['AwayTeam'] == home_team)]

            home_team_pi = self.pi_ratings.loc[self.pi_ratings['Team'] == home_team]
            away_team_pi = self.pi_ratings.loc[self.pi_ratings['Team'] == away_team]

            home_team_pairwise_pi = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == home_team) & (self.pi_pairwise['AwayTeam'] == away_team)]
            away_team_pairwise_pi = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == away_team) & (self.pi_pairwise['AwayTeam'] == home_team)]

            home_team_weighted_pi = self.pi_weighted.loc[self.pi_weighted['HomeTeam'] == home_team]
            away_team_weighted_pi = self.pi_weighted.loc[self.pi_weighted['AwayTeam'] == away_team]

            row: dict = self.construct_row(home_team, away_team, home_team_stats, away_team_stats, home_team_pairwise, away_team_pairwise, home_team_pi, away_team_pi, home_team_pairwise_pi, away_team_pairwise_pi, home_team_weighted_pi, away_team_weighted_pi)
            self.X_table = self.X_table._append(row, ignore_index=True)

        return self.X_table
    
    def construct_row(self, home_team, away_team, home_team_stats, away_team_stats, home_team_pairwise, away_team_pairwise, home_team_pi, away_team_pi, home_team_pairwise_pi, away_team_pairwise_pi, home_team_weighted_pi, away_team_weighted_pi) -> dict:
        row = {
            'HT': home_team,
            'AT': away_team,
        }

        row = self.add_result_percentage(row, home_team_stats, away_team_stats)
        row = self.add_home_away_result_percentage(row, home_team_stats, away_team_stats)
        row = self.add_shooting_stats(row, home_team_stats, away_team_stats)
        row = self.add_goal_stats(row, home_team_stats, away_team_stats)
        row = self.add_conceded_stats(row, home_team_stats, away_team_stats)
        row = self.add_last_n_matches_stats(row, home_team_stats, away_team_stats)

        if self.pi_ratings is not None:
            row = self.add_pi_ratings(row, home_team_pi, away_team_pi)

        if self.pi_pairwise is not None:
            row = self.add_pi_pairwise(row, home_team_pairwise_pi, away_team_pairwise_pi)

        if self.pi_weighted is not None:
            row = self.add_pi_weighted(row, home_team_weighted_pi, away_team_weighted_pi)

        return row
    
    def add_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_Win%'] = home_team_stats['Wins'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_Win%'] = away_team_stats['Wins'].values[0] / away_team_stats['NumOfMatches'].values[0]
        row['HT_Draw%'] = home_team_stats['Draws'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_Draw%'] = away_team_stats['Draws'].values[0] / away_team_stats['NumOfMatches'].values[0]
        row['HT_Loss%'] = home_team_stats['Losses'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_Loss%'] = away_team_stats['Losses'].values[0] / away_team_stats['NumOfMatches'].values[0]

        return row
    
    def add_home_away_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_HomeWin%'] = home_team_stats['HomeWins'].values[0] / home_team_stats['NumOfHomeMatches'].values[0]
        row['AT_AwayWin%'] = away_team_stats['AwayWins'].values[0] / away_team_stats['NumOfAwayMatches'].values[0]
        row['HT_HomeDraw%'] = home_team_stats['HomeDraws'].values[0] / home_team_stats['NumOfHomeMatches'].values[0]
        row['AT_AwayDraw%'] = away_team_stats['AwayDraws'].values[0] / away_team_stats['NumOfAwayMatches'].values[0]
        row['HT_HomeLoss%'] = home_team_stats['HomeLosses'].values[0] / home_team_stats['NumOfHomeMatches'].values[0]
        row['AT_AwayLoss%'] = away_team_stats['AwayLosses'].values[0] / away_team_stats['NumOfAwayMatches'].values[0]

        return row
    
    def add_shooting_stats(self, row, home_team_stats, away_team_stats):
        row['HT_ShotOnGoalPerMatch'] = home_team_stats['ShotsOnGoal'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_ShotOnGoalPerMatch'] = away_team_stats['ShotsOnGoal'].values[0] / away_team_stats['NumOfMatches'].values[0]
        row['HT_ShotOnTargetPerMatch'] = home_team_stats['ShotsOnTarget'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_ShotOnTargetPerMatch'] = away_team_stats['ShotsOnTarget'].values[0] / away_team_stats['NumOfMatches'].values[0]
        row['HT_ShotOnTargetAccuracy'] = home_team_stats['ShotsOnTarget'].values[0] / home_team_stats['ShotsOnGoal'].values[0]
        row['AT_ShotOnTargetAccuracy'] = away_team_stats['ShotsOnTarget'].values[0] / away_team_stats['ShotsOnGoal'].values[0]

        return row
    
    def add_goal_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalAccuracy'] = home_team_stats['Goals'].values[0] / home_team_stats['ShotsOnTarget'].values[0]
        row['AT_GoalAccuracy'] = away_team_stats['Goals'].values[0] / away_team_stats['ShotsOnTarget'].values[0]
        row['HT_GoalsPerMatch'] = home_team_stats['Goals'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_GoalsPerMatch'] = away_team_stats['Goals'].values[0] / away_team_stats['NumOfMatches'].values[0]
        row['HT_HalfTimeGoalsPerMatch'] = home_team_stats['HalfTimeGoals'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_HalfTimeGoalsPerMatch'] = away_team_stats['HalfTimeGoals'].values[0] / away_team_stats['NumOfMatches'].values[0]

        return row
    
    def add_conceded_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsConcededPerMatch'] = home_team_stats['Conceded'].values[0] / home_team_stats['NumOfMatches'].values[0]
        row['AT_GoalsConcededPerMatch'] = away_team_stats['Conceded'].values[0] / away_team_stats['NumOfMatches'].values[0]

        return row
    
    def add_last_n_matches_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsLastNMatches'] = home_team_stats['GoalsLastNMatches'].values[0]
        row['AT_GoalsLastNMatches'] = away_team_stats['GoalsLastNMatches'].values[0]
        row['HT_GoalDiffLastNMatches'] = home_team_stats['GoalDiffLastNMatches'].values[0]
        row['AT_GoalDiffLastNMatches'] = away_team_stats['GoalDiffLastNMatches'].values[0]

        return row
    
    def add_pi_ratings(self, row, home_team_pi, away_team_pi):
        row['HT_HomeRating'] = home_team_pi['HomeRating'].values[0]
        row['HT_AwayRating'] = home_team_pi['AwayRating'].values[0]
        row['AT_HomeRating'] = away_team_pi['HomeRating'].values[0]
        row['AT_AwayRating'] = away_team_pi['AwayRating'].values[0]

        return row
    
    def add_pi_pairwise(self, row, home_team_pairwise_pi, away_team_pairwise_pi):
        row['PWHT_HomeRating'] = home_team_pairwise_pi['HomeRating'].values[0]
        row['PWHT_AwayRating'] = home_team_pairwise_pi['AwayRating'].values[0]
        row['PWAT_HomeRating'] = away_team_pairwise_pi['HomeRating'].values[0]
        row['PWAT_AwayRating'] = away_team_pairwise_pi['AwayRating'].values[0]

        return row
    
    def add_pi_weighted(self, row, home_team_weighted_pi, away_team_weighted_pi):
        row['WPWHT_HomeRating'] = home_team_weighted_pi['HomeRating'].values[0]
        row['WPWHT_AwayRating'] = home_team_weighted_pi['AwayRating'].values[0]
        row['WPWAT_HomeRating'] = away_team_weighted_pi['HomeRating'].values[0]
        row['WPWAT_AwayRating'] = away_team_weighted_pi['AwayRating'].values[0]

        return row
        