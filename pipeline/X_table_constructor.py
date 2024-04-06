import pandas as pd
from features.pi_rating import PiRatingsCalculator, PiRatingsManager
from features.individual_stats import IndividualTeamStats
from features.pairwise_stats import PairwiseTeamStats
import json

class XTableConstructor:
    def __init__(self, is_pairwise_stats=False, is_pi_ratings=True, is_pi_pairwise=True, is_pi_weighted=True):
        self.is_pairwise_stats: bool = is_pairwise_stats
        self.is_pi_ratings: bool = is_pi_ratings
        self.is_pi_pairwise: bool = is_pi_pairwise
        self.is_pi_weighted: bool = is_pi_weighted

    def construct_row(self, home_team, away_team, home_team_stats, away_team_stats, home_team_pi, away_team_pi, home_team_pairwise_pi, away_team_pairwise_pi, home_team_weighted_pi, away_team_weighted_pi) -> dict:
        row = {
            'HT': home_team,
            'AT': away_team,
        }

        # row = self.add_result_percentage(row, home_team_stats, away_team_stats)
        # row = self.add_home_away_result_percentage(row, home_team_stats, away_team_stats)
        # row = self.add_shooting_stats(row, home_team_stats, away_team_stats)
        # row = self.add_goal_stats(row, home_team_stats, away_team_stats)
        # row = self.add_conceded_stats(row, home_team_stats, away_team_stats)
        # row = self.add_last_n_matches_stats(row, home_team_stats, away_team_stats)

        if self.is_pi_ratings:
            row = self.add_pi_ratings(row, home_team_pi, away_team_pi)

        if self.is_pi_pairwise:
            row = self.add_pi_pairwise(row, home_team_pairwise_pi, away_team_pairwise_pi)

        if self.is_pi_weighted:
            row = self.add_pi_weighted(row, home_team_weighted_pi, away_team_weighted_pi)

        return row

    def divide(self, x_dict, y_dict, x_label, y_label):
        return x_dict[x_label].values[0] / y_dict[y_label].values[0] if y_dict[y_label].values[0] != 0 else 0
    
    def get_value(self, df, column):
        return df[column].values[0] if not df.empty else 0

    def add_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_Win%'] = self.divide(home_team_stats, home_team_stats, 'Wins', 'NumOfMatches')
        row['AT_Win%'] = self.divide(away_team_stats, away_team_stats, 'Wins', 'NumOfMatches')
        row['HT_Draw%'] = self.divide(home_team_stats, home_team_stats, 'Draws', 'NumOfMatches')
        row['AT_Draw%'] = self.divide(away_team_stats, away_team_stats, 'Draws', 'NumOfMatches')
        row['HT_Loss%'] = self.divide(home_team_stats, home_team_stats, 'Losses', 'NumOfMatches')
        row['AT_Loss%'] = self.divide(away_team_stats, away_team_stats, 'Losses', 'NumOfMatches')

        return row
    
    def add_home_away_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_HomeWin%'] = self.divide(home_team_stats, home_team_stats, 'HomeWins', 'NumOfHomeMatches')
        row['AT_AwayWin%'] = self.divide(away_team_stats, away_team_stats, 'AwayWins', 'NumOfAwayMatches')
        row['HT_HomeDraw%'] = self.divide(home_team_stats, home_team_stats, 'HomeDraws', 'NumOfHomeMatches')
        row['AT_AwayDraw%'] = self.divide(away_team_stats, away_team_stats, 'AwayDraws', 'NumOfAwayMatches')
        row['HT_HomeLoss%'] = self.divide(home_team_stats, home_team_stats, 'HomeLosses', 'NumOfHomeMatches')
        row['AT_AwayLoss%'] = self.divide(away_team_stats, away_team_stats, 'AwayLosses', 'NumOfAwayMatches')

        return row
    
    def add_shooting_stats(self, row, home_team_stats, away_team_stats):
        row['HT_ShotOnGoalPerMatch'] = self.divide(home_team_stats, home_team_stats, 'ShotsOnGoal', 'NumOfMatches')
        row['AT_ShotOnGoalPerMatch'] = self.divide(away_team_stats, away_team_stats, 'ShotsOnGoal', 'NumOfMatches')
        row['HT_ShotOnTargetPerMatch'] = self.divide(home_team_stats, home_team_stats, 'ShotsOnTarget', 'NumOfMatches')
        row['AT_ShotOnTargetPerMatch'] = self.divide(away_team_stats, away_team_stats, 'ShotsOnTarget', 'NumOfMatches')
        row['HT_ShotOnTargetAccuracy'] = self.divide(home_team_stats, home_team_stats, 'ShotsOnTarget', 'ShotsOnGoal')
        row['AT_ShotOnTargetAccuracy'] = self.divide(away_team_stats, away_team_stats, 'ShotsOnTarget', 'ShotsOnGoal')

        return row
    
    def add_goal_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalAccuracy'] = self.divide(home_team_stats, home_team_stats, 'Goals', 'ShotsOnTarget')
        row['AT_GoalAccuracy'] = self.divide(away_team_stats, away_team_stats, 'Goals', 'ShotsOnTarget')
        row['HT_GoalsPerMatch'] = self.divide(home_team_stats, home_team_stats, 'Goals', 'NumOfMatches')
        row['AT_GoalsPerMatch'] = self.divide(away_team_stats, away_team_stats, 'Goals', 'NumOfMatches')
        row['HT_HalfTimeGoalsPerMatch'] = self.divide(home_team_stats, home_team_stats, 'HalfTimeGoals', 'NumOfMatches')
        row['AT_HalfTimeGoalsPerMatch'] = self.divide(away_team_stats, away_team_stats, 'HalfTimeGoals', 'NumOfMatches')

        return row
    
    def add_conceded_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsConcededPerMatch'] = self.divide(home_team_stats, home_team_stats, 'Conceded', 'NumOfMatches')
        row['AT_GoalsConcededPerMatch'] = self.divide(away_team_stats, away_team_stats, 'Conceded', 'NumOfMatches')

        return row
    
    def add_last_n_matches_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsLastNMatches'] = home_team_stats['GoalsLastNMatches'].values[0]
        row['AT_GoalsLastNMatches'] = away_team_stats['GoalsLastNMatches'].values[0]
        row['HT_GoalDiffLastNMatches'] = home_team_stats['GoalDiffLastNMatches'].values[0]
        row['AT_GoalDiffLastNMatches'] = away_team_stats['GoalDiffLastNMatches'].values[0]

        return row
    
    def add_pi_ratings(self, row, home_team_pi, away_team_pi):
        row['HT_HomeRating'] = self.get_value(home_team_pi, 'HomeRating')
        row['HT_AwayRating'] = self.get_value(home_team_pi, 'AwayRating')
        row['AT_HomeRating'] = self.get_value(away_team_pi, 'HomeRating')
        row['AT_AwayRating'] = self.get_value(away_team_pi, 'AwayRating')

        return row
    
    def add_pi_pairwise(self, row, home_team_pairwise_pi, away_team_pairwise_pi):
        row['PWHT_HomeRating'] = self.get_value(home_team_pairwise_pi, 'HomeRating')
        row['PWHT_AwayRating'] = self.get_value(home_team_pairwise_pi, 'AwayRating')
        row['PWAT_HomeRating'] = self.get_value(away_team_pairwise_pi, 'HomeRating')
        row['PWAT_AwayRating'] = self.get_value(away_team_pairwise_pi, 'AwayRating')

        return row
    
    def add_pi_weighted(self, row, home_team_weighted_pi, away_team_weighted_pi):
        row['WPWHT_HomeRating'] = self.get_value(home_team_weighted_pi, 'HomeRating')
        row['WPWHT_AwayRating'] = self.get_value(home_team_weighted_pi, 'AwayRating')
        row['WPWAT_HomeRating'] = self.get_value(away_team_weighted_pi, 'HomeRating')
        row['WPWAT_AwayRating'] = self.get_value(away_team_weighted_pi, 'AwayRating')

        return row

class XTestConstructor(XTableConstructor):
    def __init__(self, df, df_train, unique_teams, is_pairwise_stats=False, is_pi_ratings=True, is_pi_pairwise=True, is_pi_weighted=True):
        super().__init__(is_pairwise_stats, is_pi_ratings, is_pi_pairwise, is_pi_weighted)
        self.df: pd.DataFrame = df
        self.df_train: pd.DataFrame = df_train
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.unique_teams: list[str] = unique_teams

        self.pi_ratings, self.pi_pairwise, self.pi_weighted = PiRatingsManager(self.df_train, self.unique_teams).compute()
        self.individual_stats = IndividualTeamStats(self.df_train, self.unique_teams).compute()
        # self.pairwise_stats = PairwiseTeamStats(self.df, self.unique_teams, self.individual_stats).compute()

    def construct_table(self) -> pd.DataFrame:
        for _, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            home_team_stats = self.individual_stats.loc[self.individual_stats['Team'] == home_team]
            away_team_stats = self.individual_stats.loc[self.individual_stats['Team'] == away_team]

            home_team_pi = self.pi_ratings.loc[self.pi_ratings['Team'] == home_team]
            away_team_pi = self.pi_ratings.loc[self.pi_ratings['Team'] == away_team]

            home_team_pairwise_pi = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == home_team) & (self.pi_pairwise['AwayTeam'] == away_team)]
            away_team_pairwise_pi = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == away_team) & (self.pi_pairwise['AwayTeam'] == home_team)]

            home_team_weighted_pi = self.pi_weighted.loc[self.pi_weighted['HomeTeam'] == home_team]
            away_team_weighted_pi = self.pi_weighted.loc[self.pi_weighted['AwayTeam'] == away_team]

            row: dict = self.construct_row(home_team, away_team, home_team_stats, away_team_stats, home_team_pi, away_team_pi, home_team_pairwise_pi, away_team_pairwise_pi, home_team_weighted_pi, away_team_weighted_pi)
            self.X_test = self.X_test._append(row, ignore_index=True)

        return self.X_test

class XTrainConstructor(XTableConstructor):
    def __init__(self, df, unique_teams, is_pairwise_stats=False, is_pi_ratings=True, is_pi_pairwise=True, is_pi_weighted=True):
        super().__init__(is_pairwise_stats, is_pi_ratings, is_pi_pairwise, is_pi_weighted)
        
        self.df: pd.DataFrame = df
        self.is_pi_ratings: bool = is_pi_ratings
        self.is_pi_pairwise: bool = is_pi_pairwise
        self.is_pi_weighted: bool = is_pi_weighted
        self.unique_teams: list[str] = unique_teams
        self.is_pairwise_stats: bool = is_pairwise_stats
        self.X_train: pd.DataFrame = pd.DataFrame()
    
    def construct_table(self) -> pd.DataFrame:
        individual_stats_manager = IndividualTeamStats(self.df, self.unique_teams)
        # pairwise_stats_manager = PairwiseTeamStats(self.df, self.unique_teams, individual_stats_manager.generate_features_dataframe())

        pi_ratings_manager = PiRatingsManager(self.df, self.unique_teams)
        pi_ratings_calculator = PiRatingsCalculator()
        for _, row in self.df.iterrows():

            # Iterative implementation
            pi_ratings, pi_pairwise, pi_weighted = self.update_pi_ratings(row, pi_ratings_manager, pi_ratings_calculator)
            individual_stats = self.update_individual_stats(row, individual_stats_manager)
            
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            home_team_stats = individual_stats.loc[individual_stats['Team'] == home_team]
            away_team_stats = individual_stats.loc[individual_stats['Team'] == away_team]

            home_team_pi = pi_ratings.loc[pi_ratings['Team'] == home_team] if self.is_pi_ratings else None
            away_team_pi = pi_ratings.loc[pi_ratings['Team'] == away_team] if self.is_pi_ratings else None

            home_team_pairwise_pi = pi_pairwise.loc[(pi_pairwise['HomeTeam'] == home_team) & (pi_pairwise['AwayTeam'] == away_team)] if self.is_pi_pairwise else None
            away_team_pairwise_pi = pi_pairwise.loc[(pi_pairwise['HomeTeam'] == away_team) & (pi_pairwise['AwayTeam'] == home_team)] if self.is_pi_pairwise else None

            home_team_weighted_pi = pi_weighted.loc[pi_weighted['HomeTeam'] == home_team] if self.is_pi_weighted else None
            away_team_weighted_pi = pi_weighted.loc[pi_weighted['AwayTeam'] == away_team] if self.is_pi_weighted else None

            row: dict = self.construct_row(home_team, away_team, home_team_stats, away_team_stats, home_team_pi, away_team_pi, home_team_pairwise_pi, away_team_pairwise_pi, home_team_weighted_pi, away_team_weighted_pi)
            self.X_train = self.X_train._append(row, ignore_index=True)

        return self.X_train
    
    def update_pi_ratings(self, row, pi_ratings_manager: PiRatingsManager, pi_ratings_calculator: PiRatingsCalculator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pi_ratings_manager.update_pi_ratings(row, pi_ratings_calculator)
        pi_ratings_manager.update_pi_pairwise(row, pi_ratings_calculator)
        pi_ratings_manager.update_pi_weighted(row)

        return pi_ratings_manager.get_pi_ratings(), pi_ratings_manager.get_pi_pairwise(), pi_ratings_manager.get_pi_weighted()
    
    def update_individual_stats(self, row, individual_stats_manager: IndividualTeamStats) -> pd.DataFrame:
        individual_stats_manager.compute_team_stats(row)
        return individual_stats_manager.generate_features_dataframe()