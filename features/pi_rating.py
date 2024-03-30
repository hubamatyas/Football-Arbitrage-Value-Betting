import math
from typing import Tuple
import pandas as pd

class PiRatingsCalculator:
    C = 3
    LOG_BASE = 10
    LAMBDA = 0.156
    GAMMA = 0.739

    def __init__(self):
        self.decay_factor = None

    def set_decay_factor(self, decay_factor):
        self.decay_factor = decay_factor

    def calc_rating_update(self, home_rating, away_rating, ft_home_goals, ft_away_goals):
        home_sign = 1 if home_rating >= 0 else -1
        away_sign = 1 if away_rating >= 0 else -1

        exp_home_goal_diff = home_sign * (self.LOG_BASE ** (abs(home_rating) / self.C) - 1)
        exp_away_goal_diff = away_sign * (self.LOG_BASE ** (abs(away_rating) / self.C) - 1)
        
        exp_goal_diff = exp_home_goal_diff - exp_away_goal_diff
        act_goal_diff = ft_home_goals - ft_away_goals
        
        error = abs(act_goal_diff - exp_goal_diff)

        psi_H_e, psi_A_e = self.calculate_psi(error, act_goal_diff > exp_goal_diff)
        return psi_H_e * self.LAMBDA, psi_A_e * self.LAMBDA

    def calculate_psi(self, error, condition):
        psi_e = self.C * math.log10(error + 1)
        if condition:
            return psi_e * self.decay_factor, -psi_e * self.decay_factor
        else:
            return -psi_e * self.decay_factor, psi_e * self.decay_factor

    def update_team_ratings(self, home_team_ratings: pd.DataFrame, away_team_ratings: pd.DataFrame, ft_home_goals, ft_away_goals) -> Tuple[pd.DataFrame, pd.DataFrame]:
        psi_H_e, psi_A_e = self.calc_rating_update(home_team_ratings['HomeRating'].values[0], away_team_ratings['AwayRating'].values[0], ft_home_goals, ft_away_goals)
       
        home_team_ratings['HomeRating'] += psi_H_e
        away_team_ratings['AwayRating'] += psi_A_e
        home_team_ratings['AwayRating'] += psi_H_e * self.GAMMA
        away_team_ratings['HomeRating'] += psi_A_e * self.GAMMA
        
        return home_team_ratings, away_team_ratings

    def update_team_pair_ratings(self, home_team_ratings: pd.DataFrame, away_team_ratings: pd.DataFrame, ft_home_goals, ft_away_goals) -> Tuple[pd.DataFrame, pd.DataFrame]:
        updated_home_team_ratings, updated_away_team_ratings = self.update_team_ratings(home_team_ratings, away_team_ratings, ft_home_goals, ft_away_goals)
        hh_rating, ha_rating, ah_rating, aa_rating = updated_home_team_ratings['HomeRating'].values[0], updated_home_team_ratings['AwayRating'].values[0], updated_away_team_ratings['HomeRating'].values[0], updated_away_team_ratings['AwayRating'].values[0]
        updated_home_team_ratings['HomeRating'].values[0] = hh_rating
        updated_home_team_ratings['AwayRating'].values[0] = aa_rating
        updated_away_team_ratings['HomeRating'].values[0] = ah_rating
        updated_away_team_ratings['AwayRating'].values[0] = ha_rating
        return updated_home_team_ratings, updated_away_team_ratings
    

class RatingsManager:
    ALPHA = 0.999
    BETA = 0.5

    def __init__(self, data):
        self.data: pd.DataFrame = data
        self.latest_date: pd.Timestamp = self.get_latest_date()
        self.unique_teams: list[str] = self.get_unique_teams()
        self.pi_ratings: pd.DataFrame = self.initialize_ratings()
        self.pi_pairwise: pd.DataFrame = self.initialize_pairwise_ratings()
        self.pi_weighted: pd. DataFrame = self.initialize_weighted_ratings()

    def get_unique_teams(self) -> list[str]:
        home_teams = self.data['HomeTeam'].unique().tolist()
        away_teams = self.data['AwayTeam'].unique().tolist()
        return list(set(home_teams + away_teams))
    
    def get_latest_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.data['Date']).max()

    def initialize_ratings(self) -> pd.DataFrame:
        pi_ratings = pd.DataFrame(self.unique_teams, columns=['Team'])
        pi_ratings['HomeRating'] = 0.0
        pi_ratings['AwayRating'] = 0.0
        return pi_ratings

    def initialize_pairwise_ratings(self) -> pd.DataFrame:
        matchups = [
            (home, away)
            for home in self.unique_teams
            for away in self.unique_teams
            if home != away
        ]

        pairwise_pi = pd.DataFrame(matchups, columns=['HomeTeam', 'AwayTeam'])
        pairwise_pi['HomeRating'] = 0.0
        pairwise_pi['AwayRating'] = 0.0
        return pairwise_pi

    def initialize_weighted_ratings(self) -> pd.DataFrame:
        # Initialize weighted ratings with the same structure as pi_ratings
        weighted_pi = pd.DataFrame(self.unique_teams, columns=['Team'])
        weighted_pi['WeightedHomeRating'] = 0.0
        weighted_pi['WeightedAwayRating'] = 0.0
        return weighted_pi
    
    def get_decay_factor(self, match_date) -> float:
        time_diff = (self.latest_date - pd.to_datetime(match_date)).days
        return self.ALPHA ** time_diff

    def update_pi_ratings(self, row, calculator: PiRatingsCalculator):
        if pd.isnull(row['HomeTeam']):
            return

        team1 = row['HomeTeam']
        team2 = row['AwayTeam']

        # Add exponential time decay to the ratings
        decay_factor = self.get_decay_factor(row['Date'])
        calculator.set_decay_factor(decay_factor)

        # Update individual ratings
        team1_rating = self.pi_ratings.loc[self.pi_ratings['Team'] == team1]
        team2_rating = self.pi_ratings.loc[self.pi_ratings['Team'] == team2]


        updated_team1_rating, updated_team2_rating = calculator.update_team_ratings(team1_rating, team2_rating, row['FTHG'], row['FTAG'])

        self.pi_ratings.loc[self.pi_ratings['Team'] == team1] = updated_team1_rating
        self.pi_ratings.loc[self.pi_ratings['Team'] == team2] = updated_team2_rating

    def update_pi_pairwise(self, row, calculator: PiRatingsCalculator):
        if pd.isnull(row['HomeTeam']):
            return

        team1 = row['HomeTeam']
        team2 = row['AwayTeam']

        # Add exponential time decay to the ratings
        decay_factor = self.get_decay_factor(row['Date'])
        calculator.set_decay_factor(decay_factor)

        # Update pairwise ratings
        pair_rating = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == team1) & (self.pi_pairwise['AwayTeam'] == team2)]
        cross_pair_rating = self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == team2) & (self.pi_pairwise['AwayTeam'] == team1)]

        updated_pair_rating, updated_cross_pair_rating = calculator.update_team_pair_ratings(pair_rating, cross_pair_rating, row['FTHG'], row['FTAG'])

        self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == team1) & (self.pi_pairwise['AwayTeam'] == team2), \
                        ['HomeRating', 'AwayRating']] = updated_pair_rating['HomeRating'].values[0], updated_pair_rating['AwayRating'].values[0]
        self.pi_pairwise.loc[(self.pi_pairwise['HomeTeam'] == team2) & (self.pi_pairwise['AwayTeam'] == team1), \
                        ['HomeRating', 'AwayRating']] = updated_cross_pair_rating['HomeRating'].values[0], updated_cross_pair_rating['AwayRating'].values[0]

    def update_match_ratings(self, calculator: PiRatingsCalculator):
        for _, row in self.data.iterrows():
            self.update_pi_ratings(row, calculator)
            self.update_pi_pairwise(row, calculator)
            # self.update_weighted_ratings(row)
