import pandas as pd
from features.individual_stats import IndividualTeamStats

class PairwiseTeamStats:
    def __init__(self, df: pd.DataFrame, unique_teams: list[str], individual_stats: IndividualTeamStats):
        self.df = df
        self.unique_teams = unique_teams
        self.individual_stats = individual_stats
        self.team_last_n_matches_goals = individual_stats[['Team', 'GoalsLastNMatches']].set_index('Team').to_dict()['GoalsLastNMatches']
        self.team_last_n_matches_goal_diff = individual_stats[['Team', 'GoalDiffLastNMatches']].set_index('Team').to_dict()['GoalDiffLastNMatches']

        self.pairwise_home_team_wins = self.init_pairwise_dict()
        self.pairwise_home_team_draws = self.init_pairwise_dict()
        self.pairwise_home_team_losses = self.init_pairwise_dict()
        self.pairwise_away_team_wins = self.init_pairwise_dict()
        self.pairwise_away_team_draws = self.init_pairwise_dict()
        self.pairwise_away_team_losses = self.init_pairwise_dict()
        self.pairwise_home_team_goals = self.init_pairwise_dict()
        self.pairwise_away_team_goals = self.init_pairwise_dict()
        self.pairwise_home_team_conceded = self.init_pairwise_dict()
        self.pairwise_away_team_conceded = self.init_pairwise_dict()
        self.pairwise_home_team_shots_on_goal = self.init_pairwise_dict()
        self.pairwise_away_team_shots_on_goal = self.init_pairwise_dict()
        self.pairwise_home_team_shots_on_target = self.init_pairwise_dict()
        self.pairwise_away_team_shots_on_target = self.init_pairwise_dict()
        self.pairwise_home_team_corners = self.init_pairwise_dict()
        self.pairwise_away_team_corners = self.init_pairwise_dict()
        self.pairwise_home_team_half_time_goals = self.init_pairwise_dict()
        self.pairwise_away_team_half_time_goals = self.init_pairwise_dict()

        # Used to provide account of the recent performance of the teams
        # The last n matches doesn't look at the last n matches between
        # the two teams but simply the last n matches of the respective teams
        self.pairwise_goal_diff_last_n_matches = self.init_pairwise_dict()
        self.pairwise_total_goals_diff_last_n_matches = self.init_pairwise_dict()

    def init_pairwise_dict(self):
        matchups = [
            (home, away)
            for home in self.unique_teams
            for away in self.unique_teams
            if home != away
            ]
         
        return {pair: 0 for pair in matchups}
    
    def compute_pairwise_stats(self):
        for _, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            pair = (home_team, away_team)

            self.pairwise_home_team_goals[pair] += row['FTHG']
            self.pairwise_away_team_goals[pair] += row['FTAG']

            self.pairwise_home_team_conceded[pair] += row['FTAG']
            self.pairwise_away_team_conceded[pair] += row['FTHG']

            self.pairwise_home_team_shots_on_goal[pair] += row['HS']
            self.pairwise_away_team_shots_on_goal[pair] += row['AS']

            self.pairwise_home_team_shots_on_target[pair] += row['HST']
            self.pairwise_away_team_shots_on_target[pair] += row['AST']

            self.pairwise_home_team_corners[pair] += row['HC']
            self.pairwise_away_team_corners[pair] += row['AC']

            self.pairwise_home_team_half_time_goals[pair] += row['HTHG']
            self.pairwise_away_team_half_time_goals[pair] += row['HTAG']

            if row['FTR'] == "H":
                self.pairwise_home_team_wins[pair] += 1
                self.pairwise_home_team_losses[pair] += 1
            elif row['FTR'] == "A":
                self.pairwise_away_team_wins[pair] += 1
                self.pairwise_away_team_losses[pair] += 1
            elif row['FTR'] == "D":
                self.pairwise_home_team_draws[pair] += 1
                self.pairwise_away_team_draws[pair] += 1

    def compute_pairwise_goal_diff(self):
        for pair in self.pairwise_home_team_goals.keys():
            home_team, away_team = pair
            self.pairwise_goal_diff_last_n_matches[pair] = self.team_last_n_matches_goal_diff[home_team] - self.team_last_n_matches_goal_diff[away_team]
            self.pairwise_total_goals_diff_last_n_matches[pair] = self.team_last_n_matches_goals[home_team] - self.team_last_n_matches_goals[away_team]

    def generate_features_dataframe(self):
        data = {
            'HomeTeam': [],
            'AwayTeam': [],
            'HomeWins': [],
            'AwayWins': [],
            'HomeDraws': [],
            'AwayDraws': [],
            'HomeLosses': [],
            'AwayLosses': [],
            'HomeGoals': [],
            'AwayGoals': [],
            'HomeShotsOnGoal': [],
            'AwayShotsOnGoal': [],
            'HomeShotsOnTarget': [],
            'AwayShotsOnTarget': [],
            'HomeCorners': [],
            'AwayCorners': [],
            'HomeConceded': [],
            'AwayConceded': [],
            'HomeHalfTimeGoals': [],
            'AwayHalfTimeGoals': [],
            'GoalDiffLastNMatches': [],
            'TotalGoalsDiffLastNMatches': [],
        }

        for pair in self.pairwise_home_team_wins.keys():
            home_team, away_team = pair
            data['HomeTeam'].append(home_team)
            data['AwayTeam'].append(away_team)
            data['HomeWins'].append(self.pairwise_home_team_wins[pair])
            data['AwayWins'].append(self.pairwise_away_team_wins[pair])
            data['HomeDraws'].append(self.pairwise_home_team_draws[pair])
            data['AwayDraws'].append(self.pairwise_away_team_draws[pair])
            data['HomeLosses'].append(self.pairwise_home_team_losses[pair])
            data['AwayLosses'].append(self.pairwise_away_team_losses[pair])
            data['HomeGoals'].append(self.pairwise_home_team_goals[pair])
            data['AwayGoals'].append(self.pairwise_away_team_goals[pair])
            data['HomeShotsOnGoal'].append(self.pairwise_home_team_shots_on_goal[pair])
            data['AwayShotsOnGoal'].append(self.pairwise_away_team_shots_on_goal[pair])
            data['HomeShotsOnTarget'].append(self.pairwise_home_team_shots_on_target[pair])
            data['AwayShotsOnTarget'].append(self.pairwise_away_team_shots_on_target[pair])
            data['HomeCorners'].append(self.pairwise_home_team_corners[pair])
            data['AwayCorners'].append(self.pairwise_away_team_corners[pair])
            data['HomeConceded'].append(self.pairwise_home_team_conceded[pair])
            data['AwayConceded'].append(self.pairwise_away_team_conceded[pair])
            data['HomeHalfTimeGoals'].append(self.pairwise_home_team_half_time_goals[pair])
            data['AwayHalfTimeGoals'].append(self.pairwise_away_team_half_time_goals[pair])
            data['GoalDiffLastNMatches'].append(self.pairwise_goal_diff_last_n_matches[pair])
            data['TotalGoalsDiffLastNMatches'].append(self.pairwise_total_goals_diff_last_n_matches[pair])

        return pd.DataFrame(data)
    
    def compute(self):
        self.compute_pairwise_stats()
        self.compute_pairwise_goal_diff()
        return self.generate_features_dataframe()