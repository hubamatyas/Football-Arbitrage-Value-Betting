import pandas as pd
from itertools import combinations
import json

class DescriptiveStats:
    def __init__(self, df: pd.DataFrame, last_matches):
        self.df = df
        self.unique_teams = self.get_unique_teams()
        self.reversed_df = self.df.iloc[::-1]
        self.num_of_last_matches = last_matches

    def get_unique_teams(self):
        teams_home = self.df['HomeTeam'].unique()
        teams_away = self.df['AwayTeam'].unique()
        unique_teams = set(teams_home).union(set(teams_away))
        return list(unique_teams)

    def get_team_matches(self, team_name):
        matches = self.reversed_df[(self.reversed_df['HomeTeam'] == team_name) | (self.reversed_df['AwayTeam'] == team_name)]
        return matches[:self.num_of_last_matches]

    def compute_team_scores(self):
        team_scores = {team: 0 for team in self.unique_teams}
        for _, row in self.df.iterrows():
            if pd.isnull(row['HomeTeam']):
                continue
            team_scores[row['HomeTeam']] += row['FTHG']
            team_scores[row['AwayTeam']] += row['FTAG']
        return team_scores

    def compute_stats(self):
        stats = {
            'average_goals': {},
            'total_goals': {},
            'avg_goal_diff': {},
            'total_goal_diff': {}
        }

        for team in self.unique_teams:
            team_df = self.get_team_matches(team)
            stats['average_goals'][team], stats['total_goals'][team] = self.compute_goals_stats(team_df, team)

        self.compute_pairwise_stats(stats)
        return stats

    def compute_goals_stats(self, team_df, team):
        total_goals = team_df.loc[team_df['HomeTeam'] == team, 'FTHG'].sum() + team_df.loc[team_df['AwayTeam'] == team, 'FTAG'].sum()
        matches_played = len(team_df)
        average_goals = total_goals / matches_played if matches_played > 0 else 0
        return average_goals, total_goals

    def compute_pairwise_stats(self, stats):
        for team1, team2 in combinations(self.unique_teams, 2):
            if team1 == team2:
                continue
            self.compute_pairwise_diff(stats, team1, team2)

    def compute_pairwise_diff(self, stats, team1, team2):
        avg_diff = stats['average_goals'][team1] - stats['average_goals'][team2]
        total_diff = stats['total_goals'][team1] - stats['total_goals'][team2]
        stats['avg_goal_diff'][f"{team1} - {team2}"] = avg_diff
        stats['total_goal_diff'][f"{team1} - {team2}"] = total_diff
        # could add more pairwise stuff here, advantages is that pairwise stats capture
        # the performance of teams against each other. this (ie. STDiff = -3) might be more descriptive than
        # having Team1ST (1) and Team2ST (4)

    def generate_features_dataframe(self, stats):
        data = {
            'Team1': [],
            'Team2': [],
            'AvgGoalDiff6Matches': list(stats['avg_goal_diff'].values()),
            'TotalGoalDiff6Matches': list(stats['total_goal_diff'].values()),
            # 'AvgShotSta': list(stats['std_dev_shots_on_goal'].values()),
            # 'AvgTargetSta': list(stats['std_dev_shots_on_target'].values()),
            # 'AvgCornerSta': list(stats['std_dev_corners'].values()),
            # 'AvgFoulSta': list(stats['std_dev_fouls'].values()),
        }
        for pair in stats['avg_goal_diff'].keys():
            team1, team2 = pair.split(' - ')
            data['Team1'].append(team1)
            data['Team2'].append(team2)

        return pd.DataFrame(data)
    

class IndividualTeamStats:
    def __init__(self, df: pd.DataFrame, unique_teams: list[str]):
        self.df = df
        self.unique_teams = unique_teams

        # TODO might be better to remove simple wins, draws, losses and
        # only keep team_home_wins, team_away_wins, team_home_draws, team_away_draws, team_home_losses, team_away_losses
        self.team_wins = self.init_teams_dict()
        self.team_fouls = self.init_teams_dict()
        self.team_draws = self.init_teams_dict()
        self.team_goals = self.init_teams_dict()
        self.team_losses = self.init_teams_dict()
        self.team_seasons = self.init_teams_set()
        self.team_corners = self.init_teams_dict()
        self.team_conceded = self.init_teams_dict()
        self.team_red_cards = self.init_teams_dict()
        self.team_home_wins = self.init_teams_dict()
        self.team_away_wins = self.init_teams_dict()
        self.team_home_draws = self.init_teams_dict()
        self.team_away_draws = self.init_teams_dict()
        self.team_home_losses = self.init_teams_dict()
        self.team_away_losses = self.init_teams_dict()
        self.team_yellow_cards = self.init_teams_dict()
        self.team_shots_on_goal = self.init_teams_dict()
        self.team_shots_on_target = self.init_teams_dict()
        self.team_half_time_goals = self.init_teams_dict()

    def init_teams_dict(self):
        return {team: 0 for team in self.unique_teams}
    
    def init_teams_set(self):
        return {team: set() for team in self.unique_teams}
    
    def compute_team_stats(self):
        for _, row in self.df.iterrows():
            if pd.isnull(row['HomeTeam']):
                continue

            if row['FTR'] == "H":
                self.team_wins[row["HomeTeam"]] += 1
                self.team_losses[row["AwayTeam"]] += 1
                self.team_home_wins[row["HomeTeam"]] += 1
                self.team_away_losses[row["AwayTeam"]] += 1
            elif row['FTR'] == "A":
                self.team_wins[row["AwayTeam"]] += 1
                self.team_losses[row["HomeTeam"]] += 1
                self.team_away_wins[row["AwayTeam"]] += 1
                self.team_home_losses[row["HomeTeam"]] += 1
            elif row['FTR'] == "D":
                self.team_draws[row['HomeTeam']] += 1
                self.team_draws[row['AwayTeam']] += 1
                self.team_home_draws[row['HomeTeam']] += 1
                self.team_away_draws[row['AwayTeam']] += 1

            self.team_goals[row['HomeTeam']] += row['FTHG']
            self.team_goals[row['AwayTeam']] += row['FTAG']

            self.team_conceded[row['HomeTeam']] += row['FTAG']
            self.team_conceded[row['AwayTeam']] += row['FTHG']

            self.team_half_time_goals[row['HomeTeam']] += row['HTHG']
            self.team_half_time_goals[row['AwayTeam']] += row['HTAG']

            self.team_shots_on_goal[row['HomeTeam']] += row['HS']
            self.team_shots_on_goal[row['AwayTeam']] += row['AS']

            self.team_shots_on_target[row['HomeTeam']] += row['HST']
            self.team_shots_on_target[row['AwayTeam']] += row['AST']

            self.team_yellow_cards[row['HomeTeam']] += row['HY']
            self.team_yellow_cards[row['AwayTeam']] += row['AY']

            self.team_red_cards[row['HomeTeam']] += row['HR']
            self.team_red_cards[row['AwayTeam']] += row['AR']

            self.team_corners[row['HomeTeam']] += row['HC']
            self.team_corners[row['AwayTeam']] += row['AC']

            self.team_fouls[row['HomeTeam']] += row['HF']
            self.team_fouls[row['AwayTeam']] += row['AF']

            self.team_seasons[row['HomeTeam']].add(row['Date'].year)
            self.team_seasons[row['AwayTeam']].add(row['Date'].year)

    def generate_features_dataframe(self):
        data = {
            'Team': [],
            'Wins': [],
            'Draws': [],
            'Losses': [],
            'HomeWins': [],
            'AwayWins': [],
            'HomeDraws': [],
            'AwayDraws': [],
            'HomeLosses': [],
            'AwayLosses': [],
            'Goals': [],
            'ShotsOnGoal': [],
            'ShotsOnTarget': [],
            'YellowCards': [],
            'RedCards': [],
            'Corners': [],
            'Fouls': [],
            'Seasons': [],
            'NumOfMatches': [],
            'Conceded': [],
            'HalfTimeGoals': [],
            'ShotAccuracy': [],
        }

        for team in self.unique_teams:
            data['Team'].append(team)
            data['Wins'].append(self.team_wins[team])
            data['Draws'].append(self.team_draws[team])
            data['Losses'].append(self.team_losses[team])
            data['HomeWins'].append(self.team_home_wins[team])
            data['AwayWins'].append(self.team_away_wins[team])
            data['HomeDraws'].append(self.team_home_draws[team])
            data['AwayDraws'].append(self.team_away_draws[team])
            data['HomeLosses'].append(self.team_home_losses[team])
            data['AwayLosses'].append(self.team_away_losses[team])
            data['Goals'].append(self.team_goals[team])
            data['ShotsOnGoal'].append(self.team_shots_on_goal[team])
            data['ShotsOnTarget'].append(self.team_shots_on_target[team])
            data['YellowCards'].append(self.team_yellow_cards[team])
            data['RedCards'].append(self.team_red_cards[team])
            data['Corners'].append(self.team_corners[team])
            data['Fouls'].append(self.team_fouls[team])
            data['Seasons'].append(len(self.team_seasons[team]))
            data['NumOfMatches'].append(self.team_wins[team] + self.team_draws[team] + self.team_losses[team])
            data['Conceded'].append(self.team_conceded[team])
            data['HalfTimeGoals'].append(self.team_half_time_goals[team])
            data['ShotAccuracy'].append(self.team_shots_on_target[team] / self.team_shots_on_goal[team] if self.team_shots_on_goal[team] > 0 else 0)

        return pd.DataFrame(data)
