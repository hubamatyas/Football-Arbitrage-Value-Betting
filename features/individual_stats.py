import pandas as pd

class IndividualTeamStats:
    def __init__(self, df: pd.DataFrame, unique_teams: list[str], last_n_matches=6):
        self.df = df
        self.reversed_df = self.df.iloc[::-1]
        self.unique_teams = unique_teams
        self.last_n_matches = last_n_matches

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
        self.team_last_n_matches_goals = self.init_teams_dict()
        self.team_last_n_matches_goal_diff = self.init_teams_dict()
        self.team_last_n_matches = self.init_teams_dict_with_list()

    def init_teams_dict(self) -> dict:
        return {team: 0 for team in self.unique_teams}
    
    def init_teams_set(self) -> dict:
        return {team: set() for team in self.unique_teams}
    
    def init_teams_dict_with_list(self) -> dict:
        return {team: [] for team in self.unique_teams}
    
    def compute_team_stats(self, row):
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

        if len(self.team_last_n_matches[row['HomeTeam']]) < self.last_n_matches:
            self.team_last_n_matches[row['HomeTeam']].append(row)
            self.team_last_n_matches_goals[row['HomeTeam']] += row['FTHG']
            self.team_last_n_matches_goal_diff[row['HomeTeam']] += row['FTHG'] - row['FTAG']

        if len(self.team_last_n_matches[row['AwayTeam']]) < self.last_n_matches:
            self.team_last_n_matches[row['AwayTeam']].append(row)
            self.team_last_n_matches_goals[row['AwayTeam']] += row['FTAG']
            self.team_last_n_matches_goal_diff[row['AwayTeam']] += row['FTAG'] - row['FTHG']

    def generate_features_dataframe(self) -> pd.DataFrame:
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
            'NumOfHomeMatches': [],
            'NumOfAwayMatches': [],
            'Conceded': [],
            'HalfTimeGoals': [],
            # 'ShotAccuracy': [],
            # 'GoalAccuracy': [],
            'GoalsLastNMatches': [],
            'GoalDiffLastNMatches': [],
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
            data['NumOfHomeMatches'].append(self.team_home_wins[team] + self.team_home_draws[team] + self.team_home_losses[team])
            data['NumOfAwayMatches'].append(self.team_away_wins[team] + self.team_away_draws[team] + self.team_away_losses[team])
            data['Conceded'].append(self.team_conceded[team])
            data['HalfTimeGoals'].append(self.team_half_time_goals[team])
            # data['ShotAccuracy'].append(self.team_shots_on_target[team] / self.team_shots_on_goal[team] if self.team_shots_on_goal[team] > 0 else 0)
            # data['GoalAccuracy'].append(self.team_goals[team] / self.team_shots_on_target[team] if self.team_shots_on_target[team] > 0 else 0)
            data['GoalsLastNMatches'].append(self.team_last_n_matches_goals[team])
            data['GoalDiffLastNMatches'].append(self.team_last_n_matches_goal_diff[team])

        return pd.DataFrame(data)
    
    def compute(self) -> pd.DataFrame:
        for _, row in self.reversed_df.iterrows():
            self.compute_team_stats(row)

        return self.generate_features_dataframe()