import pandas as pd


class IndividualTeamStats:
    def __init__(self, df: pd.DataFrame, unique_teams: list[str], last_n_matches=6):
        self.df = df
        self.reversed_df = self.df.iloc[::-1]
        self.unique_teams = unique_teams
        self.last_n_matches = last_n_matches

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
        self.team_possession = self.init_teams_dict()
        self.team_win_streak = self.init_teams_dict()
        self.team_home_draws = self.init_teams_dict()
        self.team_away_draws = self.init_teams_dict()
        self.team_home_losses = self.init_teams_dict()
        self.team_away_losses = self.init_teams_dict()
        self.team_yellow_cards = self.init_teams_dict()
        self.team_shots_on_goal = self.init_teams_dict()
        self.team_shots_on_target = self.init_teams_dict()
        self.team_half_time_goals = self.init_teams_dict()
        self.team_last_n_matches_goals = self.init_teams_dict_with_list()
        self.team_last_n_matches_goal_diff = self.init_teams_dict_with_list()

    def init_teams_dict(self) -> dict:
        return {team: 0 for team in self.unique_teams}
    
    def init_teams_set(self) -> dict:
        return {team: set() for team in self.unique_teams}
    
    def init_teams_dict_with_list(self) -> dict:
        return {team: [] for team in self.unique_teams}
    
    def compute_team_stats(self, row):
        self.update_results(row)
        self.update_home_away_results(row)
        self.update_win_streak(row)
        self.update_goals(row)
        self.update_conceded_goals(row)
        self.update_half_time_goals(row)
        self.update_shots_on_goal(row)
        self.update_shots_on_target(row)
        self.update_yellow_cards(row)
        self.update_red_cards(row)
        self.update_corners(row)
        self.update_posession(row)
        self.update_fouls(row)
        self.update_seasons(row)
        self.update_last_n_matches(row)

    def update_results(self, row):
        if row['FTR'] == "H":
            self.team_wins[row['HomeTeam']] += 1
            self.team_losses[row['AwayTeam']] += 1
        elif row['FTR'] == "A":
            self.team_wins[row['AwayTeam']] += 1
            self.team_losses[row['HomeTeam']] += 1
        elif row['FTR'] == "D":
            self.team_draws[row['HomeTeam']] += 1
            self.team_draws[row['AwayTeam']] += 1

    def update_home_away_results(self, row):
        if row['FTR'] == "H":
            self.team_home_wins[row['HomeTeam']] += 1
            self.team_away_losses[row['AwayTeam']] += 1
        elif row['FTR'] == "A":
            self.team_away_wins[row['AwayTeam']] += 1
            self.team_home_losses[row['HomeTeam']] += 1
        elif row['FTR'] == "D":
            self.team_home_draws[row['HomeTeam']] += 1
            self.team_away_draws[row['AwayTeam']] += 1
    
    def update_win_streak(self, row):
        if row['FTR'] == "H":
            self.team_win_streak[row['HomeTeam']] += 1
            self.team_win_streak[row['AwayTeam']] = 0
        elif row['FTR'] == "A":
            self.team_win_streak[row['AwayTeam']] += 1
            self.team_win_streak[row['HomeTeam']] = 0
        elif row['FTR'] == "D":
            self.team_win_streak[row['HomeTeam']] = 0
            self.team_win_streak[row['AwayTeam']] = 0

    def update_goals(self, row):
        self.team_goals[row['HomeTeam']] += row['FTHG']
        self.team_goals[row['AwayTeam']] += row['FTAG']

    def update_conceded_goals(self, row):
        self.team_conceded[row['HomeTeam']] += row['FTAG']
        self.team_conceded[row['AwayTeam']] += row['FTHG']

    def update_half_time_goals(self, row):
        self.team_half_time_goals[row['HomeTeam']] += row['HTHG']
        self.team_half_time_goals[row['AwayTeam']] += row['HTAG']

    def update_shots_on_goal(self, row):
        self.team_shots_on_goal[row['HomeTeam']] += row['HS']
        self.team_shots_on_goal[row['AwayTeam']] += row['AS']

    def update_shots_on_target(self, row):
        self.team_shots_on_target[row['HomeTeam']] += row['HST']
        self.team_shots_on_target[row['AwayTeam']] += row['AST']

    def update_yellow_cards(self, row):
        self.team_yellow_cards[row['HomeTeam']] += row['HY']
        self.team_yellow_cards[row['AwayTeam']] += row['AY']

    def update_red_cards(self, row):
        self.team_red_cards[row['HomeTeam']] += row['HR']
        self.team_red_cards[row['AwayTeam']] += row['AR']

    def update_corners(self, row):
        self.team_corners[row['HomeTeam']] += row['HC']
        self.team_corners[row['AwayTeam']] += row['AC']

    def update_posession(self, row):
        self.team_possession[row['HomeTeam']] += row['HP']
        self.team_possession[row['AwayTeam']] += row['AP']

    def update_fouls(self, row):
        self.team_fouls[row['HomeTeam']] += row['HF']
        self.team_fouls[row['AwayTeam']] += row['AF']

    def update_seasons(self, row):
        self.team_seasons[row['HomeTeam']].add(row['Date'].year)
        self.team_seasons[row['AwayTeam']].add(row['Date'].year)

    def update_last_n_matches(self, row):
        if len(self.team_last_n_matches_goals[row['HomeTeam']]) >= self.last_n_matches:
            self.team_last_n_matches_goals[row['HomeTeam']].pop(0)
            self.team_last_n_matches_goal_diff[row['HomeTeam']].pop(0)

        if len(self.team_last_n_matches_goals[row['AwayTeam']]) >= self.last_n_matches:
            self.team_last_n_matches_goals[row['AwayTeam']].pop(0)
            self.team_last_n_matches_goal_diff[row['AwayTeam']].pop(0)

        self.team_last_n_matches_goals[row['HomeTeam']].append(row['FTHG'])
        self.team_last_n_matches_goals[row['AwayTeam']].append(row['FTAG'])
        self.team_last_n_matches_goal_diff[row['HomeTeam']].append(row['FTHG'] - row['FTAG'])
        self.team_last_n_matches_goal_diff[row['AwayTeam']].append(row['FTAG'] - row['FTHG']) 

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
            'GoalsLastNMatches': [],
            'GoalDiffLastNMatches': [],
            'WinStreak': [],
            'Possession': [],
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
            data['GoalsLastNMatches'].append(sum(self.team_last_n_matches_goals[team]))
            data['GoalDiffLastNMatches'].append(sum(self.team_last_n_matches_goal_diff[team]))
            data['WinStreak'].append(self.team_win_streak[team])
            data['Possession'].append(self.team_possession[team])

        return pd.DataFrame(data)
    
    def compute(self) -> pd.DataFrame:
        for _, row in self.reversed_df.iterrows():
            self.compute_team_stats(row)

        return self.generate_features_dataframe()