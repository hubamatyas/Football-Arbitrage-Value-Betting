import math
import pandas as pd

# Construct the Pi-ratings of teams from their performance in the training dataset. The specific equations can be found in the report.
def calc_pi_rating(home_team_ratings: pd.DataFrame, away_team_ratings: pd.DataFrame, ft_home_goals, ft_away_goals, decay_factor):
    C = 3
    LOG_BASE = 10
    # Hyperparameters set after tuning
    LAMBDA = 0.156
    GAMMA = 0.739

    # Initialise the updated ratings to the current ratings
    updated_home_team_ratings = home_team_ratings.copy()
    updated_away_team_ratings = away_team_ratings.copy()

    home_team_sign = 1 if home_team_ratings['HomeRating'].values[0] >= 0 else -1
    away_team_sign = 1 if away_team_ratings['AwayRating'].values[0] >= 0 else -1

    # Calculate the expected goal difference based on team's current ratings
    exp_home_goal_diff = home_team_sign * (LOG_BASE ** (abs(home_team_ratings['HomeRating'].values[0]) / C) - 1)
    exp_away_goal_diff = away_team_sign * (LOG_BASE ** (abs(away_team_ratings['AwayRating'].values[0]) / C) - 1)

    exp_goal_diff = exp_home_goal_diff - exp_away_goal_diff
    act_goal_diff = ft_home_goals - ft_away_goals

    # Error between expected and actual goal difference
    error = abs(act_goal_diff - exp_goal_diff)

    # Calculate by how much ratings should be updated
    if act_goal_diff > exp_goal_diff:
        psi_H_e = C * math.log10(error + 1)
        psi_A_e = -C * math.log10(error + 1)
    else:
        psi_H_e = -C * math.log10(error + 1)
        psi_A_e = C * math.log10(error + 1)

    # Expontential time decay to weight more recent matches more heavily
    psi_H_e *= decay_factor
    psi_A_e *= decay_factor

    # Update the ratings
    updated_home_team_ratings['HomeRating'].values[0] = home_team_ratings['HomeRating'].values[0] + psi_H_e * LAMBDA
    updated_away_team_ratings['AwayRating'].values[0] = away_team_ratings['AwayRating'].values[0] + psi_A_e * LAMBDA

    updated_home_team_ratings['AwayRating'].values[0] = home_team_ratings['AwayRating'].values[0] + (updated_home_team_ratings['HomeRating'].values[0] - home_team_ratings['HomeRating'].values[0]) * GAMMA
    updated_away_team_ratings['HomeRating'].values[0] = away_team_ratings['HomeRating'].values[0] + (updated_away_team_ratings['AwayRating'].values[0] - away_team_ratings['AwayRating'].values[0]) * GAMMA

    return updated_home_team_ratings, updated_away_team_ratings


# Evaluate how teams fare against each other in both home and away contexts by calculating unique ratings for every possible pair of teams. This takes into account the specific dynamics of each head-to-head encounter.
def calc_pair_pi_rating(home_team_ratings: pd.DataFrame, away_team_ratings: pd.DataFrame, ft_home_goals, ft_away_goals, decay_factor):
    updated_home_team_ratings,  updated_away_team_ratings = calc_pi_rating(home_team_ratings, away_team_ratings, ft_home_goals, ft_away_goals, decay_factor)
    hh_rating, ha_rating, ah_rating, aa_rating = updated_home_team_ratings['HomeRating'].values[0], updated_home_team_ratings['AwayRating'].values[0], updated_away_team_ratings['HomeRating'].values[0], updated_away_team_ratings['AwayRating'].values[0]
    updated_home_team_ratings['HomeRating'].values[0] = hh_rating
    updated_home_team_ratings['AwayRating'].values[0] = aa_rating
    updated_away_team_ratings['HomeRating'].values[0] = ah_rating
    updated_away_team_ratings['AwayRating'].values[0] = ha_rating
    return updated_home_team_ratings, updated_away_team_ratings


# Iterate over the dataset to calculate and return the ratings
def get_ratings(data: pd.DataFrame, unique_teams: list):
    ALPHA = 0.999
    BETA = 0.5

    # Initialise ratings to 0 and get the latest date in the dataset for time decay
    latest_date = pd.to_datetime(data['Date']).max()
    pi_ratings = pd.DataFrame(unique_teams, columns=['Team'])
    pi_ratings['HomeRating'] = 0.0
    pi_ratings['AwayRating'] = 0.0

    # Create a dataframe to store pairwise ratings
    matchups = [
        (home_team, away_team)
        for home_team in pi_ratings['Team']
        for away_team in pi_ratings['Team']
        if (home_team != away_team and pd.notna(home_team) and pd.notna(away_team))
    ]

    pairwise_pi = pd.DataFrame(matchups, columns=['HomeTeam', 'AwayTeam'])
    pairwise_pi['HomeRating'] = 0.0
    pairwise_pi['AwayRating'] = 0.0

    # Iterate over the dataset to calculate the ratings
    for index, row in data.iterrows():

        if pd.isnull(row['HomeTeam']):
            continue

        team1 = row['HomeTeam']
        team2 = row['AwayTeam']

        # Add exponential time decay to the ratings
        time_diff = (latest_date - pd.to_datetime(row['Date'])).days
        decay_factor = ALPHA ** time_diff

        # Update individual ratings
        team1_rating = pi_ratings.loc[pi_ratings['Team'] == team1]
        team2_rating = pi_ratings.loc[pi_ratings['Team'] == team2]

        # Update pairwise ratings
        pair_rating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team1) & (pairwise_pi['AwayTeam'] == team2)]
        cross_pair_rating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team2) & (pairwise_pi['AwayTeam'] == team1)]

        updated_team1_rating, updated_team2_rating = calc_pi_rating(team1_rating, team2_rating, row['FTHG'], row['FTAG'], decay_factor)
        updated_pair_rating, updated_cross_pair_rating = calc_pair_pi_rating(pair_rating, cross_pair_rating, row['FTHG'], row['FTAG'], decay_factor)

        pi_ratings.loc[pi_ratings['Team'] == team1] = updated_team1_rating
        pi_ratings.loc[pi_ratings['Team'] == team2] = updated_team2_rating
        pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team1) & (pairwise_pi['AwayTeam'] == team2), \
                        ['HomeRating', 'AwayRating']] = updated_pair_rating['HomeRating'].values[0], updated_pair_rating['AwayRating'].values[0]
        pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team2) & (pairwise_pi['AwayTeam'] == team1), \
                        ['HomeRating', 'AwayRating']] = updated_cross_pair_rating['HomeRating'].values[0], updated_cross_pair_rating['AwayRating'].values[0]

    # Create another dataframe to store weighted ratings where the HomeRating for HomeTeam is the weighted average of pairwise ratings of HomeTeam HomeRating and pi_ratings HomeTeam HomeRating
    weighted_pairwise_pi = pairwise_pi.copy()

    # Iterate over the pairwise ratings to calculate the weighted ratings of pairwise and individual ratings
    for index, row in weighted_pairwise_pi.iterrows():
        team1 = row['HomeTeam']
        team2 = row['AwayTeam']

        # HomeTeam home, HomeTeam away, AwayTeam home, AwayTeam away
        ht_homerating = pi_ratings.loc[pi_ratings['Team'] == team1]['HomeRating'].values[0]
        ht_awayrating = pi_ratings.loc[pi_ratings['Team'] == team1]['AwayRating'].values[0]
        at_homerating = pi_ratings.loc[pi_ratings['Team'] == team2]['HomeRating'].values[0]
        at_awayrating = pi_ratings.loc[pi_ratings['Team'] == team2]['AwayRating'].values[0]

        # Pairwise HomeTeam home, Pairwise HomeTeam away, Pairwise AwayTeam home, Pairwise AwayTeam away
        pwht_homerating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team1) & (pairwise_pi['AwayTeam'] == team2)]['HomeRating'].values[0]
        pwht_awayrating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team2) & (pairwise_pi['AwayTeam'] == team1)]['AwayRating'].values[0]
        pwat_homerating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team2) & (pairwise_pi['AwayTeam'] == team1)]['HomeRating'].values[0]
        pwat_awayrating = pairwise_pi.loc[(pairwise_pi['HomeTeam'] == team1) & (pairwise_pi['AwayTeam'] == team2)]['AwayRating'].values[0]

        # Weighted pairwise HomeTeam home, Weighted pairwise HomeTeam away, Weighted pairwise AwayTeam home, Weighted pairwise AwayTeam away
        wpwht_homerating = ht_homerating * BETA + pwht_homerating
        wpwht_awayrating = ht_awayrating * BETA + pwht_awayrating
        wpwat_homerating = at_homerating * BETA + pwat_homerating
        wpwat_awayrating = at_awayrating * BETA + pwat_awayrating

        weighted_pairwise_pi.loc[(weighted_pairwise_pi['HomeTeam'] == team1) & (weighted_pairwise_pi['AwayTeam'] == team2), \
                        ['HomeRating', 'AwayRating']] = wpwht_homerating, wpwat_awayrating
        weighted_pairwise_pi.loc[(weighted_pairwise_pi['HomeTeam'] == team2) & (weighted_pairwise_pi['AwayTeam'] == team1), \
                        ['HomeRating', 'AwayRating']] = wpwat_homerating, wpwht_awayrating

    return (pi_ratings, pairwise_pi, weighted_pairwise_pi)
