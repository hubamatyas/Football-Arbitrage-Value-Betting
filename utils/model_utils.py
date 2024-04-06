# 'is_goal_stats': True,
# 'is_shooting_stats': True,
# 'is_result': True,
# 'is_home_away_results': True,
# 'is_conceded_stats': True,
# 'is_last_n_matches': True,
# 'is_win_streak': True,
# 'is_pairwise_stats': False,
# 'is_pi_ratings': True,
# 'is_pi_pairwise': False,
# 'is_pi_weighted': False

from enum import Enum

class Feature(Enum):
    GOAL_STATS = 'is_goal_stats'
    SHOOTING_STATS = 'is_shooting_stats'
    RESULT = 'is_result'
    HOME_AWAY_RESULTS = 'is_home_away_results'
    CONCEDED_STATS = 'is_conceded_stats'
    LAST_N_MATCHES = 'is_last_n_matches'
    WIN_STREAK = 'is_win_streak'
    PAIRWISE_STATS = 'is_pairwise_stats'
    PI_RATINGS = 'is_pi_ratings'
    PI_PAIRWISE = 'is_pi_pairwise'
    PI_WEIGHTED = 'is_pi_weighted'