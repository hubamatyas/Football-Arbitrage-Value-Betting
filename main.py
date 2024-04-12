import model
from utils.data_utils import Season
from utils.model_utils import Feature
from data.load_footystats import GenerateDataFrame
from data.load_csv import DataLoader

if __name__ == '__main__':
    season = Season.Past1
    df = GenerateDataFrame(season=season).load()
    params = {
        Feature.GOAL_STATS.value: False,
        Feature.SHOOTING_STATS.value: False,
        Feature.POSSESSION_STATS.value: False,
        Feature.ODDS.value: False,
        Feature.XG.value: False,
        Feature.HOME_AWAY_RESULTS.value: False,
        Feature.CONCEDED_STATS.value: False,
        Feature.LAST_N_MATCHES.value: False,
        Feature.WIN_STREAK.value: False,
        Feature.PAIRWISE_STATS.value: False,
        Feature.PI_RATINGS.value: True,
        Feature.PI_PAIRWISE.value: False,
        Feature.PI_WEIGHTED.value: False
    }
    # df = DataLoader('epl-training.csv', season).load()
    model.run(df=df, feature_params=params)