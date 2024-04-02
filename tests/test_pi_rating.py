import pandas as pd
import math

from data.load_csv import DataLoader
from data.process import DataProcessor
from features.pi_rating import PiRatingsCalculator, RatingsManager
from data.utils import Season

def test_calc_pi_ratings_update():
    dataset_path = 'epl-training.csv'
    season = Season.Past5
    data_loader = DataLoader(dataset_path, season)
    data_loader.read_csv()
    data_loader.parse_df()
    data_loader.select_seasons(season)
    df = data_loader.get_df()
    data_processor = DataProcessor(df)
    calculator = PiRatingsCalculator()
    manager = RatingsManager(df)
    manager.update_match_ratings(calculator)

    sample_pi_rating = manager.pi_ratings.loc[(manager.pi_ratings['Team'] == 'Man United') | (manager.pi_ratings['Team'] == 'Man City')]

    # 0.651034    0.346496
    # 0.951109    0.649680

    assert round(sample_pi_rating['HomeRating'].values[0], 6) == round(0.651034, 6)
    assert round(sample_pi_rating['AwayRating'].values[0], 6) == round(0.346496, 6)
    assert round(sample_pi_rating['HomeRating'].values[1], 6) == round(0.951109, 6)
    assert round(sample_pi_rating['AwayRating'].values[1], 6) == round(0.649680, 6)

    del data_loader
    del data_processor
    del calculator
    del manager

def test_calc_pairwise_pi_update():
    dataset_path = 'epl-training.csv'
    season = Season.Past5
    data_loader = DataLoader(dataset_path, season)
    data_loader.read_csv()
    data_loader.parse_df()
    data_loader.select_seasons(season)
    df = data_loader.get_df()
    data_processor = DataProcessor(df)
    calculator = PiRatingsCalculator()
    manager = RatingsManager(df)
    manager.update_match_ratings(calculator)

    sample_pi_pairwise = manager.pi_pairwise.loc[(manager.pi_pairwise['HomeTeam'] == 'Man United') & (manager.pi_pairwise['AwayTeam'] == 'Man City') | (manager.pi_pairwise['HomeTeam'] == 'Man City') & (manager.pi_pairwise['AwayTeam'] == 'Man United')]

    # 0.370189   -0.513256
    # -0.366265    0.509333

    assert round(sample_pi_pairwise['HomeRating'].values[0], 6) == round(0.370189, 6)
    assert round(sample_pi_pairwise['AwayRating'].values[0], 6) == round(-0.513256, 6)
    assert round(sample_pi_pairwise['HomeRating'].values[1], 6) == round(-0.366265, 6)
    assert round(sample_pi_pairwise['AwayRating'].values[1], 6) == round(0.509333, 6)

    del data_loader
    del data_processor
    del calculator
    del manager